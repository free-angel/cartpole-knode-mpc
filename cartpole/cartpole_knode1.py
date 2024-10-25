import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import mpc1
from mpc1 import mpc
from mpc1.mpc import QuadCost, LinDx, GradMethods

from env_dx import cartpole

from torchdiffeq import odeint

from nn_models1 import NominalModel, UncertaintyModel, HybridModel



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lossMSE = nn.MSELoss()
num_traj = 1 #number of training trajectories
data_size = 60
niters = 200
batch_time = 2
test_freq = 10

#version with more than one training trajectory
theta_array = (1 / 3) * np.pi +  (1 / 3) * np.pi * torch.rand(num_traj)
theta_dot_array = (1 / 4) * np.pi - (1 / 2) * np.pi * torch.rand(num_traj)

t = torch.linspace(0., 3., data_size).to(device)

class TrueCartpoleDynamics(nn.Module):
    def __init__(self, p=torch.tensor((9.8, 0.98, 0.102, 0.51)), f=20., theta_0=np.pi, theta_dot_0=-np.pi/2):
        super(TrueCartpoleDynamics, self).__init__()
        self.cartpole_model = cartpole.CartpoleDx(params=p)
        self.cartpole_model.force_mag = f
        self.u_init = None

        # uncertainties
        self.friction_coeff = 0.1
        self.noise_level = 0.01

        self.nominal_y0 = torch.tensor([0, 0, np.cos(theta_0), np.sin(theta_0), theta_dot_0], dtype=torch.float32)

        # self.mpc_T = 25 #pred horizon
        self.mpc_T = 25 #pred horizon

        self.n_batch = 1
        self.q, self.p = self.cartpole_model.get_true_obj()
        self.Q = torch.diag(self.q).unsqueeze(0).unsqueeze(0).repeat(
            self.mpc_T, self.n_batch, 1, 1
        )
        self.p = self.p.unsqueeze(0).repeat(self.mpc_T, self.n_batch, 1)

        self.liqr = mpc.MPC(
            self.cartpole_model.n_state, self.cartpole_model.n_ctrl, self.mpc_T,
            u_init=self.u_init,
            u_lower=self.cartpole_model.lower, u_upper=self.cartpole_model.upper,
            lqr_iter=50,
            verbose=0,
            exit_unconverged=False,
            detach_unconverged=False,
            linesearch_decay=self.cartpole_model.linesearch_decay,
            max_linesearch_iter=self.cartpole_model.max_linesearch_iter,
            grad_method=GradMethods.AUTO_DIFF,
            eps=1e-2,
        )
        

    def forward(self, t, y):
        print(t)
        state = y.unsqueeze(0) if y.ndimension() == 1 else y
        nominal_states, nominal_actions, nominal_objs = self.liqr(state, QuadCost(self.Q, self.p), self.cartpole_model)
        next_action = nominal_actions[0]
        # action_history.append(next_action)
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, self.n_batch, self.cartpole_model.n_ctrl)), dim=0)
        u_init[-2] = u_init[-3]
        state = self.cartpole_model(state, next_action).squeeze()

        self.liqr.u_init = u_init

        #introduce uncertainties
        # apply friction to xdot
        state[1] -= self.friction_coeff * state[1]
        # add random noise
        noise = torch.randn_like(state) * self.noise_level
        state += noise
        # 把控制变量收集起来。

        print(f"True Force: {next_action.item():.2f}N, cos theta: {state[2]:.2f}, sine theta: {state[3]:.2f}")
        return (state - y) / self.cartpole_model.dt


true_y_list = []
for idx in range(num_traj):
    print(idx, theta_array[idx], theta_dot_array[idx])
    true_y0 = torch.tensor([0, 0, np.cos(theta_array[idx]), np.sin(theta_array[idx]), theta_dot_array[idx]], dtype=torch.float32).to(device)
    true_y = odeint(TrueCartpoleDynamics(theta_0=theta_array[idx], theta_dot_0=theta_dot_array[idx]),
                    true_y0, t, method='rk4', options=dict(step_size=0.05))
    
    true_y_list.append(true_y)

def get_batch(t, true_y, data_size, num_traj, batch_time, device):
    s           = torch.arange(data_size - 1)
    batch_y0    = true_y[s]
    batch_t     = t[:batch_time]
    batch_y     = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


if __name__ == '__main__':
    nominal_model = NominalModel().to(device)
    uncertainty_model = UncertaintyModel(state_dim=5).to(device)
    hybrid_model = HybridModel(nominal_model, uncertainty_model).to(device)

    optimizer = optim.Rprop(hybrid_model.parameters(), lr=1e-4)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    lowest_loss_so_far  = 1e5
    batch_y0_list       = []
    batch_t_list        = []
    batch_y_list        = []

    for idx in range(num_traj):
        batch_y0, batch_t, batch_y  = get_batch(t, true_y_list[idx], data_size, num_traj, batch_time, device)
        batch_y0_list.append(batch_y0)
        batch_t_list.append(batch_t)
        batch_y_list.append(batch_y)

        # print(batch_y0.size(), batch_t.size(), batch_y.size())

    for itr in tqdm.tqdm(range(1, niters + 1)):
        optimizer.zero_grad()
        loss = 0.0

        for idx in range(num_traj):
            pred_y     = odeint(hybrid_model, batch_y0_list[idx], batch_t_list[idx], method='rk4', options=dict(step_size=0.05)).to(device)
            loss      += lossMSE(pred_y[1, :, :], batch_y_list[idx][1, :, :])

        loss.backward(retain_graph=True)
        optimizer.step()

        # time_meter.update(time.time() - end)
        # loss_meter.update(loss.item())

        if itr % test_freq == 0:
            print('Iter {:4d} | Training Loss {:e}'.format(itr, loss.item()))


        end = time.time()

    torch.save(hybrid_model.state_dict(), 'hybrid_model_cartpole.pth')


