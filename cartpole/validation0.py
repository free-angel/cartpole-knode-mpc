import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


import numpy as np
from torchdiffeq import odeint


import mpc1
from mpc1 import mpc
from mpc1.mpc import QuadCost, LinDx, GradMethods
from env_dx import cartpole
from nn_models1 import NominalModel, UncertaintyModel, HybridModel

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# initial conditions
num_val_traj = 8
data_size = 60
t = torch.linspace(0., 3., data_size).to(device)
theta_val_array = (1 / 3) * np.pi + (1 / 3) * np.pi * torch.rand(num_val_traj)
theta_dot_val_array = (1 / 4) * np.pi - (1 / 2) * np.pi * torch.rand(num_val_traj)

# load trained hybrid model
nominal = NominalModel(n_batch=1)

hybrid_model = HybridModel(nominal.to(device), UncertaintyModel(state_dim=5).to(device))
hybrid_model.load_state_dict(torch.load('hybrid_model_cartpole.pth'))
hybrid_model.eval()  # Set the model to evaluation mode


true_y_val_list = []
pred_y_val_list = []

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

        print(f"True Force: {next_action.item():.2f}N, cos theta: {state[2]:.2f}, sine theta: {state[3]:.2f}")
        return (state - y) / self.cartpole_model.dt
    

def get_batch(t, true_y, data_size, num_traj, batch_time, device):
    s           = torch.arange(data_size - 1)
    batch_y0    = true_y[s]
    batch_t     = t[:batch_time]
    batch_y     = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)

    print(batch_y0.size())
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

for idx in range(num_val_traj):
    true_y0_val = torch.tensor([0, 0, np.cos(theta_val_array[idx]), np.sin(theta_val_array[idx]), theta_dot_val_array[idx]], dtype=torch.float32).to(device)
    
    # simulate true dynamics
    true_y_val = odeint(TrueCartpoleDynamics(theta_0=theta_val_array[idx], theta_dot_0=theta_dot_val_array[idx]), true_y0_val, t, method='rk4', options=dict(step_size=0.05))
    true_y_val_list.append(true_y_val)

    # generate predictions
    # with torch.no_grad():
    pred_y_val = odeint(hybrid_model, true_y0_val, t, method='rk4', options=dict(step_size=0.05))
    pred_y_val_list.append(pred_y_val)

    mse_loss_val = mse_loss(pred_y_val, true_y_val)
    print(f"Trajectory {idx+1} MSE Loss: {mse_loss_val.item()}")


# visualize results
import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to TkAgg
import matplotlib.pyplot as plt


# for idx in range(num_val_traj):
#     true_y_val = true_y_val_list[idx].cpu().detach().numpy()
#     pred_y_val = pred_y_val_list[idx].cpu().detach().numpy()
    
#     plt.figure(figsize=(12, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(t.cpu().detach().numpy(), true_y_val[:, 0, 0], label='True x')
#     plt.plot(t.cpu().detach().numpy(), pred_y_val[:, 0, 0], label='Predicted x', linestyle='--')
#     plt.title(f'Trajectory {idx+1}: x Position')
#     plt.xlabel('Time')
#     plt.ylabel('x Position')
#     plt.legend()
    
#     plt.subplot(1, 2, 2)
#     theta_true = np.arctan2(true_y_val[:, 0, 3], true_y_val[:, 0, 2])
#     theta_pred = np.arctan2(pred_y_val[:, 0, 3], pred_y_val[:, 0, 2])
#     plt.plot(t.cpu().detach().numpy(), theta_true, label='True theta')
#     plt.plot(t.cpu().detach().numpy(), theta_pred, label='Predicted theta', linestyle='--')
#     plt.title(f'Trajectory {idx+1}: Pole Angle')
#     plt.xlabel('Time')
#     plt.ylabel('Pole Angle (rad)')
#     plt.legend()
    
#     plt.tight_layout()
#     plt.show()

for idx in range(num_val_traj):
    # Assuming true_y_val and pred_y_val are of shape [time, state_dim]
    true_y_val = true_y_val_list[idx].cpu().detach().numpy()
    pred_y_val = pred_y_val_list[idx].cpu().detach().numpy()
    
    plt.figure(figsize=(12, 4))
    
    # For x position
    plt.subplot(1, 2, 1)
    plt.plot(t.cpu().detach().numpy(), true_y_val[:, 0], label='True x')  # Adjusted indexing
    plt.plot(t.cpu().detach().numpy(), pred_y_val[:, 0], label='Predicted x', linestyle='--')  # Adjusted indexing
    plt.title(f'Trajectory {idx+1}: x Position')
    plt.xlabel('Time')
    plt.ylabel('x Position')
    plt.legend()
    
    # For theta
    plt.subplot(1, 2, 2)
    # Assuming the states are [x, x_dot, cos(theta), sin(theta), theta_dot]
    theta_true = np.arctan2(true_y_val[:, 3], true_y_val[:, 2])  # Adjusted indexing
    theta_pred = np.arctan2(pred_y_val[:, 3], pred_y_val[:, 2])  # Adjusted indexing
    plt.plot(t.cpu().detach().numpy(), theta_true, label='True theta')
    plt.plot(t.cpu().detach().numpy(), theta_pred, label='Predicted theta', linestyle='--')
    plt.title(f'Trajectory {idx+1}: Pole Angle')
    plt.xlabel('Time')
    plt.ylabel('Pole Angle (rad)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'val/trajectory_{idx+1}.png')
    plt.close()  
    # plt.show()
