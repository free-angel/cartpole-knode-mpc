from torch import nn
import torch

import mpc1
from mpc1 import mpc
from mpc1.mpc import QuadCost, LinDx, GradMethods

from mpc1.env_dx import cartpole

import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class NominalModel(nn.Module):

    def __init__(self, p=torch.tensor((9.8, 1.0, 0.1, 0.5)), f=20., theta_0=np.pi, theta_dot_0=-np.pi/2, n_batch=59):
        super(NominalModel, self).__init__()  
        self.cartpole_model = cartpole.CartpoleDx(params=p)
        self.cartpole_model.force_mag = f

        self.u_init = None


        self.nominal_y0 = torch.tensor([0, 0, np.cos(theta_0), np.sin(theta_0), theta_dot_0], dtype=torch.float32)

        # self.mpc_T = 25 #pred horizon
        self.mpc_T = 25 #pred horizon

        self.n_batch = n_batch
        self.q, self.p = self.cartpole_model.get_true_obj()
        self.Q = torch.diag(self.q).unsqueeze(0).unsqueeze(0).repeat(
            self.mpc_T, self.n_batch, 1, 1
        )
        self.p = self.p.unsqueeze(0).repeat(self.mpc_T, self.n_batch, 1)

        # self.actions = []    
        # dimensions of state and action
        # n_state = 5  # [x position, x dot, cos(theta), sin(theta), theta dot]
        # n_ctrl = 1   # [force applied to the cart]    

    def forward(self, t, y):
        # state = y.repeat(2) if y.ndimension() == 1 else y
        # # Ensure state is 2D: [n_batch, state_dim]
        # if state.ndimension() == 1:
        #     state = state.repeat(2)  # Add batch dimension if it's missing
        # elif state.ndimension() > 2:
        #     raise ValueError("State input to MPC must be 2D [n_batch, state_dim].")
        if y.ndimension() == 1:
            state = y.unsqueeze(0)  # Add batch dimension if it's missing
        else:
            state = y

        print(state.size(), self.n_batch)

        nominal_states, nominal_actions, nominal_objs = mpc.MPC(
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
        )(state, QuadCost(self.Q, self.p), self.cartpole_model)
        next_action = nominal_actions[0]
        # action_history.append(next_action)
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, self.n_batch, self.cartpole_model.n_ctrl)), dim=0)
        u_init[-2] = u_init[-3]
        
        # state = self.cartpole_model(state, next_action).squeeze()
        state = self.cartpole_model(state, next_action)


        # print(f"Nominal Force: {next_action.item():.2f}N,  cos theta: {state[2]:.2f}, sine theta: {state[3]:.2f}")



        # return state
        return (state - y) / self.cartpole_model.dt

    

class UncertaintyModel(nn.Module):
    def __init__(self, state_dim, *, hidden_dim=50):
        super(UncertaintyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, state_dim)
        )

    def forward(self, t, y):
        return self.net(y)


class HybridModel(nn.Module):
    def __init__(self, nominal_model, uncertainty_model):
        super(HybridModel, self).__init__()
        self.nominal_model = nominal_model
        self.uncertainty_model = uncertainty_model

    def forward(self, t, y):
        nominal_dydt = self.nominal_model.forward(t, y)
        uncertainty_dydt = self.uncertainty_model.forward(t, y)
        return nominal_dydt + uncertainty_dydt
