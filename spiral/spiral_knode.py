import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

from models import ODEFunc

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='rk4')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=998)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lossMSE = nn.MSELoss()
num_traj = 1;


# true_y0 = torch.tensor([[2., 0.]]).to(device)

#version with more than one trainint trajectory
x_array = 6. * torch.rand(num_traj) - 3.
y_array = 6. * torch.rand(num_traj) - 3.

t = torch.linspace(0., 5., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.5], [-2.5, -0.1]]).to(device)

#true dynamics
class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y**3, true_A)

true_y_list = []
for idx in range(num_traj):
    true_y0     = torch.tensor([[x_array[idx], y_array[idx]]]).to(device)
    with torch.no_grad():
        true_y = odeint(Lambda(), true_y0, t, method='rk4', options=dict(step_size=0.005))
    
    true_y_list.append(true_y)


def get_batch(t, true_y, data_size, num_traj, batch_time, device):
    # s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    # batch_y0 = true_y[s]  # (M, D)
    # batch_t = t[:args.batch_time]  # (T)
    # batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    # return batch_y0.to(device), batch_t.to(device), batch_y.to(device)
    s           = torch.arange(data_size - 1)
    batch_y0    = true_y[s]
    batch_t     = t[:batch_time]
    batch_y     = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


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

    ii = 0

    func = ODEFunc().to(device)
    
    optimizer = optim.Rprop(func.parameters(), lr=1e-4)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)

    lowest_loss_so_far  = 1e5
    batch_y0_list       = []
    batch_t_list        = []
    batch_y_list        = []

    for idx in range(num_traj):
        batch_y0, batch_t, batch_y  = get_batch(t, true_y_list[idx], args.data_size, num_traj, args.batch_time, device)
        batch_y0_list.append(batch_y0)
        batch_t_list.append(batch_t)
        batch_y_list.append(batch_y)

    for itr in tqdm.tqdm(range(1, args.niters + 1)):
        optimizer.zero_grad()
        # batch_y0, batch_t, batch_y = get_batch()
        # pred_y = odeint(func, batch_y0, batch_t, method=args.method, options=dict(step_size=0.005)).to(device)
        # loss = lossMSE(batch_y, pred_y)

        loss = 0.0
        for idx in range(num_traj):
            pred_y     = odeint(func, batch_y0_list[idx], batch_t_list[idx], method=args.method, options=dict(step_size=0.02)).to(device)
            loss      += lossMSE(pred_y[1, :, :], batch_y_list[idx][1, :, :])

        loss.backward()
        optimizer.step()

        # time_meter.update(time.time() - end)
        # loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            print('Iter {:4d} | Training Loss {:e}'.format(itr, loss.item()))

        # if itr % args.test_freq == 0:
        #     # loss = loss.item()
        #     print('Iter {:04d} | Total Loss {:}'.format(itr, loss.item()))
        #     visualize(true_y, pred_y, func, ii)
        #     ii += 1

        end = time.time()

    # torch.save({'state_dict': func.state_dict()}, 'model.pth')
    for i in range(10):
        test_y0 = (3. * torch.rand(1, 2)).to(device)

        print(test_y0)
        with torch.no_grad():
            test_y = odeint(Lambda(), test_y0, t, method='rk4', options=dict(step_size=0.005))
            pred_y = odeint(func, test_y0, t, method=args.method, options=dict(step_size=0.005)).to(device)
        
        
        loss = lossMSE(test_y, pred_y)
        print(loss.item())




