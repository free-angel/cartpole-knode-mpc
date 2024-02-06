
import torch
import numpy as np
import mpc1
from mpc1 import mpc
from mpc1.mpc import QuadCost, LinDx, GradMethods

from mpc1.env_dx import cartpole

import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to TkAgg

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#cartpole dynamics model
#default params : # gravity = 9.8, masscart = 1.0, masspole = 0.1, length = 0.5
cartpole_model = cartpole.CartpoleDx(params=torch.tensor((9.8, 1.0, 0.1, 0.5))) 
cartpole_model.force_mag = 20. #default: 30N

# System dynamics
def cartpole_dynamics(state, action):
    return cartpole_model(state, action) #built in forward method



# MPC
mpc_T = 25 #pred horizon
n_batch = 1
q, p = cartpole_model.get_true_obj()
Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
    mpc_T, n_batch, 1, 1
)
p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)


# Matplotlib figure
fig, ax = plt.subplots()
ax.set_xlim(-2, 2)
ax.set_ylim(-1, 1.5)
line, = ax.plot([], [], lw=2, color='red')  # Pole line
cart = plt.Rectangle((-0.2, -0.1), 0.4, 0.2, color='blue')  # Cart
ax.add_patch(cart)

# states storage for animation
states = []

# Animation update function
def update(frame):
    x, _, cos_theta, sin_theta, _ = states[frame]
    theta = np.arctan2(sin_theta, cos_theta)

    cart.set_xy((x - 0.2, -0.1))  # Update cart position
    pole_end_x = x + 0.5 * np.sin(theta)  # Pole end x-coordinate (0.5 is half-length of pole)
    pole_end_y = 0.5 * np.cos(theta)  # Pole end y-coordinate
    line.set_data([x, pole_end_x], [0, pole_end_y])

    return line, cart




# dimensions of state and action
n_state = 5  # [x position, x dot, cos(theta), sin(theta), theta dot]
n_ctrl = 1   # [force applied to the cart]

# Initial state
theta_0 = 12 * np.pi/12  # Initial pole angle
theta_dot_0 = -np.pi/4

state = torch.tensor([0, 0, np.cos(theta_0), np.sin(theta_0), theta_dot_0], dtype=torch.float32)

# Time step
dt = 0.02
num_time_steps = 350

state = state.unsqueeze(0)
u_init = None

# Simulation loop
for t in range(num_time_steps):
    # print(state.size())
    state = state.unsqueeze(0) if state.ndimension() == 1 else state
    nominal_states, nominal_actions, nominal_objs = mpc.MPC(
        cartpole_model.n_state, cartpole_model.n_ctrl, mpc_T,
        u_init=u_init,
        u_lower=cartpole_model.lower, u_upper=cartpole_model.upper,
        lqr_iter=50,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=cartpole_model.linesearch_decay,
        max_linesearch_iter=cartpole_model.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,
    )(state, QuadCost(Q, p), cartpole_model)
    next_action = nominal_actions[0]
    # action_history.append(next_action)
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, cartpole_model.n_ctrl)), dim=0)
    u_init[-2] = u_init[-3]
    state = cartpole_dynamics(state, next_action)


    # action = torch.tensor([0], dtype=torch.float32)  # No control action yet
    # state = cartpole_dynamics(state, action)  # Update state

    # Convert state to NumPy array for matplotlib animation
    state_np = state.detach().numpy()

    # print(state_np)
    # print(state)


    states.append(state_np[0])  # Storing state for animation


    theta = np.arctan2(state_np[0, 3], state_np[0, 2])  # Recompute theta from cos and sin
    state_np[0,2] = np.cos(theta)  # Update cos(theta)
    state_np[0,3] = np.sin(theta) 
    state = torch.squeeze(torch.from_numpy(state_np))

    print(f"Step {t+1}/{num_time_steps}, Time: {t*dt:.2f}s, State: [x: {state_np[0,0]:.2f}, x_dot: {state_np[0,1]:.2f}, theta: {theta * 180 / np.pi:.2f}°, theta_dot: {state_np[0,4]:.2f}], Force: {next_action.item():.2f}N")



# Create animation
ani = FuncAnimation(fig, update, frames=num_time_steps, interval=dt*1000, blit=True)
plt.show()

    
