
import torch
import numpy as np
import mpc
from mpc.mpc import QuadCost, LinDx

from mpc.env_dx.cartpole import CartpoleDx

import matplotlib
matplotlib.use('Qt5Agg')  # Set the backend to TkAgg

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

#cartpole dynamics model
cartpole_model = CartpoleDx()

# System dynamics
def cartpole_dynamics(state, action):
    # Now calling the forward method on the instance of CartpoleDx
    return cartpole_model(state, action)



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
theta_0 = 11 * np.pi/12  # Initial pole angle
state = torch.tensor([0, 0, np.cos(theta_0), np.sin(theta_0), 0], dtype=torch.float32)

# Time step
dt = 0.02
num_time_steps = 1000



# Simulation loop
for t in range(num_time_steps):
    action = torch.tensor([0], dtype=torch.float32)  # No control action yet
    state = cartpole_dynamics(state, action)  # Update state

    # Convert state to NumPy array for matplotlib animation
    state_np = state.detach().numpy()

    # print(state_np)
    # print(state)


    states.append(state_np[0])  # Storing state for animation


    theta = np.arctan2(state_np[0, 3], state_np[0, 2])  # Recompute theta from cos and sin
    state_np[0,2] = np.cos(theta)  # Update cos(theta)
    state_np[0,3] = np.sin(theta) 
    state = torch.squeeze(torch.from_numpy(state_np))


# Create animation
ani = FuncAnimation(fig, update, frames=num_time_steps, interval=dt*1000, blit=True)
plt.show()

    
