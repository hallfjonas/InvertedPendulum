import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from integrators import euler_controlled

# Discretization Grid
T = 20
N = 200
h = T/N
t = np.arange(0,T+h,h)

# State space (p, v, phi, phi_dot)
nx = 4

# Random initial state (if p = v = 0, then this should just be a pendulum)
# Remark(1): Choose phi0 = math.pi to see unphysical behavior
# Remark(2): Why is the velocity slowly increasing over time?
p0 = 0
v0 = 0
phi0 = np.random.uniform(low=-math.pi/4, high=math.pi/4)
phi_dot0 = 0
x0 = np.array([p0, v0, phi0, phi_dot0])

# Controll sequence (currently constant)
nu = 1
u_fixed = 0
u_trajectory = np.repeat(u_fixed, N)

# System Dynamics 
# States: x = [p, p_dot,     phi, phi_dot]
#     x_dot = [v,     u, phi_dot, -phi - u]
# Remark: I think phi_dot_dot needs adjustments
def f(x, u):
    return( np.array([x[1], u, x[3], -x[2] - u]) )

# Integrate
x_trajectory = euler_controlled(f, x0, u_trajectory, h, N)

## Plot states and controls
fig1, axs = plt.subplots(nrows=nx+nu,ncols=1)

lines = []
# Plot states
for i in range(nx):
    axs[i].plot(t, x_trajectory[:,i], 'b-')
    axs[i].set_xlim(0, T)
    axs[i].set_ylim(min(x_trajectory[:,i]), max(x_trajectory[:,i]))

# Plot controls
axs[nx].plot(t, np.append(u_trajectory, 0),'r-')
axs[nx].set_xlim(0, T)
axs[nx].set_ylim(min(u_trajectory), max(u_trajectory))

## Create an animation of the pendulum
fig2, ax = plt.subplots()
line0, = ax.plot([],[],'k-')
ax.set_xlim(min(x_trajectory[:,0]) - 1.5, max(x_trajectory[:,0]) + 1.5)
ax.set_ylim(-1.5, 1.5)
p_data, phi_data = [], []

ln = [line0]

def update(frame):
    pk = x_trajectory[frame,0]
    phik = x_trajectory[frame,2] - math.pi/2.0
    ln[0].set_data([pk, np.cos(phik) + pk], [0, np.sin(phik)])    
    return ln

ani = FuncAnimation(fig2, update, frames=N, interval=T, blit=True, repeat=False)

plt.show()
