import math
import numpy as np
import matplotlib.pyplot as plt
from integrators import euler_controlled_step, rk_step
import keyboard
import time

# Pendulum dimensions
m = 1       # mass   [kg]
l = 2       # length [m]
g = 9.81    # Gravitational force

# State space (p, v, phi, phi_dot)
nx = 4

# Random initial state (if p = v = 0, then this should just be a pendulum)
# Remark(1): Choose phi0 = math.pi to see unphysical behavior
# Remark(2): Why is the velocity slowly increasing over time?
p0 = 0
v0 = 0
phi0 = np.random.uniform(low=-math.pi, high=math.pi)
phi_dot0 = 0
x0 = np.array([p0, v0, phi0, phi_dot0])

# System Dynamics 
# States: x = [p, p_dot, phi, phi_dot]
#     x_dot = [v,     u, phi_dot, -phi - u]
# Remark: I think phi_dot_dot needs adjustments
def f(x, u):
    return( np.array([x[1], u, x[3], -g/l*math.sin(x[2]) + 1/(m*(l**2.0))]) )

print(x0)
t0 = time.time() 
t1 = time.time()

xk = x0

min_cycle = 0.1

## Create an animation of the pendulum
fig2, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect("equal")
p_data, phi_data = [], []
ln, = ax.plot([],[], 'r-')
plt.ion()
plt.show()


def update(xk):
    print(xk)
    pk = xk[0]
    phik = xk[2] - math.pi/2.0
    ln.set_data([pk, np.cos(phik) + pk], [0, np.sin(phik)])
    
    plt.draw()
    plt.pause(0.001)

update(x0)
## Open control loop
while (True):

    # terminate on esc
    if (keyboard.is_pressed("esc")):
        break

    t0 = t1
    t1 = time.time()

    h = t1 - t0
    
    # Assume no control
    u = 0

    if (keyboard.is_pressed("esc")):
        break
    # Check if applied force to right
    if (keyboard.is_pressed("left")):
        u = -1
    elif (keyboard.is_pressed("right")):
        u = 1 
     
    xk = rk_step(f, xk, u, h, 'rk4')
    update(xk)

    print(xk)
