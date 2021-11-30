import imp
from casadi import SX, Function, nlpsol, vertcat, cos, sin
from math import inf, log, exp, pi 

from numpy.core.function_base import linspace
from two_joint_pendulum import fun, create_animation, create_subplot_animation
import matplotlib.pyplot as plt

T = 4.0
N = 50
nx = 4
nu = 1

# The initial position
x0 = [pi/2.0 + 0.0, -0.051, 0.0, 0.0]

# Control
u = SX.sym("u")
bound_u = 40.0

# Cost
penalize_angles = True              # Penalty term on xk[0], xk[1] => Convex cost terms
penalize_position = True           # Penalty term on (sin, cos of xk[0], xk[1]) => Non-convex cost terms
penalize_velocity = False           # Also include convex cost terms of velocity
penalize_control = False            # Also include convex cost terms of control

# Cost factors begin at fact, and multiply by beta after each node
fact = 10e-2
beta = exp(log(100.0/fact)/N)

# State
x = SX.sym("x", 4)

# ODE right hand side function
fx = fun(x, u)

# Fixed step Runge-Kutta 4 integrator
M = 4 # RK4 steps per interval
DT = T/N/M
f = Function('f', [x, u], [fx])
X0 = SX.sym('X0', 4)
U = SX.sym('U')
X = X0
Q = 0
for j in range(M):
    k1 = f(X, U)
    k2 = f(X + DT/2 * k1, U)
    k3 = f(X + DT/2 * k2, U)
    k4 = f(X + DT * k3, U)
    X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
F = Function('F', [X0, U], [X],['x0','p'],['xf'])

# Start with an empty NLP
w=[]
w0 = []
lbw = []
ubw = []
J = 0

# Formulate the NLP
Xk = SX(x0)
for k in range(N):
    # New NLP variable for the control
    Uk = SX.sym('U_' + str(k))
    w += [Uk]
    lbw += [-bound_u]
    ubw += [bound_u]
    w0 += [0]

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=Uk)
    Xk = Fk['xf']

    # Cost term: First angle at 90°, second angle at 0°
    if penalize_angles:
        # J += fact*(Xk[0] - pi/2.0)**2.0
        J += fact*(Xk[1])**2.0

    # Alternative cost term: Position of endeffector all the way up
    if penalize_position:
        J += fact*((cos(Xk[0]) + cos(Xk[0]+Xk[1]))**2.0 + (sin(Xk[0]) + sin(Xk[0]+Xk[1]) - 1.0)**2.0)

    # Add cost term to velocity
    if penalize_velocity:
        J += (Xk[2])**2.0
        J += (Xk[3])**2.0

    # Add cost term to control if desired
    if penalize_control:
        J += fact*(Uk[0])**2.0/5.0

    fact *= beta

# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w)}
solver = nlpsol('solver', 'ipopt', prob);

# Solve the NLP
sol = solver(x0=w0, lbx=lbw, ubx=ubw)
u_opt = sol['x'].full().flatten()

x1_opt = [x0[0]]
x2_opt = [x0[1]]
x3_opt = [x0[2]]
x4_opt = [x0[3]]
x_opt = [x0]
for u in u_opt:
    tmp = F(x0=x_opt[-1],p=u)['xf'].full().flatten()
    x1_opt.append(tmp[0])
    x2_opt.append(tmp[1])
    x3_opt.append(tmp[2])
    x4_opt.append(tmp[3])
    x_opt.append([tmp[0], tmp[1], tmp[2], tmp[3]])

create_subplot_animation(linspace(0,T,num=len(u_opt)), {'u': u_opt, 'x': [x1_opt, x2_opt, x3_opt, x4_opt]}, "TEST")
