import imp
from casadi import SX, Function, mtimes, nlpsol, vertcat, SX_inf
from math import inf, pi

from numpy.core.function_base import linspace
from two_joint_pendulum import fun, create_animation, create_subplot_animation
import matplotlib.pyplot as plt

T = 8.0
N = 100
nx = 4
nu = 1

x0 = [-pi/2.0, 0.0, 0.0, 0.0]

# Control
u = SX.sym("u")
bound_u = 10.0

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

fact = 10e-4
beta = 1.2
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

    J += fact*(Xk[0] - pi/2.0)**2.0
    J += fact*(Xk[1])**2.0

    # Add cost term to control if desired
    # J += fact*(Uk[0])**2.0/10.0

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
