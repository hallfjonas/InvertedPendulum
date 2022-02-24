from casadi import SX, Function, nlpsol, vertcat, cos, sin
from math import inf, log, exp, pi 

from numpy.core.function_base import linspace, logspace
from two_joint_pendulum import fun, create_subplot_animation, initialize_mpc_plot, update_mpc_plot
import matplotlib.pyplot as plt

## Settings
T = 2.0             # Horizon length
T0 = 0.0            # Initial time
TEnd = 30.0         # Final time    
N = 40              # Number of nodes
nx = 4              # Number of states
nu = 1              # Number of controls
h = T/N

# Initial state
x0 = [pi/2.0 + 0.0, -0.051, 0.0, 0.0]

# Cost
penalize_com = True                 # Penalty term on the center of mass (m1 + m2)/2
penalize_angles = False             # Penalty term on xk[0], xk[1] => Convex cost terms
penalize_position = False           # Penalty term on (sin, cos of xk[0], xk[1]) => Non-convex cost terms
penalize_velocity = False           # Also include convex cost terms of velocity
penalize_control = False            # Also include convex cost terms of control

# Cost factors begin at fact, and multiply by beta after each node
fact = 10e-2
beta = exp(log(10.0/fact)/N)
weights_beginning = []
for k in range(N):
    weights_beginning.append(fact*pow(beta, k))

# Control bound
bound_u = 20.0

# Symbolic variables
x = SX.sym("x", nx)
u = SX.sym("u", nu)

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
Xk = SX.sym('x0', 4)
weights = SX.sym('w', N)
P = vertcat(Xk, weights)
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

    # Cost term: center of mass on y axis above 0
    if penalize_com:
        com_x = cos(Xk[0]) + cos(Xk[0]+Xk[1])
        com_y = sin(Xk[0]) + sin(Xk[0]+Xk[1])

        # Quadratic term on com x
        J += weights[k]*com_x**2.0

        # Linear term on com y (small perturbations are more expensive)
        J += -weights[k]*com_y

    # Cost term: first angle at 90°, second angle at 0°
    if penalize_angles:
        J += weights[k]*(Xk[0] - pi/2.0)**2.0
        J += weights[k]*(Xk[1])**2.0 / 2.0

    # Cost term: position of endeffector all the way up
    if penalize_position:
        J += weights[k]*((cos(Xk[0]) + cos(Xk[0]+Xk[1]))**2.0 + (sin(Xk[0]) + sin(Xk[0]+Xk[1]) - 1.0)**2.0)

    # Cost term: velocity
    if penalize_velocity:
        J += (Xk[2])**2.0
        J += (Xk[3])**2.0

    # Cost term: control
    if penalize_control:
        J += weights[k]*(Uk[0])**2.0/5.0

# Create an NLP solver
prob = {'f': J, 'x': vertcat(*w), 'p': P}
solver = nlpsol('solver', 'ipopt', prob);

def swush_x0(x0):
    x0_new = []
    for i in range(1, len(x0)):
        x0_new.append(x0[i])
    x0_new.append(x0[-1])
    return x0_new

fig, ln = initialize_mpc_plot(T0, TEnd, bound_u, block=False)
t_vec = []
u_vec = []
x1_vec = []
x2_vec = []

t = 0.0
w = weights_beginning
while (t < TEnd):
    # Solve the NLP
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, p=vertcat(x0, w))

    # Get solution
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

    # Plot simulation 
    update_mpc_plot(t_vec, u_vec, x1_vec, x2_vec, t, T, N, u_opt, x_opt, ln)

    # Get new initial guess
    w0 = swush_x0(u_opt)
    w = swush_x0(w)

    # Get new initial state
    x0 = x_opt[1]
    
    # Update time value
    t += h
