from math import pi
from casadi.casadi import Function, vertcat, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from casadi import solve, mtimes, SX
from numpy.core.fromnumeric import repeat

def get_constants():
    M = [1.0, 1.0]
    I = [1.0, 1.0]
    L = [2.0, 2.0]
    R = [1.0, 1.0]
    return M, I, L, R

def fun(x, u):
    M, I, L, R = get_constants()

    theta = x[0:2]
    theta_dot = x[2:4]

    c1 = 0.5*(M[0]*(R[0]**2.0) + I[0])
    c2 = 0.5*(M[1]*(R[1]**2.0) + I[1])
    f1 = 0.5*M[1]*(L[0]**2 + R[1]**2 + 2*L[0]*R[1]*cos(theta[1]))
    f2 = M[1]*(R[1]**2 + L[0]*R[1]*cos(theta[1]))
    fprime = -M[1]*L[0]*R[1]*sin(theta[1])
    
    # Compute the matrix M (TODO: SX Style...)
    M_mat = SX.sym('M', 2, 2)
    M_mat[0] = 2*(c1 + f1)
    M_mat[1] = f2
    M_mat[2] = f2
    M_mat[3] = 2*c2

    # Compute the inverse of M
    det_M = M_mat[0]*M_mat[3] - M_mat[1]*M_mat[2]
    M_inv = vertcat(
        M_mat[3]/det_M,
        -M_mat[2]/det_M,
        -M_mat[1]/det_M,
        M_mat[0]/det_M
    )

    # Compute the matrix C
    C11 = 2*fprime*theta_dot[1]
    C12 = fprime*theta_dot[1]
    C21 = -fprime*theta_dot[0]
    C22 = 0.0*theta_dot[0]

    Cx = [C11*x[2] + C12*x[3], C21*x[2] + C22*x[3]]

    # Get q
    g = 9.81
    q1 = g*M[0]*R[0]*cos(theta[0]) + g*M[1]*(L[0]*cos(theta[0]) + R[1]*cos(theta[0]+theta[1]))
    q2 = g*M[1]*R[1]*cos(theta[0]+theta[1])
    q = [q1, q2]

    # RHS of state space equation
    tmp = SX.sym('tmp', 2)
    tmp[0] = 0.0 - Cx[0] - q[0]
    tmp[1] = u - Cx[1] - q[1]

    sol = vertcat(x[2], x[3], M_inv[0]*tmp[0] + M_inv[1]*tmp[1], M_inv[2]*tmp[0] + M_inv[3]*tmp[1])

    return sol

def create_animation(t_vec, states, expname, save=False):
    
    M, I, L, R = get_constants()

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ln, = plt.plot([], [], 'b-')

    length = L[0]+L[1]+0.5

    def init():
        ax.set_xlim(-length,length)
        ax.set_ylim(-length,length)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        return ln,

    def update(frame):
        px1 = L[0]*cos(frame[0])
        px2 = L[0]*cos(frame[0]) + L[1]*cos(frame[0]+frame[1])
        py1 = L[0]*sin(frame[0])
        py2 = L[0]*sin(frame[0]) + L[1]*sin(frame[0]+frame[1])

        xdata = [0.0, px1, px2]
        ydata = [0.0, py1, py2]
        ln.set_data(xdata, ydata)
        return ln,

    interval = 1000/len(states)*(t_vec[-1]-t_vec[0])

    ani = FuncAnimation(fig, update, frames=states, interval=interval,
                        init_func=init, blit=True)

    if save:
        dur = t_vec[-1] - t_vec[0]
        count = len(t_vec)
        fps = round(count/dur)

        writergif = PillowWriter(fps=fps)
        ani.save("figures/manipulator" + expname + ".gif",writer=writergif)
    else:
        plt.show()
        

def create_subplot_animation(t_vec, states, expname, save=False):

    u = states['u']
    x = states['x']

    states = []
    for i in range(len(u)):
        states.append([x[0][i], x[1][i], x[2][i], x[3][i], u[i], t_vec[i]])
    
    M, I, L, R = get_constants()

    fig = plt.figure(figsize=(10, 5))
    

    plt.subplot(121)
    ln1, = plt.plot([], [], 'k-')
    
    plt.subplot(222)
    ln2, = plt.plot([], [], 'k-')

    plt.subplot(224)
    ln3, = plt.plot([], [], 'b-')
    ln4, = plt.plot([], [], 'r-')
    ln = [ln1, ln2, ln3, ln4]

    t_data = []
    u_data = []
    x1_data = []
    x2_data = []

    length = L[0]+L[1]+0.5

    def init():
        plt.subplot(121)
        plt.xlim(-length,length)
        plt.ylim(-length,length)
        plt.xlabel("$x_1$")
        plt.ylabel("$x_2$")
        
        plt.subplot(222)
        plt.xlim([min(t_vec), max(t_vec)])
        plt.ylim([min(u), max(u)])
        plt.xlabel("$t$")
        plt.ylabel("control")
        
        plt.subplot(224)
        plt.xlim([min(t_vec), max(t_vec)])
        plt.ylim([min(min(x[0]), min(x[1])), max(max(x[0]), max(x[1]))])
        plt.xlabel("$t$")
        plt.ylabel("angle")
        plt.axhline(y=pi/2, color='b', linestyle=':')
        plt.axhline(y=0.0, color='r', linestyle=':')
        
        return ln

    def update(frame):

        px1 = L[0]*cos(frame[0])
        px2 = L[0]*cos(frame[0]) + L[1]*cos(frame[0]+frame[1])
        py1 = L[0]*sin(frame[0])
        py2 = L[0]*sin(frame[0]) + L[1]*sin(frame[0]+frame[1])

        # Plot the animation
        xdata = [0.0, px1, px2]
        ydata = [0.0, py1, py2]
        ln[0].set_data(xdata, ydata)

        # Clear data when t=t0
        t_val = frame[-1]
        if (t_val == t_vec[0]):
            t_data.clear()
            u_data.clear()
            x1_data.clear()
            x2_data.clear()

        # Plot the control
        u_val = frame[-2]
        u_data.append(u_val)
        t_data.append(t_val)        
        ln[1].set_data(t_data, u_data)

        # Plot the angles
        x1_data.append(frame[0])
        x2_data.append(frame[1])        
        ln[2].set_data(t_data, x1_data)
        ln[3].set_data(t_data, x2_data)

        return ln

    interval = 1000/len(states)*(t_vec[-1]-t_vec[0])

    ani = FuncAnimation(fig, update, frames=states, interval=interval,
                        init_func=init, blit=True, repeat=True)

    if save:
        dur = t_vec[-1] - t_vec[0]
        count = len(t_vec)
        fps = round(count/dur)

        writergif = PillowWriter(fps=fps)
        ani.save("figures/manipulator" + expname + ".gif",writer=writergif)
    else:
        plt.show()