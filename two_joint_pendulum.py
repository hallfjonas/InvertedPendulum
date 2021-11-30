from math import pi
from casadi.casadi import Function, vertcat, sin, cos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from casadi import solve, mtimes, SX
from numpy.core.fromnumeric import repeat
from numpy.core.function_base import linspace
from numpy.lib.type_check import _nan_to_num_dispatcher

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
        
def draw_simulation(frame, line):
    M, I, L, R = get_constants()

    px1 = L[0]*cos(frame[0])
    px2 = L[0]*cos(frame[0]) + L[1]*cos(frame[0]+frame[1])
    py1 = L[0]*sin(frame[0])
    py2 = L[0]*sin(frame[0]) + L[1]*sin(frame[0]+frame[1])

    # Plot the animation
    xdata = [0.0, px1, px2]
    ydata = [0.0, py1, py2]
    line.set_data(xdata, ydata)

def draw_angles(frame, line_theta1, line_theta2):
    M, I, L, R = get_constants()

    num_thetas = 50
    thetas_1 = linspace(0, frame[0], num=num_thetas)
    thetas_2 = linspace(frame[0], frame[1], num=num_thetas)

    x_theta_1 = []
    y_theta_1 = []
    x_theta_2 = []
    y_theta_2 = []

    for i in range(num_thetas):
        theta1 = thetas_1[i]
        x_theta_1.append(L[0]*0.2*cos(theta1))
        y_theta_1.append(L[0]*0.2*sin(theta1))
            
        theta2 = thetas_2[i]
        x_theta_2.append(L[0]*cos(frame[0]) + L[1]*0.2*cos(theta1 + theta2))
        y_theta_2.append(L[0]*sin(frame[0]) + L[1]*0.2*sin(theta1 + theta2))
        
    line_theta1.set_data(x_theta_1, y_theta_1)
    line_theta2.set_data(x_theta_2, y_theta_2)

def initialize_mpc_plot(T0, TEnd, bound_u, block=False):
    M, I, L, R = get_constants()
    fig = plt.figure(figsize=(10, 5))
    
    plt.subplot(121)
    ln_sim, = plt.plot([], [], 'k-')
    ltheta1, = plt.plot([], [], 'b:')
    ltheta2, = plt.plot([], [], 'r:')
    plt.xlim([-L[0]-L[1], L[0]+L[1]])
    plt.ylim([-L[0]-L[1], L[0]+L[1]])
    
    plt.subplot(222)
    ln_u, = plt.plot([], [], 'k-')
    ln_u_fut, = plt.plot([], [], 'k:')
    plt.xlim([T0, TEnd])
    plt.ylim([-bound_u, bound_u])
    plt.xlabel("$t$")
    plt.ylabel("control")

    plt.subplot(224)
    ln_theta_1, = plt.plot([], [], 'b-')
    ln_theta_2, = plt.plot([], [], 'r-')
    ln_theta_1_fut, = plt.plot([], [], 'b:')
    ln_theta_2_fut, = plt.plot([], [], 'r:')
    plt.xlim([T0, TEnd])
    plt.ylim([-10, 10])
    plt.xlabel("$t$")
    plt.ylabel("angles")

    ln = [ln_sim, ltheta1, ltheta2, ln_u, ln_u_fut, ln_theta_1, ln_theta_2, ln_theta_1_fut, ln_theta_2_fut]

    plt.show(block=False)

    return fig, ln

def update_mpc_plot(t_vec, u_vec, x1_vec, x2_vec, t, T, N, u_opt, x_opt, ln):    
    M, I, L, R = get_constants()

    x0 = x_opt[0]
    draw_simulation(x0, ln[0])
    draw_angles(x0, ln[1], ln[2])

    # Plot the control
    u_vec.append(u_opt[0])
    t_vec.append(t)
    ln[3].set_data(t_vec, u_vec)
    
    u_fut = []
    t_fut_vec = linspace(t, t+T, num=len(u_opt))
    for u in u_opt:
        u_fut.append(u)
    ln[4].set_data(t_fut_vec, u_fut)

    # Plot the angles
    x1_vec.append(x0[0])
    x2_vec.append(x0[1])  
    ln[5].set_data(t_vec, x1_vec)
    ln[6].set_data(t_vec, x2_vec)

    x1_fut = []
    x2_fut = []
    for x in x_opt:
        x1_fut.append(x[0])
        x2_fut.append(x[1])

    ln[7].set_data(t_fut_vec, x1_fut[:len(x1_fut)-1])
    ln[8].set_data(t_fut_vec, x2_fut[:len(x2_fut)-1])

    plt.pause(0.001)

    return x1_vec, x2_vec

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
    ltheta1, = plt.plot([], [], 'b:')
    ltheta2, = plt.plot([], [], 'r:')
    
    plt.subplot(222)
    ln2, = plt.plot([], [], 'k-')

    plt.subplot(224)
    ln3, = plt.plot([], [], 'b-')
    ln4, = plt.plot([], [], 'r-')
    ln = [ln1, ln2, ln3, ln4, ltheta1, ltheta2]

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
        plt.ylim([min(min(x[0]), min(x[1])) - 0.1, max(max(x[0]), max(x[1])) + 0.1])
        plt.xlabel("$t$")
        plt.ylabel("angle")
        
        # Plot all the (positive) solutions for the first angle
        cosp = pi/2.0
        while (cosp < max(x[0])):
            plt.axhline(y=cosp, color='b', linestyle=':')
            cosp += 2*pi
            
        # Plot all the (negative) solutions for the first angle
        cosm = pi/2.0
        while (cosm > min(x[0])):
            plt.axhline(y=cosm, color='b', linestyle=':')
            cosm -= 2*pi

        # Plot the desired solution for the second angle
        plt.axhline(y=0.0, color='r', linestyle=':')
        
        return ln

    def update(frame):
        draw_simulation(frame, ln[0])
        draw_angles(frame, ln[4], ln[5])

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