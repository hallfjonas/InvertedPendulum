from threading import Semaphore
import numpy as np

# Controlled Euler Integrator
def euler_uncontrolled(fun, x0, h, N):
    xk = x0
    x = np.array(xk)
    for i in range(N):
        xk = xk + h*fun(xk)
        x = np.vstack([x, xk])

    return(x)

# Controlled Euler Integrator
def euler_controlled_step(fun, xk, u, h):
    return (xk + h*fun(xk, u))

# Controlled Euler Integrator
def euler_controlled(fun, x0, u, h, N):
    xk = x0
    x = np.array(xk)
    for i in range(N):
        xk = xk + h*fun(xk, u[i])
        x = np.vstack([x, xk])

    return(x)

# Get RK Tableau
def get_rk_tableau(type):
    if (type == 'rk4'):
        a = np.array([[0,0,0,0],[0.5,0,0,0],[0,0.5,0,0],[0,0,1,0]])
        b = np.array([1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0])
        c = np.array([0,0.5,0.5,1])

    return (a,b,c)

# Runge Kutta Methods (time-implicit)
def rk_step(fun, xk, u, h, type):

    (a,b,c) = get_rk_tableau('rk4')
    s = len(b)

    sm = xk
    k = []
    for i in range(s):
        eval_point = xk
        for j in range(i):
            eval_point = eval_point + h*a[i,j]*k[j]

        new_k = fun(eval_point, u)
        k.append(new_k)
        sm = sm + h*b[i]*new_k

    return sm