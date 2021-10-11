import numpy as np

# Controlled Euler Integrator
def euler_controlled(fun, x0, h, N):
    xk = x0
    x = np.array(xk)
    for i in range(N):
        xk = xk + h*fun(xk)
        x = np.vstack([x, xk])

    return(x)

# Controlled Euler Integrator
def euler_controlled(fun, x0, u, h, N):
    xk = x0
    x = np.array(xk)
    for i in range(N):
        xk = xk + h*fun(xk, u[i])
        x = np.vstack([x, xk])

    return(x)