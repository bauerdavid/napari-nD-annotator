import numpy as np

def rk4(dydt, t0, y0, Q, Q_dot, theta, theta_dot, h=1, num_points=5):
    y_prev = y0
    t_prev = t0

    h # step size: this will determine how frequent the sampling should be in the function: for big h, the sampling is small
    num_points # this determines how many points should be "interpolated". does not influence sampling frequency, just the number of sampling points.
    y = [y0] # function to be determined: we only know its value in 1 point
    t = [t0]
    debug_vars = []

    for n in range(num_points):
        debug_vars.append(np.tan(y_prev))

        k1 = dydt(t_prev,y_prev, Q, Q_dot, theta, theta_dot)
        k2 = dydt(int(t_prev+h/2), y_prev+h*(k1/2), Q, Q_dot, theta, theta_dot)
        k3 = dydt(int(t_prev+h/2), y_prev+h*(k2/2), Q, Q_dot, theta, theta_dot)
        k4 = dydt(t_prev+h, y_prev+h*k3, Q, Q_dot, theta, theta_dot)

        qp = Q[t_prev]
        qdp = Q_dot[t_prev]
        tp = theta_dot[t_prev]
        tdp = theta_dot[t_prev]

        qn = Q[int(t_prev+h/2)]
        qdn = Q_dot[int(t_prev+h/2)]
        tn = theta_dot[int(t_prev+h/2)]
        tdn = theta_dot[int(t_prev+h/2)]

        y_next = y_prev+(1/6)*h*(k1+2*k2+2*k3+k4)

        tanyprev = np.tan(y_prev)
        tanynext = np.tan(y_next)

        #if np.sin(y_next)<0:
        #    y_next = 2*np.pi-y_next
        t_next = t_prev+h

        r_next = np.cbrt( np.sin(y_next)* Q[int(t_prev+h/2)]/theta_dot[int(t_prev+h/2)] )

        y.append(y_next)
        t.append(t_next)

        y_prev = y_next
        t_prev = t_next
    
    debug_vars.append(0)
    return t, y, debug_vars
