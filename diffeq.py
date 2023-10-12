import numpy as np
import matplotlib.pyplot as plt
from math import *
from scipy.optimize import curve_fit

G = 9.81
E = np.e
PI = pi
STEP = 0.001

def integrate(f, a, b, step=STEP):
    return sum([f(x) * step for x in np.arange(a, b, step)])

def upper_bound(f, a, bound, step=STEP):
    intg = 0
    x = a

    while intg < bound:
        intg += f(x) * step
        x += step

    return x

def solve(theta, speed, M, K):
    # the initial horizontal velocity
    vx_i = speed * cos(theta)
    # the initial vertical velocity
    vy_i = speed * sin(theta)
    
    # velocity equation for v_y(t) > 0
    vy_p = lambda t: (M/K) * (abs(G + K*vy_i/M) / E**(K*t/M) - G)
    # velocity equation for v_y(t) < 0 
    vy_n = lambda t: (M/K) * (abs(K*vy_i/M - G) * E**(K*t/M) + G)
    # velocity equation for v_x(t)
    vx = lambda t: (M/K) * abs(K*vx_i/M) / E**(K*t/M)

    # calculate the time 't1' it takes to reach the vertex of the curve
    t1 = (M / K) * log(abs(G + K*vy_i/M) / G)
    # calculate the height reached by the particle in time 't1'
    H = integrate(vy_p, 0, t1)
    # calculate the time 't2' it takes for the particle to fall back down
    t2 =  upper_bound(vy_n, t1, -H) 
    # integrate the horizontal displacement over the air time 't1 + t2'
    distance = integrate(vx, 0, t1 + t2)

    return distance

def run(theta, speed, M, K):
    p = [0, 0]
    v = [speed * cos(theta), speed * sin(theta)]

    while True:
        if p[1] < 0 and v[1] < 0:
            break

        Rx = v[0] * K
        Ry = v[1] * K

        F = [-Rx, -G*M - Ry]

        v[0] += F[0] / M * STEP
        v[1] += F[1] / M * STEP

        p[0] += v[0] * STEP
        p[1] += v[1] * STEP
    
    return p[0]

def graph():
    M = 1
    V = 1
    SAMPLES = 40

    fig = plt.figure(figsize=(9, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.set_ylabel('launch angle (rad)')
    ax1.set_zlabel('launch distance (m)')
    ax1.set_xlabel('coefficient of friction')

    ax1.set_title('Drag vs. Angle vs. Distance Graph')


    MAX_Ts = []
    Ks = []

    for k in np.linspace(0.1, 10.0, num=SAMPLES):
        Ts = []
        Ds = []

        max_t = 0
        max_d = 0

        for t in np.arange(0, PI/2, STEP):
            d = solve(t, V, M, k)
            
            if d > max_d:
                max_d = d
                max_t = t

            Ts += [t]
            Ds += [d]

        MAX_Ts += [max_t]
        Ks += [k]

        ax1.plot(Ts, Ds, zs=k, zdir='x')


    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('coefficient of friction')
    ax2.set_ylabel('optimal launch angle (rad)')
    ax2.plot(Ks, MAX_Ts)

    fun = lambda x, a, b, c: a * np.log(b*x) + c

    [a, b, c], _ = curve_fit(fun, Ks, MAX_Ts)

    ax2.annotate(f'Curve Fit: f(k) = {a}ln({b}k) + {c}', xy=(3.3, 1))

    Os = [fun(k, a, b, c) for k in Ks]    

    ax2.plot(Ks, Os, '--', label=f'{a:0.2f} ln({b:0.2f}k) + {c:0.2f}')

    ax2.set_title('Optimal angle vs Drag Graph')
    ax2.legend(loc='best')
 
    plt.show()

graph()
