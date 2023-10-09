import numpy as np
from math import *

G = 9.81

def integrate(f, a, b, step=0.01):
    return sum([f(x) for x in np.arange(a, b, step)])

def integrate_to(f, a, end, step=0.01):
    total = 0
    x = a

    while total <= end:
        total += f(x)

    return x

def solve(theta, speed, mass, K):
    vx_i = speed * cos(theta)
    vy_i = speed * sin(theta)

    t1 = (mass / K) * log(abs(G + K * speed / mass) / G)

    height = integrate(lambda t: (mass / K) * (abs(G + K * vy_i / mass) / np.e ** (K * t / mass)), 0, t1)

    t2 =  integrate_to(lambda t: G * (np.e ** (K * t / mass) + 1, 0, t1)) 

    distance = integrate(lambda t: m * abs(k * vx_i / mass) / (k * np.e ** (K * t / mass)), 0, t1 + t2)

    return distance
    
