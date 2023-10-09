import matplotlib.pyplot as plt
import numpy as np
from math import *
from scipy.optimize import curve_fit

TIME_STEP = 0.001 # s
G = 9.81          # m / s^2
MASS = 10.0       # kg
PI = np.pi


def run(theta, v_0, mass, friction, density):
    p = [0, 0]
    v = [v_0 * cos(theta), v_0 * sin(theta)]

    Xs = []
    Ys = []

    while True:
        if p[1] < 0 and v[1] < 0:
            break

        mag = sqrt(v[0]**2 + v[1]**2)

        Rx = 0.5 * v[0]*mag * density * friction
        Ry = 0.5 * v[1]*mag * density * friction

        F = [-Rx, -G*mass-Ry]

        v[0] += F[0] / mass * TIME_STEP
        v[1] += F[1] / mass * TIME_STEP

        p[0] += v[0] * TIME_STEP
        p[1] += v[1] * TIME_STEP

        Xs += [p[0]]
        Ys += [p[1]]

    return Xs, Ys

def optimize(v_0, mass, friction, density, STEP=0.01):
    theta = 0 # radians

    max_dist  = 0
    max_theta = 0

    while theta < PI / 2:
        Xs, Ys = run(theta, v_0, mass, friction, density)
        dist = max(Xs)

        if dist > max_dist:
            max_dist  = dist
            max_theta = theta
    
        theta += STEP
    
    return max_theta

Angles = []
Drag = []

for i in range(0, 10):
    drag = i / 10 
    Max = optimize(100, MASS, drag, 1.225)
    Angles.append(180 * Max / PI)
    Drag.append(drag)

def objective(x, a, b, c):
    return a * b**x + c

popt, _ = curve_fit(objective, Drag, Angles)
a, b, c = popt

x_line = np.arange(0, 1, 0.05)
y_line = objective(x_line, a, b, c)

print(f"found f(x) = {a}*{b}**x + {c}")

plt.plot(Drag, Angles)
plt.plot(x_line, y_line)
plt.show()
