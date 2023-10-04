import matplotlib.pyplot as plt
import numpy as np

PI = 3.14159

# simulation timestep in (s)
DT = 0.01 
# gravitational constant in (s)
G  = 9.81 
# initial velocity (m / s)
V  = 10
# initial launch angle (r)
THETA = PI / 2

# current position
pos = (0, 0)
# current velocity
vel = (0, 0)

t = 0
dt = 0.01
v = [100, 100]
p = [0, 0]
mass = 1

Ts = []
Xs = []
Ys = []

while True:

    Xs.append(p[0])
    Ys.append(p[1])
    Ts.append(t)

    # Fd = (1/2) * A * v^2 * c * d
    # c -> coefficient of air resistance
    # d -> density of fluid
    # A -> cross sectional area of the mass
    # v -> velocity of the mass

    Dx = np.sign(v[0]) * 0.2 * v[0]**2 # 0.5 * 1.255 * v[0]**2 * 0.2
    Dy = np.sign(v[1]) * 0.2 * v[1]**2 # 0.5 * 1.255 * v[1]**2 * 0.2
   
    F = [-Dx, -G-Dy]

    v[0] += F[0] / mass * dt
    v[1] += F[1] / mass * dt
    
    p[0] += v[0] * dt
    p[1] += v[1] * dt

    t += dt

    if v[1] < 0 and p[1] <= 0:
        break;

plt.plot(Ts, Xs)
plt.show()

