import math

import numpy as np
import sympy
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from matplotlib.animation import FuncAnimation

step = 1000

t = np.linspace(0, 10, step)

phi = np.cos(t)
psi = np.sin(t)

r1 = 1
r2 = 0.125
r3 = 0.05
a = r1 - r3
b = r1 - r2

# Анимация
fig = plt.figure()
gr = fig.add_subplot(1, 1, 1)
gr.axis('equal')

def rotation2D(x, y, angle):
    Rx = x * np.cos(angle) - y * np.sin(angle)
    Ry = x * np.sin(angle) + y * np.cos(angle)
    return Rx, Ry

Ox, Oy = 0, 1

pO = plt.Circle((Ox, Oy), r3, color='black')

C1x = np.linspace(0, 10, step)
C1y = np.linspace(0, 10, step)

for i in range(len(phi)):
    Rx, Ry = rotation2D(0, -a, phi[i])
    C1x[i] = Ox + Rx
    C1y[i] = Oy + Ry

C2x = np.linspace(0, 10, step)
C2y = np.linspace(0, 10, step)

for i in range(len(phi)):
    Rx, Ry = rotation2D(0, -b, psi[i])
    C2x[i] = C1x[i] + Rx
    C2y[i] = C1y[i] + Ry

gr.add_patch(pO)

A1 = plt.Circle((C1x[0], C1y[0]), r1, color='black', fill=False)
A2 = plt.Circle((C2x[0], C2y[0]), r2, color='black', fill=False)

gr.add_patch(A1)
gr.add_patch(A2)

def animate(i):
    A1.center = (C1x[i], C1y[i])
    A2.center = (C2x[i], C2y[i])

gr.set(xlim=[-2, 2], ylim=[-2, 2])

anim = FuncAnimation(fig, animate, frames = step, interval = 1)

plt.show()