#!/usr/bin/python3

import numpy as np
import scipy as sp
from scipy import special  # used for hankel functions
from scipy import optimize
from matplotlib import pyplot as plt
from multiprocessing import Pool


class Particle:
    def __init__(self, x_pos, y_pos, radius, wp, loss):
        self.R = np.array([x_pos, y_pos])
        self.radius = radius
        self.plasma = wp
        self.loss = loss


def green(k, distance):
    x = distance[0]
    y = distance[1]
    R = np.linalg.norm(distance)
    arg = k*R

    xx_type = 0.25j * (
    k**2 * (sp.special.hankel1(0, arg))
    - (k/R) * sp.special.hankel1(1, arg)
    + ((k**2 * x**2)/(2*R**2)) * (sp.special.hankel1(2, arg) - sp.special.hankel1(0, arg))
    + ((k * x**2)/(R**3)) * sp.special.hankel1(1, arg)
    )

    xy_type = 0.25j * sp.special.hankel1(2, arg) * ((k**2 * x*y)/(R**2))

    yy_type = 0.25j * (
    k**2 * (sp.special.hankel1(0, arg))
    - (k/R) * sp.special.hankel1(1, arg)
    + ((k**2 * y**2)/(2*R**2)) * (sp.special.hankel1(2, arg) - sp.special.hankel1(0, arg))
    + ((k * y**2)/(R**3)) * sp.special.hankel1(1, arg))

    return np.array([[xx_type, xy_type], [xy_type, yy_type]])


def honeycomb(spacing, radius, wp, loss):
    particle_coords = []

    particle_coords.append(Particle(spacing, 0, radius, wp, loss).R)
    particle_coords.append(Particle(spacing*0.5, -spacing*np.sqrt(3)/2, radius, wp, loss).R)
    particle_coords.append(Particle(-spacing*0.5, -spacing*np.sqrt(3)/2, radius, wp, loss).R)
    particle_coords.append(Particle(-spacing, 0, radius, wp, loss).R)
    particle_coords.append(Particle(-spacing*0.5, spacing*np.sqrt(3)/2, radius, wp, loss).R)
    particle_coords.append(Particle(spacing*0.5, spacing*np.sqrt(3)/2, radius, wp, loss).R)

    return np.array(particle_coords)


def interactions(cell, spacing, sum_range, a1, a2, k, q):
    H = np.zeros((12, 12), dtype=np.complex_)
    i = 1

    intercell = supercell(spacing, cell, a1, a2, 1)[0]

    for n in np.arange(len(cell)):
        for m in np.arange(start=i, stop=len(cell)):
            H[2*n:2*n+2, 2*m:2*m+2] = sum([green(k, cell[n] - cell[m] + inter) for inter in intercell])
        i += 1

    return H + np.conjugate(H).T


def supercell(a, cell, t1, t2, max):
    pos = cell

    points = []
    particles = []

    for n in np.arange(-max, max+1):
        if n < 0:
            for m in np.arange(-n-max, max+1):
                points.append(n*t1+m*t2)
        elif n > 0:
            for m in np.arange(-max, max-n+1):
                points.append(n*t1+m*t2)
        elif n == 0:
            for m in np.arange(-max, max+1):
                points.append(n*t1+m*t2)

    for p in points:
        for g in pos:
            particles.append(p + g)

    return [points, particles]


if __name__ == "__main__":
    a = 30.*10**-9
    r = 10.*10**-9
    wp = 3.5
    g = 0.01
    trans_1 = np.array([3*a, 0])
    trans_2 = np.array([3*a/2, np.sqrt(3)*3*a/2])

    intracell = honeycomb(a, r, wp, g)

    intercell = supercell(a, intracell, trans_1, trans_2, 0)

    i=1
    to_plot = []
    for n in np.arange(len(intracell)):
        for m in np.arange(len(intracell)):
            for inter in intercell[0]:
                point = intracell[m] + inter
                to_plot.append([[intracell[n][0], point[0]], [intracell[n][1], point[1]]])
        i+=1

print(to_plot[0])
for i in to_plot:
    plt.plot(i[0],i[1],zorder=0, alpha=0.1, c='r')
for i in intercell[1]:
    plt.scatter(i[0],i[1],c='k',zorder=1, alpha=0.1)
plt.show()
