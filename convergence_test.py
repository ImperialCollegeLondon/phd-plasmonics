#! python3

"""Testing out methods for convering Green's function lattice sums"""

import numpy as np
import scipy as sp
from scipy import special
from multiprocessing import Pool
import itertools
from matplotlib import pyplot as plt


def square(spacing, size):
    """Define a simple square lattice. Returns array of lattice points"""
    particles = []
    size_range = np.arange(-size, size+1)
    for i, j in itertools.product(size_range, size_range):
        if i != 0 or j != 0:
            particles.append([i*spacing, j*spacing])
    return np.array(particles)

def green(k, q, r):
    return (-0.25j * sp.special.hankel1(0, k * np.linalg.norm(r) * np.exp(-1j * np.dot(q, r))))

def green_wrap(args):
    return green(*args)

def lattice_sum(k, q, particles):
    res = []
    points = [(k, q, r) for r in particles]
    pool = Pool(16)
    res.append(pool.map(green_wrap, points))
    final = sum(res[0])
    pool.close()
    return final

if __name__ == '__main__':
    k = 2.82*10**6
    a = 30*10**-8
    q = np.array([2*np.pi/a, 0])

    num_particles = []
    summation = []
    avg = []
    for num in range(1,200):
        print(num)
        particles = square(a, num)
        num_particles.append(len(particles))
        result = lattice_sum(k, q, particles)
        summation.append(result)
        avg.append(sum(summation)/len(summation))

    fig, ax = plt.subplots(2)
    ax[0].plot(num_particles, [i.real for i in summation])
    ax[0].plot(num_particles, [i.real for i in avg])
    ax[1].plot(num_particles, [i.imag for i in summation])
    ax[1].plot(num_particles, [i.imag for i in avg])

    plt.show()
