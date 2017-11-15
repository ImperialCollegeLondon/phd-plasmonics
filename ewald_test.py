#! /usr/bin/python3

import numpy as np
import scipy as sp
from scipy import special  # used for hankel functions
from scipy import optimize
from matplotlib import pyplot as plt
from multiprocessing import Pool
import itertools


def latticeGenerator(a1, a2, number):
    neighbours = np.arange(-number, number+1)
    for n, m in itertools.product(neighbours, repeat=2):
        if n!=0 or m!=0:
            yield n*np.array(a1) + m*np.array(a2)


def green(distance, k, q):
    return 0.25j*sp.special.hankel1(0, k*np.linalg.norm(distance))*np.exp(-1j*np.dot(q, distance))


def _green(args):
    return green(*args)


def latticeSum(lattice, k, q):
    results = []
    pool = Pool()
    values = [(i, k, q) for i in lattice]
    results.append(pool.map(_green, values))
    pool.close()
    return sum(results[0])


def plotSum(k, q, size):
    result = []
    for i in range(1,size):
        print(i, i**2-1)
        lattice = latticeGenerator([a,0],[0,a],i)
        result.append(latticeSum(lattice, k, q))
    plt.plot(np.arange(1,size)**2 -1, [i.real for i in result])
    plt.plot(np.arange(1,size)**2 -1, [i.imag for i in result])
    plt.show()


def integralFunc(j_range, ewald, k, r, lattice):
    _sum = 0
    for j in range(0, j_range):
        _sum += (1./np.math.factorial(j)) * (k/(2*ewald))**(2*j) * sp.special.expn(j+1, r**2*ewald**2)
    return _sum





if __name__ == '__main__':
    ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)
    a = 30*10**-9
    k = 3.5 * ev
    q = np.array([0, 0])

    #plotSum(k, q, 50)

    result = []
    for i in range(1, 20):
        result.append(integralFunc(i, np.pi/a, k, a))
    print(result)
    plt.plot(result)
    plt.show()
