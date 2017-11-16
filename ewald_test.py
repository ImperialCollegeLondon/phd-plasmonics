#! /usr/bin/python3

import numpy as np
import scipy as sp
from scipy import special  # used for hankel functions
from scipy import optimize
from matplotlib import pyplot as plt
from multiprocessing import Pool
import itertools


def latticeGenerator(a1, a2, number, origin):
    neighbours = np.arange(-number, number+1)
    for n, m in itertools.product(neighbours, repeat=2):
        if origin == False:
            if n != 0 or m != 0:
                yield n*np.array(a1) + m*np.array(a2)
        else:
                yield n*np.array(a1) + m*np.array(a2)


def green(distance, k, q):
    return 0.25j*sp.special.hankel1(0, k*np.linalg.norm(distance))*np.exp(-1j*np.dot(q, distance))


def _green(args):
    return green(*args)


def latticeSum(a1, a2, k, q, lat_size):
    lattice = latticeGenerator(a1, a2, lat_size, False)
    results = []
    pool = Pool()
    values = [(i, k, q) for i in lattice]
    results.append(pool.map(_green, values))
    pool.close()
    return sum(results[0])


def plotSum(a1, a2, k, q, size):
    result = []
    for i in range(1, size):
        print(i, i**2-1)
        result.append(latticeSum(a1, a2, k, q, i))
    fig, ax = plt.subplots(2)
    avg_real = sum([i.real for i in result])/len(result)
    avg_imag = sum([i.imag for i in result])/len(result)

    ax[0].plot(np.arange(1, size)**2 - 1, [i.real for i in result], c='r')
    ax[0].plot([0, size**2-1], [avg_real, avg_real])
    ax[1].plot(np.arange(1, size)**2 - 1, [i.imag for i in result], c='b')
    ax[1].plot([0, size**2-1], [avg_imag, avg_imag])

    plt.show()


def green_dyadic(k, distance):
    """
    Green's function interaction.

    Calculate the Green's function tensor between particles which are separated
    by a vector distance at a frequency k. For a 2D Green's function, the
    interactions are modelled with Hankel functions.

    Returns a matrix of the form [[G_xx, G_xy],[G_xy, G_yy]]
    """
    x = distance[0]
    y = distance[1]
    R = np.linalg.norm(distance)
    arg = k*R

    xx_type = 0.25j * k**2 * (
    (y**2/R**2) * sp.special.hankel1(0,arg) +
    (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg)
    )

    yy_type = 0.25j * k**2 * (
    (x**2/R**2) * sp.special.hankel1(0,arg) -
    (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg)
    )

    xy_type = 0.25j * k**2 * x*y/R**2 * sp.special.hankel1(2,arg)

    return np.array([[xx_type, xy_type], [xy_type, yy_type]])


def lattice_sum(a1, a2, size, k, q):
    lattice = latticeGenerator(a1, a2, size, False)
    _sum = 0
    for position in lattice:
        _sum += green_dyadic(k, position) * np.exp(-1j * np.dot(q, position))
    return _sum


def getPolarisability(k, wp, loss, radius):
    """
    Return the polarisability for a particle.

    """
    w = k / ev

    permittivity = 1 - (wp**2)/(w**2 - 1j*loss*w)
    eps = (permittivity-1)/(permittivity+1)

    return 2*np.pi*radius**2 * eps * 1/(1 - 0.25j*np.pi*(k*radius)**2 * eps)


class EwaldSum:
    def __init__(self, a1, a2, b_lat_size, r_lat_size, k, q, origin, ewald, accuracy):
        self.bravais = list(latticeGenerator(a1, a2, b_lat_size, False))
        self.k = k
        self.q = q
        self.E = ewald
        self.j_max = accuracy
        self.r = origin
        R = np.array([[0, -1],[1, 0]])
        b1 = 2*np.pi * np.dot(R, a2)/np.dot(a1, np.dot(R, a2))
        b2 = 2*np.pi * np.dot(R, a1)/np.dot(a2, np.dot(R, a1))

        self.recip = list(latticeGenerator(b1, b2, r_lat_size, True))

    def sumG1(self):
        area = float(np.cross(a1, a2))
        _sum = 0
        for position in self.recip:
            _sum += (1./area) * np.exp(-1j*np.dot(self.q+position, self.r)) * np.exp((self.k**2 - np.linalg.norm(self.q+position)**2)/(4*self.E**2))/(self.k**2 - np.linalg.norm(self.q+position)**2)
        return _sum

    def integralFunc(self, separation):
        _sum = 0
        for j in range(0, self.j_max+1):
            _sum += (1./np.math.factorial(j)) * (self.k/(2*self.E))**(2*j) * sp.special.expn(j+1, separation**2*self.E**2)
        return _sum

    def integralFunc_deriv(self, separation):
        _sum = 0

        for j in range(1, self.j_max+1):
            _sum += (1./np.math.factorial(j)) * (self.k/2)**(2*j) * (self.E)**(4-(2*j)) * sp.special.expn(j-1, separation**2*self.E**2)
        _sum += (separation**2*self.E**2+1)*np.exp(-separation**2 * self.E**2)/separation**4
        return _sum

    def sumG2(self):
        _sum = 0
        for position in self.bravais:
            distance = np.linalg.norm(position + self.r)
            _sum += (1./(2*np.pi)) * np.exp(-1j * np.dot(self.q, self.r + position)) * self.integralFunc(distance)
        return _sum

    def total_sum(self):
        return self.sumG1() + self.sumG2()

    def sumG1_deriv(self, _type):
        area = float(np.cross(a1, a2))
        _sum = 0
        for position in self.recip:
            if _type == "xx":
                _sum += -(1./area) * (self.q[0] + position[0])**2 * np.exp(-1j*np.dot(self.q+position, self.r)) * np.exp((self.k**2 - np.linalg.norm(self.q+position)**2)/(4*self.E**2))/(self.k**2 - np.linalg.norm(self.q+position)**2)
            elif _type == "yy":
                _sum += -(1./area) * (self.q[1] + position[1])**2 * np.exp(-1j*np.dot(self.q+position, self.r)) * np.exp((self.k**2 - np.linalg.norm(self.q+position)**2)/(4*self.E**2))/(self.k**2 - np.linalg.norm(self.q+position)**2)
            elif _type == "xy":
                _sum += -(1./area) * (self.q[0] + position[0])*(self.q[1] + position[1]) * np.exp(-1j*np.dot(self.q+position, self.r)) * np.exp((self.k**2 - np.linalg.norm(self.q+position)**2)/(4*self.E**2))/(self.k**2 - np.linalg.norm(self.q+position)**2)
        return _sum

    def sumG2_deriv(self, _type):
        _sum = 0
        for position in self.bravais:
            distance = np.linalg.norm(position + self.r)
            if _type is "xx":
                _sum += -self.q[0]**2 * 4*(position + self.r)[0]**2 * (1./(2*np.pi)) * np.exp(-1j * np.dot(self.q, self.r + position)) * self.integralFunc_deriv(distance)
            elif _type is "yy":
                _sum += -self.q[1]**2 * 4*(position + self.r)[1]**2 * (1./(2*np.pi)) * np.exp(-1j * np.dot(self.q, self.r + position)) * self.integralFunc_deriv(distance)
            elif _type is "xy":
                _sum += -self.q[0] * self.q[1] * 4*(position + self.r)[0] * (position + self.r)[1] * (1./(2*np.pi)) * np.exp(-1j * np.dot(self.q, self.r + position)) * self.integralFunc_deriv(distance)
        return _sum

    def total_deriv(self):
        xx_comp = self.sumG1_deriv("xx") + self.sumG2_deriv("xx")
        yy_comp = self.sumG1_deriv("yy") + self.sumG2_deriv("yy")
        xy_comp = self.sumG1_deriv("xy") + self.sumG2_deriv("xy")
        return np.array([[xx_comp, xy_comp],[xy_comp, yy_comp]])

    def dyadic_sum(self):
        return self.k**2*self.total_sum()*np.identity(2) + self.total_deriv()

    def extinction(self):
        matrix = self.dyadic_sum() - (np.identity(2) * 1/getPolarisability(self.k, wp, loss, radius))
        return 4*np.pi*self.k*(sum(1/sp.linalg.eigvals(matrix)).imag)


def getReciprocalLattice(size, spacing):
    """
    Generate set of reciprocal lattice points from
    Gamma to X to M to Gamma.

    args:
    - size: number of points to create
    """

    Gamma_X_x = np.linspace(0, np.pi/spacing, size/3,
                            endpoint=False)
    Gamma_X_y = np.zeros(int(size/3))

    X_M_x = np.ones(int(size/3))*np.pi/spacing
    X_M_y = np.linspace(0, np.pi/spacing, size/3, endpoint=False)

    M_Gamma_x = np.linspace(np.pi/spacing, 0, size/3, endpoint=True)
    M_Gamma_y = np.linspace(np.pi/spacing, 0, size/3, endpoint=True)

    q_x = np.concatenate((Gamma_X_x, X_M_x, M_Gamma_x))
    q_y = np.concatenate((Gamma_X_y, X_M_y, M_Gamma_y))

    return np.array(list(zip(q_x, q_y)))


def plotConvergence():
    summations = []
    k = wp*ev
    q = qrange[20]
    for i in range(1,10):
        print(i)
        summations.append(EwaldSum(a1, a2, i, i, k, q, pos, np.pi/a, 2).dyadic_sum())
    fig, ax = plt.subplots(3,2)
    ax[0][0].plot([i[0][0].real for i in summations], c='r')
    ax[0][1].plot([i[0][0].imag for i in summations], c='r', alpha=0.5)

    ax[1][0].plot([i[0][1].real for i in summations], c='b')
    ax[1][1].plot([i[0][1].imag for i in summations], c='b', alpha=0.5)

    ax[2][0].plot([i[1][1].real for i in summations], c='g')
    ax[2][1].plot([i[1][1].imag for i in summations], c='g', alpha=0.5)

    plt.show()


if __name__ == '__main__':
    ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)
    a = 30*10**-9
    radius = 10*10**-9
    wp = 4
    loss = 0.05
    a1 = np.array([0, a])
    a2 = np.array([a, 0])
    pos = np.array([0, 0])

    res = 60

    result = []
    #qrange = getReciprocalLattice(res, a)
    qrange = list(zip(np.zeros(res), np.linspace(-2*np.pi/a,2*np.pi/a, res)))
    wmin = 1
    wmax = 10
    for w in np.linspace(wmin,wmax,res):
        print(w)
        for q in qrange:
            k = w*ev
            ewald = EwaldSum(a1, a2, 6, 6, k, q, pos, 2*np.pi/a, 2)
            result.append(ewald.extinction())

    result = np.array(result).reshape((res,res))

    light_line = [(np.linalg.norm(qval)/ev) for q, qval in enumerate(qrange)]
    plt.plot(light_line, 'r--', zorder=1, alpha=0.5)

    plt.imshow(result, origin="lower", extent=[0, res-1, wmin, wmax], aspect='auto')
    plt.show()

    #plotConvergence()
