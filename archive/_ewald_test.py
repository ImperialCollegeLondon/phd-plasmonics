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


def green(distance, pos, k, q):
    return 0.25j*sp.special.hankel1(0, k*np.linalg.norm(pos+distance))*np.exp(1j*np.dot(q, distance))


def _green(args):
    return green(*args)


def latticeSum(a1, a2, k, q, lat_size):
    lattice = list(latticeGenerator(a1, a2, lat_size, True))
    results = []
    pool = Pool()
    values = [(i, pos, k, q) for i in lattice]
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

    print(avg_real, avg_imag)
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
        _sum += green_dyadic(k, position) * np.exp(1j * np.dot(q, position))
    return _sum


def getPolarisability(w, wp, loss, radius):
    """
    Return the polarisability for a particle.

    """
    k = w * ev

    permittivity = 1 - (wp**2)/(w**2 - 1j*loss*w)
    eps = (permittivity-1)/(permittivity+1)

    return 2*np.pi*radius**2 * eps * 1/(1 - 0.25j*np.pi*(k*radius)**2 * eps)


class EwaldSum:
    def __init__(self, a1, a2, b_lat_size, r_lat_size, origin, ewald, accuracy):
        self.bravais = list(latticeGenerator(a1, a2, b_lat_size, True))
        self.E = ewald
        self.j_max = accuracy
        self.r = origin
        R = np.array([[0, -1],[1, 0]])
        b1 = 2*np.pi * np.dot(R, a2)/np.dot(a1, np.dot(R, a2))
        b2 = 2*np.pi * np.dot(R, a1)/np.dot(a2, np.dot(R, a1))

        self.recip = list(latticeGenerator(b1, b2, r_lat_size, True))

    def returnRecip(self):
        return self.recip

    def sumG1(self, q, w):
        k = w*ev
        area = float(np.cross(a1, a2))
        _sum = 0
        for position in self.recip:
            _sum += (1./area) * (np.exp(1j*np.dot(q+position, self.r)) * np.exp((k**2 - np.linalg.norm(q+position)**2)/(4*self.E**2)))/(k**2 - np.linalg.norm(q+position)**2)
        return _sum

    def integralFunc(self, separation, q, w):
        k = w*ev
        _sum = 0
        for j in range(0, self.j_max+1):
            _sum += (1./np.math.factorial(j)) * (k/(2*self.E))**(2*j) * sp.special.expn(j+1, separation**2*self.E**2)
        return _sum

    def sumG2(self, q, w):
        _sum = 0
        for position in self.bravais:
            distance = np.linalg.norm(self.r - position)
            _sum += (1./(4*np.pi)) * np.exp(1j * np.dot(q, position)) * self.integralFunc(distance, q, w)
        return _sum

    def total_sum(self, q, w):
        #print(self.sumG1(q,w), self.sumG2(q,w))
        return self.sumG1(q,w) + self.sumG2(q,w)

    def sumG1_deriv(self, _type, q, w):
        _sum = 0
        area = float(np.cross(a1, a2))
        k = w*ev
        for position in self.recip:
            if _type == "xx":
                _sum += -(q+position)[0]**2 * (1./area) * (np.exp(1j*np.dot(q+position, self.r)) * np.exp((k**2 - np.linalg.norm(q+position)**2)/(4*self.E**2)))/(np.linalg.norm(q+position)**2 - k**2)
            elif _type == "yy":
                _sum += -(q+position)[1]**2 * (1./area) * (np.exp(1j*np.dot(q+position, self.r)) * np.exp((k**2 - np.linalg.norm(q+position)**2)/(4*self.E**2)))/(np.linalg.norm(q+position)**2 - k**2)
            elif _type == "xy":
                _sum += -(q+position)[0] * (q+position)[1] * (1./area) * (np.exp(1j*np.dot(q+position, self.r)) * np.exp((k**2 - np.linalg.norm(q+position)**2)/(4*self.E**2)))/(np.linalg.norm(q+position)**2 - k**2)
        print(_sum)
        return _sum

    def integralFunc_deriv(self, separation, _type, q, w):
        r_ewald_product = np.linalg.norm(separation)**2 * self.E**2
        k = w*ev
        if _type is "xx":
            _sum = 0
            pre = separation[0]**2
            for j in range(1, self.j_max+1):
                _sum += (1./np.math.factorial(j)) * (k/(2*self.E))**(2*j) * (-2*self.E**2*sp.special.expn(j, r_ewald_product) + 4*pre*self.E**2*sp.special.expn(j-1, r_ewald_product))
        elif _type is "xy":
            _sum = 0
            pre = separation[0] * separation[1]
            for j in range(1, self.j_max+1):
                _sum += (1./np.math.factorial(j)) * (k/(2*self.E))**(2*j) * (-2*self.E**2*sp.special.expn(j, r_ewald_product) + 4*pre*self.E**2*sp.special.expn(j-1, r_ewald_product))
        elif _type is "yy":
            _sum = 0
            pre = separation[1]**2
            for j in range(1, self.j_max+1):
                _sum += (1./np.math.factorial(j)) * (k/(2*self.E))**(2*j) * (-2*self.E**2*sp.special.expn(j, r_ewald_product) + 4*pre*self.E**2*sp.special.expn(j-1, r_ewald_product))

        return _sum

    def sumG2_deriv(self, __type, q, w):
        _sum = 0
        for position in self.bravais:
            position = np.array(position)
            distance = self.r - position
            r_ewald_product = np.linalg.norm(distance)**2 * self.E**2
            if __type is "xx":
                pre = distance[0]**2
                _sum += 1/(4*np.pi) * np.exp(1j*np.dot(q, position)) * (self.integralFunc_deriv(distance, __type, q, w) - 2*self.E**2*np.exp(-r_ewald_product)/r_ewald_product + 4*pre*(r_ewald_product+1)*np.exp(-r_ewald_product)/np.linalg.norm(distance)**4)
            elif __type is "xy":
                pre = distance[0] * distance[1]
                _sum += 1/(4*np.pi) * np.exp(1j*np.dot(q, position)) * (self.integralFunc_deriv(distance, __type, q, w) - 2*self.E**2*np.exp(-r_ewald_product)/r_ewald_product + 4*pre*(r_ewald_product+1)*np.exp(-r_ewald_product)/np.linalg.norm(distance)**4)
            elif __type is "yy":
                pre = distance[1]**2
                _sum += 1/(4*np.pi) * np.exp(1j*np.dot(q, position)) * (self.integralFunc_deriv(distance, __type, q, w) - 2*self.E**2*np.exp(-r_ewald_product)/r_ewald_product + 4*pre*(r_ewald_product+1)*np.exp(-r_ewald_product)/np.linalg.norm(distance)**4)

        return _sum

    def total_deriv(self, q, w):
        xx_comp = self.sumG1_deriv("xx", q, w) + self.sumG2_deriv("xx", q, w)
        yy_comp = self.sumG1_deriv("xx", q, w) + self.sumG2_deriv("yy", q, w)
        xy_comp = self.sumG1_deriv("xx", q, w) + self.sumG2_deriv("xy", q, w)
        return np.array([[xx_comp, xy_comp],[xy_comp, yy_comp]])

    def dyadic_sum(self, q, w):
        k = w * ev
        return k**2*self.total_sum(q, w)*np.identity(2) + self.total_deriv(q, w)

    def extinction(self, q, w):
        k = w * ev
        matrix = self.dyadic_sum(q, w) - (np.identity(2) * 1/getPolarisability(w, wp, loss, radius))
        return 4*np.pi*k*(sum(1/sp.linalg.eigvals(matrix)).imag)

    def recipSum(self, q, w):
        k = w*ev
        area = float(np.cross(a1, a2))
        _sum = 0
        for position in self.recip:
            _sum += (1./area) * (np.exp(1j*np.dot(q+position, self.r)))/(k**2 - np.linalg.norm(q+position)**2)
        return _sum

    def test(self, q, w):
        k = w*ev
        return self.recipSum(q, w) + 0.25j*sp.special.hankel1(0, k * np.linalg.norm(self.r))

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


def plotConvergence(size):
    summations = []
    w = wp
    k = w*ev
    q = qrange[0]
    for i in range(1,size):
        print(i)
        summations.append(EwaldSum(a1, a2, i, i, pos, 2*np.pi/(a), 5).total_sum(q, w))
        #summations.append(lattice_sum(a1, a2, i, k, q))
    fig, ax = plt.subplots(2)

    ax[0].plot([i**2-1 for i in range(1,size)], [i.real for i in summations], c='r')
    ax[1].plot([i**2-1 for i in range(1,size)], [i.imag for i in summations], c='b')
    print(summations[size-2])

    # fig, ax = plt.subplots(4,2)
    # ax[0][0].plot([i**2-1 for i in range(1,size)],[i[0][0].real for i in summations], c='r')
    # ax[0][1].plot([i**2-1 for i in range(1,size)],[i[0][0].imag for i in summations], c='r', alpha=0.5)
    #
    # ax[1][0].plot([i**2-1 for i in range(1,size)],[i[0][1].real for i in summations], c='b')
    # ax[1][1].plot([i**2-1 for i in range(1,size)],[i[0][1].imag for i in summations], c='b', alpha=0.5)
    #
    # ax[2][0].plot([i**2-1 for i in range(1,size)],[i[1][0].real for i in summations], c='k')
    # ax[2][1].plot([i**2-1 for i in range(1,size)],[i[1][0].imag for i in summations], c='k', alpha=0.5)
    #
    # ax[3][0].plot([i**2-1 for i in range(1,size)],[i[1][1].real for i in summations], c='g')
    # ax[3][1].plot([i**2-1 for i in range(1,size)],[i[1][1].imag for i in summations], c='g', alpha=0.5)

    plt.show()

    def _extinction(args):
        return EwaldSum(*args)


if __name__ == '__main__':
    ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)
    a = 15*10**-9
    radius = 5*10**-9
    wp = 3.5
    loss = 0.01
    a1 = np.array([0, a])
    a2 = np.array([a, 0])
    pos = np.array([a/3, 0])

    res = 90

    result = []
    qrange = getReciprocalLattice(res, a)
    #qrange = list(zip(np.zeros(res), np.linspace(3.5*np.pi/(np.sqrt(3)*a),4.5*np.pi/(np.sqrt(3)*a), res)))
    wmin = 2
    wmax = 3

    # for w in np.linspace(wmin,wmax,res):
    #     print(w)
    #     for q in qrange:
    #         k = w*ev
    #         ewald = EwaldSum(a1, a2, 5, 5, pos, 2*np.pi/a, 1)
    #         result.append(ewald.extinction(q, w))
    #
    # result = np.array(result).reshape((res,res))
    #
    # light_line = [(np.linalg.norm(qval)/ev) for q, qval in enumerate(qrange)]
    # plt.plot(light_line, 'r--', zorder=1, alpha=0.5)
    #
    # plt.imshow(result, origin="lower", extent=[0, res-1, wmin, wmax], aspect='auto')
    # plt.show()
    #plotSum(a1, a2, wp*ev, qrange[20], 50)
    plotConvergence(10)
