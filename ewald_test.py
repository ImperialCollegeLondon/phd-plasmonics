#! /usr/bin/python3

import numpy as np
import scipy as sp
from scipy import special  # used for hankel functions
from scipy import optimize
from matplotlib import pyplot as plt
from multiprocessing import Pool
import itertools

global ev
ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)


class Lattice:
    def __init__(self, a1, a2):
        R = np.array([[0, -1],[1, 0]])
        self.a1 = a1
        self.a2 = a2
        self.b1 = 2*np.pi * np.dot(R, self.a2)/np.dot(self.a1, np.dot(R, self.a2))
        self.b2 = 2*np.pi * np.dot(R, self.a1)/np.dot(self.a2, np.dot(R, self.a1))

    def genBravais(self, number, origin):
        bravais = []
        neighbours = np.arange(-number, number + 1)
        for n, m in itertools.product(neighbours, repeat=2):
            if origin is False:
                if n != 0 or m != 0:
                    bravais.append(n*np.array(self.a1) + m*np.array(self.a2))
            else:
                    bravais.append(n*np.array(self.a1) + m*np.array(self.a2))
        return bravais

    def genReciprocal(self, number, origin):
        reciprocal = []
        neighbours = np.arange(-number, number + 1)

        for n, m in itertools.product(neighbours, repeat=2):
            if origin is False:
                if n != 0 or m != 0:
                    reciprocal.append(n*np.array(self.b1) + m*np.array(self.b2))
            else:
                    reciprocal.append(n*np.array(self.b1) + m*np.array(self.b2))
        return reciprocal

    def getBravaisVectors(self):
        return [self.a1, self.a2]

    def getReciprocalVectors(self):
        return [self.b1, self.b2]


class Interaction:
    def __init__(self, q, lattice, neighbours, position):
        """
        args:
        - q: point in reciprocal space
        - lattice: set of lattice points
        - position: point in real space
        """
        self.q = q
        self.lattice = lattice
        self.bravais = lattice.genBravais(neighbours, True)
        self.reciprocal = lattice.genReciprocal(neighbours, True)
        self.pos = position

    def monopolar_green(self, w, distance):
        k = w*ev
        return 0.25j*sp.special.hankel1(0, k*np.linalg.norm(distance))*np.exp(1j*np.dot(self.q, distance))

    def _monopolar_green(self, args):
        # wrapper for multiprocessing
        return self.monopolar_green(*args)

    def monopolarSum(self, w):
        results = []
        pool = Pool()
        values = [(w, np.array(i+self.pos)) for i in self.bravais]
        results.append(pool.map(self._monopolar_green, values))
        pool.close()
        return sum(results[0])


class Ewald:
    def __init__(self, ewald, j_max, q, lattice, neighbours, position):
        self.q = q
        self.lattice = lattice
        self.bravais = lattice.genBravais(neighbours, True)
        self.reciprocal = lattice.genReciprocal(neighbours, True)
        self.pos = position
        self.E = ewald
        self.j_max = j_max

    def ewaldG1(self, w):
        k = w*ev
        a1, a2 = self.lattice.getBravaisVectors()

        area = float(np.cross(a1, a2))
        _sum = 0
        for G_pos in self.reciprocal:
            _sum += (1./area) * (np.exp(1j*np.dot(q+G_pos, self.pos)) * np.exp((k**2 - np.linalg.norm(q+G_pos)**2)/(4*self.E**2)))/(np.linalg.norm(q+G_pos)**2 - k**2)
        return _sum

    def integralFunc(self, separation, w):
        k = w*ev
        _sum = 0
        for j in range(0, self.j_max+1):
            _sum += (1./np.math.factorial(j)) * (k/(2*self.E))**(2*j) * sp.special.expn(j+1, separation**2*self.E**2)
        return _sum

    def ewaldG2(self, w):
        _sum = 0
        for R_pos in self.bravais:
            distance = np.linalg.norm(self.pos - R_pos)
            _sum += (1./(4*np.pi)) * np.exp(1j * np.dot(self.q, R_pos)) * self.integralFunc(distance, w)
        return _sum

    def monopolarSum(self, w):
        return self.ewaldG1(w) + self.ewaldG2(w)

    def ewaldG2_deriv(self, w, type):
        k = w*ev
        _sum = 0
        for R_pos in self.bravais:
            rho = self.pos - R_pos
            n_rho = np.linalg.norm(rho)
            rho_E_sq = n_rho**2 * self.E**2  # rho^2 eta^2 product
            if type is 'xx':
                for j in range(0, self.j_max + 1):
                    if j == 0:
                        _sum += (1/np.math.factorial(j)) * (k/(2*self.E))**(2*j) * (-self.q[0]**2 * sp.special.expn(j+1, rho_E_sq) - (2*self.E**2 + 4j*self.E**2*self.q[0]*rho[0])*sp.special.expn(j, rho_E_sq))
                    else:
                        _sum += (1/np.math.factorial(j)) * (k/(2*self.E))**(2*j) * (-self.q[0]**2 * sp.special.expn(j+1, rho_E_sq) - (2*self.E**2 + 4j*self.E**2*self.q[0]*rho[0])*sp.special.expn(j, rho_E_sq) +
                        4*self.E**4*rho[0]**2*sp.special.expn(j-1, rho_E_sq))
                _sum += (4/n_rho**4)*rho[0]**2*(rho_E_sq + 1)*np.exp(-rho_E_sq)
            elif type is 'yy':
                for j in range(0, self.j_max + 1):
                    if j == 0:
                        _sum += (1/np.math.factorial(j)) * (k/(2*self.E))**(2*j) * (-self.q[1]**2 * sp.special.expn(j+1, rho_E_sq) - (2*self.E**2 + 4j*self.E**2*self.q[1]*rho[1])*sp.special.expn(j, rho_E_sq))
                    else:
                        _sum += (1/np.math.factorial(j)) * (k/(2*self.E))**(2*j) * (-self.q[1]**2 * sp.special.expn(j+1, rho_E_sq) - (2*self.E**2 + 4j*self.E**2*self.q[1]*rho[1])*sp.special.expn(j, rho_E_sq) + 4*self.E**4*rho[1]**2*sp.special.expn(j-1, rho_E_sq))
                _sum += (4/n_rho**4)*rho[1]**2*(rho_E_sq + 1)*np.exp(-rho_E_sq)
            elif type is 'xy':
                for j in range(0, self.j_max + 1):
                    if j == 0:
                        _sum += (1/np.math.factorial(j)) * (k/(2*self.E))**(2*j) * (-self.q[0]*self.q[1] * sp.special.expn(j+1, rho_E_sq) - 2j*self.E**2*(self.q[0]*rho[1] + self.q[1]*rho[0])*sp.special.expn(j, rho_E_sq))
                    else:
                        _sum += (1/np.math.factorial(j)) * (k/(2*self.E))**(2*j) * (-self.q[0]*self.q[1] * sp.special.expn(j+1, rho_E_sq) - 2j*self.E**2*(self.q[0]*rho[1] + self.q[1]*rho[0])*sp.special.expn(j, rho_E_sq) + 4*self.E**4*rho[0]*rho[1]*sp.special.expn(j-1, rho_E_sq))
                _sum += (4/n_rho**4)*rho[0]*rho[1]*(rho_E_sq + 1)*np.exp(-rho_E_sq)
            _sum = _sum*(1/(4*np.pi))*np.exp(1j*np.dot(self.q, R_pos))
        return _sum

def testLatticeSum(vector_1, vector_2, neighbour_range, inc_origin, q, w, pos):
    results = []
    ewald_results = []
    loop_range = range(1, neighbour_range+1)
    lattice = Lattice(a1, a2)
    for i in loop_range:
        results.append(Interaction(q, lattice, i, pos).monopolarSum(w))
        print(i**2-1)
    print("done")
    for i in range(1, 20):
        ewald_results.append(Ewald(2*np.pi/(a), 5, q, lattice, i, pos).monopolarSum(w))

    fig, ax = plt.subplots(2,2)
    ax[0][0].plot([i**2-1 for i in loop_range], [i.real for i in results])
    ax[0][1].plot([i**2-1 for i in loop_range], [i.imag for i in results])
    print(sum(results)/len(results))

    ax[1][0].plot([i**2-1 for i in range(1, 20)], [i.real for i in ewald_results])
    ax[1][1].plot([i**2-1 for i in range(1, 20)], [i.imag for i in ewald_results])
    print(results[neighbour_range-1])
    plt.show()


if __name__ == '__main__':
    ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)
    a = 15*10**-9
    radius = 5*10**-9
    wp = 3.5
    loss = 0.01
    a1 = np.array([0, a])
    a2 = np.array([a, 0])
    pos = np.array([a/3, 0])
    q = np.array([0.1/a, 0])

    result = []
    lattice = Lattice(a1, a2)
    for i in range(1,20):
        result.append(Ewald(2*np.pi/a, 5, q, lattice, i, pos).ewaldG2_deriv(wp, 'xx'))
    plt.plot(result)
    print(result)
    plt.show()
