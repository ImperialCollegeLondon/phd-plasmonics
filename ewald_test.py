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
        values = [(w, np.array(self.pos-i)) for i in self.bravais]
        results.append(pool.map(self._monopolar_green, values))
        pool.close()
        return sum(results[0])

    def green_dyadic(self, w, distance):
        k = w*ev

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

        return np.array([xx_type, xy_type, yy_type])

    def _green_dyadic(self, args):
        return self.green_dyadic(*args)

    def dyadicSum(self, w):
        results = []
        pool = Pool()
        values = [(w, np.array(self.pos-i)) for i in self.bravais]
        results.append(pool.map(self._green_dyadic, values))
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
            _sum += -(1./area) * (np.exp(1j*np.dot(self.q+G_pos, self.pos)) * np.exp((k**2 - np.linalg.norm(self.q+G_pos)**2)/(4*self.E**2)))/( np.linalg.norm(self.q+G_pos)**2 - k**2)
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
            _sum += -(1./(4*np.pi)) * np.exp(1j * np.dot(self.q, R_pos)) * self.integralFunc(distance, w)
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
        return -_sum

    def ewaldG1_deriv(self, w, type):
        k = w*ev
        a1, a2 = self.lattice.getBravaisVectors()

        area = float(np.cross(a1, a2))
        _sum = 0
        for G_pos in self.reciprocal:
            if type is 'xx':
                _sum += -(self.q+G_pos)[0]**2 * (1./area) * (np.exp(1j*np.dot(self.q+G_pos, self.pos)) * np.exp((k**2 - np.linalg.norm(q+G_pos)**2)/(4*self.E**2)))/(np.linalg.norm(self.q+G_pos)**2 - k**2)
            elif type is 'yy':
                _sum += -(self.q+G_pos)[1]**2 * (1./area) * (np.exp(1j*np.dot(self.q+G_pos, self.pos)) * np.exp((k**2 - np.linalg.norm(q+G_pos)**2)/(4*self.E**2)))/(np.linalg.norm(self.q+G_pos)**2 - k**2)
            if type is 'xy':
                _sum += -(self.q+G_pos)[0]*(self.q+G_pos)[1] * (1./area) * (np.exp(1j*np.dot(self.q+G_pos, self.pos)) * np.exp((k**2 - np.linalg.norm(q+G_pos)**2)/(4*self.E**2)))/(np.linalg.norm(self.q+G_pos)**2 - k**2)
        return -_sum

    def derivSum(self, w):
        k = w*ev
        pre_factor = k**2*self.monopolarSum(w)
        xx_comp = pre_factor + self.ewaldG1_deriv(w, 'xx') + self.ewaldG2_deriv(w, 'xx')
        xy_comp = self.ewaldG1_deriv(w, 'xy') + self.ewaldG2_deriv(w, 'xy')
        yy_comp = pre_factor + self.ewaldG1_deriv(w, 'yy') + self.ewaldG2_deriv(w, 'yy')
        return [xx_comp, xy_comp, yy_comp]


def testLatticeSum(vector_1, vector_2, neighbour_range, q, w, pos):
    results = []
    ewald_results = []
    loop_range = range(1, neighbour_range+1)
    lattice = Lattice(vector_1, vector_2)
    for i in loop_range:
        results.append(Interaction(q, lattice, i, pos).monopolarSum(w))
        print(i**2-1)
    print("done")
    for i in range(1, 20):
        ewald_results.append(Ewald(2*np.pi/(a), 5, q, lattice, i, pos).monopolarSum(w))

    fig, ax = plt.subplots(2,2)
    ax[0][0].set_title("Non-Ewald, Re")
    ax[1][0].set_title("Non-Ewald, Im")
    ax[0][1].set_title("Ewald, Re")
    ax[1][1].set_title("Ewald, Im")

    ax[0][0].plot([i**2-1 for i in loop_range], [i.real for i in results],'r')
    ax[1][0].plot([i**2-1 for i in loop_range], [i.imag for i in results],'r--')
    print(sum(results)/len(results))

    ax[0][1].plot([i**2-1 for i in range(1, 20)], [i.real for i in ewald_results],'g')
    ax[1][1].plot([i**2-1 for i in range(1, 20)], [i.imag for i in ewald_results],'g--')
    print(ewald_results[18])
    fig.text(0.5, 0.04, 'Number of terms in sum', ha='center')

    plt.show()


def testDyadicSum(vector_1, vector_2, neighbour_range, q, w, pos):
    results = []
    ewald_results = []
    loop_range = range(1, neighbour_range+1)
    lattice = Lattice(vector_1, vector_2)
    for i in loop_range:
        results.append(Interaction(q, lattice, i, pos).dyadicSum(wp))
        ewald_results.append(Ewald(2*np.pi/a, 5, q, lattice, i, pos).derivSum(wp))
        print(i**2-1)
    print("done")

    fig, ax = plt.subplots(3,4)
    for j in range(3):

        ax[0][0].set_title("Non-Ewald, Re")
        ax[0][1].set_title("Non-Ewald, Im")
        ax[0][2].set_title("Ewald, Re")
        ax[0][3].set_title("Ewald, Im")

        ax[0][0].set_ylabel("$\partial^2/\partial x^2$ sum")
        ax[1][0].set_ylabel("$\partial^2/\partial x \partial y$ sum")
        ax[2][0].set_ylabel("$\partial^2/\partial y^2$ sum")

        ax[j][0].plot([i**2-1 for i in loop_range], [i[j].real for i in results], 'r')
        ax[j][1].plot([i**2-1 for i in loop_range], [i[j].imag for i in results], 'r--')
        ax[j][2].plot([i**2-1 for i in loop_range], [i[j].real for i in ewald_results], 'g')
        ax[j][3].plot([i**2-1 for i in loop_range], [i[j].imag for i in ewald_results], 'g--')
    fig.text(0.5, 0.04, 'Number of terms in sum', ha='center')
    print(sum(results)/len(results))

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

    #testDyadicSum(a1, a2, 50, q, wp, pos)
    testLatticeSum(a1, a2, 20, q, wp, pos)
