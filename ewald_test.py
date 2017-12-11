#! /usr/bin/python3

"""
Small script for testing Ewald's method on interactions via a 2D Green's function on a 2D lattice
"""

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
    """
    Quick class to set up a lattice defined by two lattice vectors a1, a2.
    """
    def __init__(self, a1, a2):
        R = np.array([[0, -1],[1, 0]])  # Rotation matrix for defining reciprocal lattice vectors
        self.a1 = a1
        self.a2 = a2
        self.b1 = 2*np.pi * np.dot(R, self.a2)/np.dot(self.a1, np.dot(R, self.a2))
        self.b2 = 2*np.pi * np.dot(R, self.a1)/np.dot(self.a2, np.dot(R, self.a1))

    def genBravais(self, number, origin):
        """
        Returns a list of Bravais lattice points. Origin is True or False depending on whether [0,0] point is required.
        """
        bravais = []
        neighbours = np.arange(-number, number + 1)
        for n, m in itertools.product(neighbours, repeat=2):
            if origin is False:
                if n != 0 or m != 0:  # to exclude origin
                    bravais.append(n*np.array(self.a1) + m*np.array(self.a2))
            else:
                    bravais.append(n*np.array(self.a1) + m*np.array(self.a2))
        return bravais

    def genReciprocal(self, number, origin):
        """
        Returns a list of reciprocal lattice points. Origin is True or False depending on whether [0,0] point is required.
        """
        reciprocal = []
        neighbours = np.arange(-number, number + 1)

        for n, m in itertools.product(neighbours, repeat=2):
            if origin is False:
                if n != 0 or m != 0:  # to exclude origin
                    reciprocal.append(n*np.array(self.b1) + m*np.array(self.b2))
            else:
                    reciprocal.append(n*np.array(self.b1) + m*np.array(self.b2))
        return reciprocal

    def getBravaisVectors(self):
        """
        Returns Bravais lattice vectors.
        """
        return [self.a1, self.a2]

    def getReciprocalVectors(self):
        """
        Returns reciprocal lattice vectors.
        """
        return [self.b1, self.b2]


class Interaction:
    """
    Defines interactions between particles via 2D Green's function.
    """
    def __init__(self, q, lattice, neighbours, position, origin):
        """
        args:
        - q: point in reciprocal space
        - lattice: set of lattice points
        - position: point in real space
        """
        self.q = q
        self.lattice = lattice
        self.bravais = lattice.genBravais(neighbours, origin)
        self.reciprocal = lattice.genReciprocal(neighbours, True)
        self.pos = position

    def monopolar_green(self, w, distance, lat_vec):
        """
        Usual Green's function term in 2D.
        """
        k = w*ev
        return 0.25j*sp.special.hankel1(0, k*np.linalg.norm(distance))*np.exp(1j*np.dot(self.q, lat_vec))

    def _monopolar_green(self, args):
        # wrapper for multiprocessing
        return self.monopolar_green(*args)

    def monopolarSum(self, w):
        """
        Calculates sums for monopolar Green's function over Bravais lattice.
        """
        # results = []
        # pool = Pool()  # multiprocessing speeds things up
        # values = [(w, np.array(self.pos - i), i) for i in self.bravais]
        # results.append(pool.map(self._monopolar_green, values))
        # pool.close()
        # return sum(results[0])

        _sum = 0
        for i in self.bravais:
            _sum += self.monopolar_green(w, np.array(self.pos - i), i)
        return _sum

    def green_dyadic(self, w, distance, lat_vec):
        """
        Dyadic Green's function term in 2D, for dipolar sources.
        """
        k = w*ev

        x = distance[0]
        y = distance[1]
        R = np.linalg.norm(distance)
        theta = np.angle(x + 1j*y)
        arg = k*R

        xx_type = 0.25j * (k**2 *sp.special.hankel1(0,arg) - ((k**2*x**2)/(2*R**2))*(sp.special.hankel1(0, arg) - sp.special.hankel1(2, arg)) + (k*x**2/R**3) * sp.special.hankel1(1, arg) - (k/R)*sp.special.hankel1(1, arg))

        xy_type = 0.25j * ((k**2*x*y)/(2*R**2))*sp.special.hankel1(2, arg)

        yy_type = 0.25j * (k**2 * sp.special.hankel1(0,arg) - ((k**2*y**2)/(2*R**2))*(sp.special.hankel1(0, arg) - sp.special.hankel1(2, arg)) + (k*y**2/R**3) * sp.special.hankel1(1, arg) - (k/R)*sp.special.hankel1(1, arg))

        # xx_type = 0.25j * k**2 * (
        # (y**2/R**2) * sp.special.hankel1(0,arg) +
        # (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg)
        # )
        #
        # yy_type = 0.25j * k**2 * (
        # (x**2/R**2) * sp.special.hankel1(0,arg) -
        # (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg)
        # )
        #
        # xy_type = 0.25j * k**2 * x*y/R**2 * sp.special.hankel1(2,arg)
        # a = 0.125j * k**2 * sp.special.hankel1(0, arg)
        # b = 0.125j * k**2 * sp.special.hankel1(2, arg) * np.sin(2*theta)
        # c = 0.125j * k**2 * sp.special.hankel1(2, arg) * np.cos(2*theta)
        #
        # xx_type = a + c
        # xy_type = b
        # yy_type = a - c

        return np.array([xx_type, xy_type, yy_type])

    def _green_dyadic(self, args):
        # wrapper for multiprocessing
        return self.green_dyadic(*args)

    # def dyadicSum(self, w):
    #     """
    #     Calculates sums for dyadic Green's function over Bravais lattice.
    #     """
    #     results = []
    #     pool = Pool()
    #     values = [(w, np.array(self.pos - i), i) for i in self.bravais]
    #     results.append(pool.map(self._green_dyadic, values))
    #     pool.close()
    #     return sum(results[0])

    def dyadicSum(self, w):
        _sum = 0
        for i in self.bravais:
            _sum += self.green_dyadic(w, np.array(self.pos - i), i)*np.exp(1j*np.dot(self.q, i))
        return _sum

class Ewald:
    # This class could benefit from some cleaning up as a lot of the expressions are long and messy.
    """
    Contains all the terms required in Ewald's summation.
    """
    def __init__(self, ewald, j_max, q, lattice, neighbours, position):
        self.q = q
        self.lattice = lattice
        self.bravais = lattice.genBravais(neighbours, True)
        self.reciprocal = lattice.genReciprocal(neighbours, True)
        self.pos = position
        self.E = ewald
        self.j_max = j_max
        self.neighbours = neighbours

    def ewaldG1(self, w):
        k = w*ev
        a1, a2 = self.lattice.getBravaisVectors()

        area = float(np.cross(a1, a2))
        _sum = 0
        for G_pos in self.reciprocal:
            _sum += (1./area) * (np.exp(1j*np.dot(self.q+G_pos, self.pos)) * np.exp((k**2 - np.linalg.norm(self.q+G_pos)**2)/(4*self.E**2)))/(np.linalg.norm(self.q+G_pos)**2 - k**2)
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

    def dyadicEwaldG1(self, w, _type):
        k = w*ev
        a1, a2 = self.lattice.getBravaisVectors()
        area = float(np.cross(a1, a2))
        _sum = 0
        if _type is "xx":
            for G_pos in self.reciprocal:
                beta = self.q + G_pos
                beta_norm = np.linalg.norm(beta)

                _sum += (k**2 + beta[0]**2) * (1./area) * (np.exp(1j*np.dot(beta, self.pos)) * np.exp((k**2 - beta_norm**2)/(4*self.E**2)))/(beta_norm**2 - k**2)
        elif _type is "xy":
            for G_pos in self.reciprocal:
                beta = self.q + G_pos
                beta_norm = np.linalg.norm(beta)

                _sum += (beta[0]*beta[1]) * (1./area) * (np.exp(1j*np.dot(beta, self.pos)) * np.exp((k**2 - beta_norm**2)/(4*self.E**2)))/(beta_norm**2 - k**2)
        elif _type is "yy":
            for G_pos in self.reciprocal:
                beta = self.q + G_pos
                beta_norm = np.linalg.norm(beta)
                _sum += (k**2 + beta[1]**2) * (1./area) * (np.exp(1j*np.dot(beta, self.pos)) * np.exp((k**2 - beta_norm**2)/(4*self.E**2)))/(beta_norm**2 - k**2)
        return _sum

    def dyadicEwaldG2(self, w, _type):
        k = w*ev
        _sum = 0
        if _type is "xx":
            for R_pos in self.bravais:
                rho = self.pos - R_pos
                rho_norm =  np.linalg.norm(rho)
                _sum += (np.exp(1j * np.dot(self.q, R_pos))) * (self.dyadicIntegralFunc(w, rho, _type) + (np.exp(-rho_norm**2*self.E**2)/rho_norm**2)*(((4*rho[0]**2)/rho_norm**2)*(rho_norm**2*self.E**2 + 1) - 2))

        elif _type is "xy":
            for R_pos in self.bravais:
                rho = self.pos - R_pos
                rho_norm =  np.linalg.norm(rho)
                _sum += np.exp(1j * np.dot(self.q, R_pos)) * (self.dyadicIntegralFunc(w, rho, _type) + (np.exp(-rho_norm**2*self.E**2) * (4*rho[0]*rho[1]/rho_norm**2)*(rho_norm**2*self.E**2 + 1)))

        elif _type is "yy":
            for R_pos in self.bravais:
                rho = self.pos - R_pos
                rho_norm =  np.linalg.norm(rho)
                _sum += (np.exp(1j * np.dot(self.q, R_pos))) * (self.dyadicIntegralFunc(w, rho, _type) + (np.exp(-rho_norm**2*self.E**2)/rho_norm**2)*(((4*rho[1]**2)/rho_norm**2)*(rho_norm**2*self.E**2 + 1) - 2))
        return _sum/(4*np.pi)

    def dyadicIntegralFunc(self, w, rho, _type):
        k = w*ev
        _sum = 0
        rho_norm = np.linalg.norm(rho)
        if _type is "xx":
            for j in range(1, self.j_max+1):
                _sum += ((1./np.math.factorial(j)) * (k/(2*self.E))**(2*j)) * (4*rho[0]**2*self.E**4*sp.special.expn(j-1, rho_norm**2*self.E**2) - 2*self.E**2*sp.special.expn(j, rho_norm**2*self.E**2))

        elif _type is "xy":
            for j in range(1, self.j_max+1):
                _sum += ((1./np.math.factorial(j)) * (k/(2*self.E))**(2*j)) * (4*rho[0]*rho[1]*self.E**4*sp.special.expn(j-1, rho_norm**2*self.E**2))

        elif _type is "yy":
            for j in range(1, self.j_max+1):
                _sum += ((1./np.math.factorial(j)) * (k/(2*self.E))**(2*j)) * (4*rho[1]**2*self.E**4*sp.special.expn(j-1, rho_norm**2*self.E**2) - 2*self.E**2*sp.special.expn(j, rho_norm**2*self.E**2))

        return _sum

    def dyadicSumEwald(self, w):
        k = w*ev
        xx_comp = (self.dyadicEwaldG1(w, "xx") + self.dyadicEwaldG2(w, "xx") + k**2*self.ewaldG2(w))
        xy_comp = (self.dyadicEwaldG1(w, "xy") + self.dyadicEwaldG2(w, "xy"))
        yy_comp = (self.dyadicEwaldG1(w, "yy") + self.dyadicEwaldG2(w, "yy") + k**2*self.ewaldG2(w))
        return [xx_comp, xy_comp, yy_comp]

    def t0(self, w, n):
        k = w*ev
        if n == 0:
            return (-1 - (1j/np.pi)*sp.special.expi(k**2/(4*self.E**2)))
        else:
            return 0

    def t1(self, w, n):
        k = w*ev
        _sum = 0
        a1, a2 = self.lattice.getBravaisVectors()
        area = float(np.cross(a1, a2))
        for G_pos in self.reciprocal:
            beta = self.q + G_pos
            beta_norm = np.linalg.norm(beta)

            phi = np.angle(beta[0] + 1j*beta[1])
            _sum += (beta_norm/k)**n * 1/(k**2 - beta_norm**2) * np.exp((k**2 - beta_norm**2)/(4*self.E**2)) * np.exp(-1j*n*phi)
        return -(4*1j**(n+1)/area) * _sum

    def t2(self, w, n):
        k = w*ev
        _sum = 0
        for R_pos in self.lattice.genBravais(self.neighbours, False):
            R_norm = np.linalg.norm(R_pos)
            alpha = np.angle(R_pos[0] + 1j*R_pos[1])

            _sum += np.exp(1j*np.dot(self.q, R_pos)) * np.exp(-1j*n*alpha) * (R_norm/k)**n * self.t2IntegralFunc(R_norm, w)
        if n < 0:
            return -np.conjugate(-2**n*(1j/np.pi) * _sum)
        else:
            return -2**n*(1j/np.pi) * _sum

    def t2IntegralFunc(self, dist, w):
        k = w*ev
        _sum = 0
        for j in range(0, self.j_max+1):
            _sum = 1/(np.math.factorial(j)) * (k/(2*self.E))**(2*j) * sp.special.expn(j+1, dist**2*self.E**2)
        return _sum

    def gammaFunc(self, dist, w, n):
        k = w*ev
        _sum = 0
        for j in range(0, n):
            _sum = 1/(np.math.factorial(j)) * (k/2)**(2*j) * dist**(2*j-2*n) * sp.special.gammaincc(n-j, self.E)
        return _sum

    def t2Dyadic(self, w, n, _type):
        k = w*ev
        _sum = 0
        if _type is "xx":
            for R_pos in self.lattice.genBravais(self.neighbours, False):
                R_norm = np.linalg.norm(R_pos)

                _sum += (np.exp(1j * np.dot(self.q, R_pos))) * (self.t2DyadicIntegralFunc(w, R_pos, _type) + (np.exp(-R_norm**2*self.E**2)/R_norm**2)*(((4*R_pos[0]**2)/R_norm**2)*(R_norm**2*self.E**2 + 1) - 2))

        elif _type is "xy":
            for R_pos in self.lattice.genBravais(self.neighbours, False):
                R_norm = np.linalg.norm(R_pos)

                _sum += np.exp(1j * np.dot(self.q, R_pos)) * (self.t2DyadicIntegralFunc(w, R_pos, _type) + (np.exp(-R_norm**2*self.E**2) * (4*R_pos[0]*R_pos[1]/R_norm**2)*(R_norm**2*self.E**2 + 1)))

        elif _type is "yy":
            for R_pos in self.lattice.genBravais(self.neighbours, False):
                R_norm = np.linalg.norm(R_pos)

                _sum += (np.exp(1j * np.dot(self.q, R_pos))) * (self.t2DyadicIntegralFunc(w, R_pos, _type) + (np.exp(-R_norm**2*self.E**2)/R_norm**2)*(((4*R_pos[1]**2)/R_norm**2)*(R_norm**2*self.E**2 + 1) - 2))
        return -0.25**n*(1j/np.pi) * _sum

    def t2DyadicIntegralFunc(self, w , R_pos, _type):
        k = w*ev
        _sum = 0
        R_norm = np.linalg.norm(R_pos)
        if _type is "xx":
            for j in range(1, self.j_max+1):
                _sum += ((1./np.math.factorial(j)) * (k/(2*self.E))**(2*j)) * (4*R_pos[0]**2*self.E**4*sp.special.expn(j-1, R_norm**2*self.E**2) - 2*self.E**2*sp.special.expn(j, R_norm**2*self.E**2))

        elif _type is "xy":
            for j in range(1, self.j_max+1):
                _sum += ((1./np.math.factorial(j)) * (k/(2*self.E))**(2*j)) * (4*R_pos[0]*R_pos[1]*self.E**4*sp.special.expn(j-1, R_norm**2*self.E**2))

        elif _type is "yy":
            for j in range(1, self.j_max+1):
                _sum += ((1./np.math.factorial(j)) * (k/(2*self.E))**(2*j)) * (4*R_pos[1]**2*self.E**4*sp.special.expn(j-1, R_norm**2*self.E**2) - 2*self.E**2*sp.special.expn(j, R_norm**2*self.E**2))

        return _sum

    def reducedLatticeSum(self, w):
        k = w*ev
        
        _sum = 0
        for n in [0]:
            _sum += self.t0(w, n) + self.t1(w, n) + self.t2(w, n)
            #print("n={}, t0={}, t1={}, t2={}".format(n, self.t0(w, n), self.t1(w, n), self.t2(w, n)))
        return (0.25j)*_sum

    def dyadicReducedSumEwald(self, w):
        k = w*ev
        n=0
        xx_comp = k**2 * (self.t2Dyadic(w, n, "xx") + self.reducedLatticeSum(w))
        xy_comp = k**2 * self.t2Dyadic(w, n, "xy")
        yy_comp = k**2 * (self.t2Dyadic(w, n, "yy") + self.reducedLatticeSum(w))
        return [xx_comp, xy_comp, yy_comp]

    def testDyadicEwaldG2(self, w, _type):
        k = w*ev
        _sum = 0
        if _type is "xx":
            for R_pos in self.lattice.genBravais(self.neighbours, False):
                rho = R_pos
                rho_norm =  np.linalg.norm(rho)
                _sum += (np.exp(1j * np.dot(self.q, R_pos))) * (self.dyadicIntegralFunc(w, rho, _type) + (np.exp(-rho_norm**2*self.E**2)/rho_norm**2)*(((4*rho[0]**2)/rho_norm**2)*(rho_norm**2*self.E**2 + 1) - 2))

        elif _type is "xy":
            for R_pos in self.lattice.genBravais(self.neighbours, False):
                rho = R_pos
                rho_norm =  np.linalg.norm(rho)
                _sum += np.exp(1j * np.dot(self.q, R_pos)) * (self.dyadicIntegralFunc(w, rho, _type) + (np.exp(-rho_norm**2*self.E**2) * (4*rho[0]*rho[1]/rho_norm**2)*(rho_norm**2*self.E**2 + 1)))

        elif _type is "yy":
            for R_pos in self.lattice.genBravais(self.neighbours, False):
                rho = R_pos
                rho_norm =  np.linalg.norm(rho)
                _sum += (np.exp(1j * np.dot(self.q, R_pos))) * (self.dyadicIntegralFunc(w, rho, _type) + (np.exp(-rho_norm**2*self.E**2)/rho_norm**2)*(((4*rho[1]**2)/rho_norm**2)*(rho_norm**2*self.E**2 + 1) - 2))
        return _sum/(4*np.pi)

    def testEwaldG2(self, w):
        _sum = 0
        for R_pos in self.lattice.genBravais(self.neighbours, False):
            distance = np.linalg.norm(R_pos)
            _sum += (1./(4*np.pi)) * np.exp(1j * np.dot(self.q, R_pos)) * self.integralFunc(distance, w)
        return _sum

    def testDyadicSumEwald(self, w):
        k = w*ev
        xx_comp = (self.dyadicEwaldG1(w, "xx") + self.testDyadicEwaldG2(w, "xx") + k**2*self.testEwaldG2(w))
        xy_comp = (self.dyadicEwaldG1(w, "xy") + self.testDyadicEwaldG2(w, "xy"))
        yy_comp = (self.dyadicEwaldG1(w, "yy") + self.testDyadicEwaldG2(w, "yy") + k**2*self.testEwaldG2(w))
        return [xx_comp, xy_comp, yy_comp]

def testLatticeSum(vector_1, vector_2, neighbour_range, q, w, pos):
    """
    Plots graphs of monopolar lattice sums. Returns graph of non-converging original method and converging sums with Ewald's method.
    """
    results = []
    ewald_results = []
    loop_range = range(1, neighbour_range+1)
    lattice = Lattice(vector_1, vector_2)

    print("monopolar sum")

    if np.linalg.norm(np.array(pos)) == 0:  # if you are the origin, then exclude it from the sum
        print("Sum excluding origin")
        for i in loop_range:
            results.append(Interaction(q, lattice, i, pos, False).monopolarSum(w))
            print(i**2-1)

        for i in range(1, 20):
            ewald_results.append(Ewald(2*np.pi/(a), 5, q, lattice, i, pos).reducedLatticeSum(w))
        print("done")
    else:
        print("Sum including origin")
        for i in loop_range:
            results.append(Interaction(q, lattice, i, pos, True).monopolarSum(w))
            print(i**2-1)

        for i in range(1, 20):
            ewald_results.append(Ewald(2*np.pi/(a), 5, q, lattice, i, pos).monopolarSum(w))
        print("done")

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


def testDyadicSum(vector_1, vector_2, neighbour_range, q, w, pos=[0, 0]):
    """
    Plots graphs of dipolar lattice sums. Returns graph of non-converging original method and converging sums with Ewald's method.
    """
    results = []
    ewald_results = []
    loop_range = range(1, neighbour_range+1)
    lattice = Lattice(vector_1, vector_2)

    print("dyadic sum")

    if np.linalg.norm(np.array(pos)) == 0:  # if you are the origin, then exclude it from the sum
        print("Sum excluding origin")
        for i in loop_range:
            results.append(Interaction(q, lattice, i, pos, False).dyadicSum(wp))
            ewald_results.append(Ewald(2*np.pi/a, 20, q, lattice, i, pos).testDyadicSumEwald(wp))
            print(i**2-1)
        print("done")
    else:
        print("Sum including origin")
        for i in loop_range:
            results.append(Interaction(q, lattice, i, pos, True).dyadicSum(wp))
            ewald_results.append(Ewald(2*np.pi/a, 20, q, lattice, i, pos).dyadicSumEwald(wp))
            print(i**2-1)
        print("done")

    fig, ax = plt.subplots(3,4)
    ax[0][0].set_title("Non-Ewald, Re")
    ax[0][1].set_title("Non-Ewald, Im")
    ax[0][2].set_title("Ewald, Re")
    ax[0][3].set_title("Ewald, Im")

    ax[0][0].set_ylabel("$\partial^2/\partial x^2$ sum")
    ax[1][0].set_ylabel("$\partial^2/\partial x \partial y$ sum")
    ax[2][0].set_ylabel("$\partial^2/\partial y^2$ sum")
    for j in range(3):
        ax[j][0].plot([i**2-1 for i in loop_range], [i[j].real for i in results], 'r')
        ax[j][1].plot([i**2-1 for i in loop_range], [i[j].imag for i in results], 'r--')
        ax[j][2].plot([i**2-1 for i in loop_range], [i[j].real for i in ewald_results], 'g')
        ax[j][3].plot([i**2-1 for i in loop_range], [i[j].imag for i in ewald_results], 'g--')
    fig.text(0.5, 0.04, 'Number of terms in sum', ha='center')
    print("d_xx, d_xy, d_yy averages: {}".format(sum(results)/len(results)))
    print("convergent ewald result: {}".format(ewald_results[neighbour_range-2]))
    print("finished for q={}".format(q))

    plt.show()


if __name__ == '__main__':
    ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)
    a = 15*10**-9
    radius = 5*10**-9
    wp = 3.5
    loss = 0.01
    a1 = np.array([0, a])
    a2 = np.array([a, 0])
    pos = np.array([0, 0])
    q = np.array([83775804, 837758040])
    #83775804.0957278

    testDyadicSum(a1, a2, 20, q, wp, pos)
    #testLatticeSum(a1, a2, 50, q, wp, pos)
