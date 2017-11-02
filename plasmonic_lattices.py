#! python3

import numpy as np
import scipy as sp
from scipy import special  # used for hankel functions
from scipy import optimize
from matplotlib import pyplot as plt
from multiprocessing import Pool
import itertools


class Particle:
    """
    Particle class.

    Each particle has a position, radius, plasma frequency and loss.
    """
    def __init__(self, radius, wp, loss, x_pos=0, y_pos=0):
        self.radius = radius
        self.plasma = wp
        self.loss = loss
        self.x = x_pos
        self.y = y_pos
        self.pos = np.array([x_pos, y_pos])

    def getPermittivity(self, w):
        """
        Return permittivity for a particle.

        args:
        - w: frequency (eV)
        """
        return 1 - (self.plasma**2)/(w**2 - 1j*self.loss*w)

    def getPolarisability(self, permittivity):
        """
        Return the polarisability for a particle.

        args:
        - permittivity: permittivity of the particle
        """
        return 2*np.pi*self.radius**2 * (permittivity - 1)/(permittivity + 1)


class Square(Particle):
    """
    Square lattice class.
    """
    def __init__(self, spacing, radius, wp, loss, x_pos=0, y_pos=0):
        """
        Unit cell for square lattice contains single particle at (0, 0).
        """
        Particle.__init__(self, radius, wp, loss, 0, 0)
        self.spacing = spacing  # Lattice constant

    def makeBravaisLattice(self, neighbours):
        """
        Function to create square lattice structure, ignoring the origin.

        args:
        - neighbours: number of neighbours to create in each primitive lattice
        vector direction.
        """
        primitive1 = np.array([0, self.spacing])
        primitive2 = np.array([self.spacing, 0])

        lattice_points = []
        lattice_range = np.arange(-neighbours, neighbours+1)
        for (i, j) in itertools.product(lattice_range, repeat=2):
            if i != 0 or j != 0:  # ignore the origin
                lattice_points.append(i*primitive1 + j*primitive2)

        return np.array(lattice_points)

    def makeReciprocalLattice(self, number):
        """
        Generate set of reciprocal lattice points from
        Gamma to X to M to Gamma.

        args:
        - number: number of points to create
        """

        Gamma_X_x = np.linspace(0, np.pi/self.spacing, number/3,
                                endpoint=False)
        Gamma_X_y = np.zeros(int(number/3))

        X_M_x = np.ones(int(number/3))*np.pi/self.spacing
        X_M_y = np.linspace(0, np.pi/self.spacing, number/3, endpoint=False)

        M_Gamma_x = np.linspace(np.pi/self.spacing, 0, number/3, endpoint=True)
        M_Gamma_y = np.linspace(np.pi/self.spacing, 0, number/3, endpoint=True)

        q_x = np.concatenate((Gamma_X_x, X_M_x, M_Gamma_x))
        q_y = np.concatenate((Gamma_X_y, X_M_y, M_Gamma_y))

        return np.array(list(zip(q_x, q_y)))


a = Square(30, 10, 3.5, 0, 0, 0)
points = a.makeBravaisLattice(5)
recip = a.makeReciprocalLattice(300)
