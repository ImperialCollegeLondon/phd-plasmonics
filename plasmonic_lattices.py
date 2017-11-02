#! python3

import numpy as np
import scipy as sp
from scipy import special  # used for hankel functions
from scipy import optimize
from matplotlib import pyplot as plt
from multiprocessing import Pool
import itertools

# k-> w conversion
global evc
evc = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)


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
        self.pos = np.array([(x_pos, y_pos)])

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

    def __init__(self, spacing, radius, wp, loss):
        """
        Unit cell for square lattice contains single particle at (0, 0).
        """
        Particle.__init__(self, radius, wp, loss, 0, 0)
        self.unitcell = Particle(radius, wp, loss, 0, 0)
        self.spacing = spacing  # Lattice constant

    def getUnitCell(self):
        return self.unitcell.pos

    def getBravaisLattice(self, neighbours):
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

    def getReciprocalLattice(self, number):
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

    def getCellSize(self):
        return 1


class Interaction:
    """
    Calculating interactions.
    """

    def green(self, k, distance):
        """
        Green's function interaction.

        Calculate the Green's function tensor between particles which are separated
        by a vector distance at a frequency k. For a 2D Green's function, the
        interactions are modelled with Hankel function.s

        Returns a matrix of the form [[G_xx, G_xy],[G_xy, G_yy]]
        """
        x = distance[0]
        y = distance[1]
        R = np.linalg.norm(distance)
        arg = k*R

        xx_type = 0.25j * k**2 * (
            (y**2/R**2) * sp.special.hankel1(0, arg) +
            (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg))

        yy_type = 0.25j * k**2 * (
            (x**2/R**2) * sp.special.hankel1(0, arg) -
            (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg))

        xy_type = 0.25j * k**2 * x*y/R**2 * sp.special.hankel1(2, arg)

        return np.array([[xx_type, xy_type], [xy_type, yy_type]])

    def generateMatrix(self, w, q, unit_cell, neighbours):
        size = len(unit_cell)
        k = w*evc

        if size > 1:  # for unit cells with more than 1 particle
            # initialise matrix with (size*2) since we have x and y direction
            h_matrix = np.zeros((size*2, size*2), dtype=np.complex_)

            indices = np.arange(len(unit_cell))
            for (n, m) in itertools.combinations(indices, 2):
                # do something
                f
        elif size == 1:  # for unit cell with single particle
            h_matrix = sum([self.green(k, inter) *
                            np.exp(-1j * np.dot(q, inter))
                            for inter in neighbours])

        return h_matrix


square_lattice = Square(30*10**-9, 10*10**-9, 3.5, 0)
qspace = square_lattice.getReciprocalLattice(50)

g = Interaction()
mat = g.generateMatrix(2.4, qspace[1], square_lattice.getUnitCell(), square_lattice.getBravaisLattice(2))
print(mat)
