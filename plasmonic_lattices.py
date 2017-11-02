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


class Square(Particle):
    """
    Square lattice class.
    """
    def __init__(self, spacing, radius, wp, loss, x_pos=0, y_pos=0):
        """
        Unit cell for square lattice contains single particle at (0, 0)
        """
        Particle.__init__(self, radius, wp, loss, 0, 0)
        self.spacing = spacing  # Lattice constant

    def makeLattice(self, neighbours):
        """
        Function to create 
        """
        primitive1 = np.array([0, self.spacing])
        primitive2 = np.array([self.spacing, 0])

        lattice_points = []
        lattice_range = np.arange(-neighbours, neighbours+1)
        for (i,j) in itertools.product(lattice_range, repeat=2):
            lattice_points.append(i*primitive1 + j*primitive2)

        return np.array(lattice_points)


a = Square(30, 10, 3.5, 0, 0, 0)
points = a.makeLattice(5)

plt.scatter([i[0] for i in points], [i[1] for i in points])
plt.show()
