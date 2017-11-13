#! /usr/bin/python3

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
        self.pos = np.array([x_pos, y_pos])
        self.radius = radius
        self.plasma = wp
        self.loss = loss

    def getPermittivity(self, w):
        """
        Return permittivity for a particle.

        args:
        - w: frequency (eV)
        """
        return 1 - (self.plasma**2)/(w**2 - 1j*self.loss*w)

    def getPolarisability(self, w):
        """
        Return the polarisability for a particle.

        """
        k = w * ev
        permittivity = self.getPermittivity(w)
        eps = (permittivity-1)/(permittivity+1)
        return 2*np.pi*self.radius**2 * eps * 1/(1 - 0.25j*np.pi*(k*self.radius)**2 * eps)


class Honeycomb(Particle):
    def __init__(self, spacing, radius, wp, loss):
        self.spacing = spacing
        self.wp = wp
        self.loss = loss
        Particle.__init__(self, radius, wp, loss)

    def getExtendedCell(self):
        particle_list = []
        for x, y in [(self.spacing, 0), (self.spacing*0.5, -self.spacing*np.sqrt(3)/2), (-self.spacing*0.5, -self.spacing*np.sqrt(3)/2), (-self.spacing, 0), (-self.spacing*0.5, self.spacing*np.sqrt(3)/2), (self.spacing*0.5, self.spacing*np.sqrt(3)/2)]:
            particle_list.append(Particle(self.radius, self.wp, self.loss, x, y))
        return particle_list

    def getSupercells(self, number, scaling):
        """
        Create a repeated symmetrical list of points for the honeycomb lattice
        supercell structure. Returns a list of supercell positions (points).
        """
        points = []

        b = 3 * self.spacing

        t1 = np.array([scaling*b, 0])  # 1st Bravais lattice vector
        t2 = np.array([scaling*b/2, scaling*np.sqrt(3)*b/2])  # 2nd Bravais lattice vector

        for n in np.arange(-number, number+1):
            if n < 0:
                for m in np.arange(-n-number, number+1):
                    points.append(n*t1+m*t2)
            elif n > 0:
                for m in np.arange(-number, number-n+1):
                    points.append(n*t1+m*t2)
            elif n == 0:
                for m in np.arange(-number, number+1):
                    points.append(n*t1+m*t2)

        return points

    def getReciprocalLattice(self, size):
        """
        Create set of (x,y) coordinates for path in reciprocal space.

        From K to Gamma to M.
        """
        b = self.spacing * 3
        K_Gamma_x = np.linspace((4*np.pi)/(3*b), 0, size/2, endpoint=False)
        K_Gamma_y = np.zeros(int(size/2))

        Gamma_M_x = np.zeros(int(size/2))
        Gamma_M_y = np.linspace(0, (2*np.pi)/(np.sqrt(3)*b), size/2, endpoint=True)

        q_x = np.concatenate((K_Gamma_x, Gamma_M_x))
        q_y = np.concatenate((K_Gamma_y, Gamma_M_y))

        return np.array(list(zip(q_x, q_y)))


def green(k, distance):
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
    (y**2/R**2) * sp.special.hankel1(0,arg) +
    (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg)
    )

    yy_type = 0.25j * k**2 * (
    (x**2/R**2) * sp.special.hankel1(0,arg) -
    (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg)
    )

    xy_type = 0.25j * k**2 * x*y/R**2 * sp.special.hankel1(2,arg)

    return np.array([[xx_type, xy_type], [xy_type, yy_type]])


def calculateInteraction(cell, w, q):
    k = w*ev

    intracell = cell.getExtendedCell()
    intercell = cell.getSupercells(1, 1)
    indices = np.arange(len(intracell))

    matrix_size = len(intracell)*2

    H = np.zeros((matrix_size, matrix_size), dtype=np.complex_)

    for n, m in itertools.combinations(indices, 2):
        H[2*n:2*n+2, 2*m:2*m+2] = sum([green(k, -intracell[n].pos + intracell[m].pos + inter) * np.exp(1j * np.dot(q, -intracell[n].pos + intracell[m].pos + inter)) for inter in intercell])
        H[2*m:2*m+2, 2*n:2*n+2] = sum([green(k, -(-intracell[n].pos + intracell[m].pos + inter)) * np.exp(1j * np.dot(q, -(-intracell[n].pos + intracell[m].pos +  inter))) for inter in intercell])

    for n in indices:
        to_sum = []
        for inter in intercell:
            if np.linalg.norm(inter) != 0:  # ignore (0,0) position
                to_sum.append(green(k, inter) * np.exp(1j * np.dot(q, inter)))
        H[2*n:2*n+2, 2*n:2*n+2] = sum(to_sum)
    return H

def extinction(w, q, cell):
    print(w)
    k = w*ev
    H_matrix = calculateInteraction(cell, w, q)
    for i in range(len(H_matrix[0])):
        H_matrix[i][i] = H_matrix[i][i] - 1/cell.getPolarisability(w)

    return 4*np.pi*k*(sum(1/sp.linalg.eigvals(H_matrix)).imag)


def extinction_wrap(args):
    return extinction(*args)


def calculate_extinction(wrange, qrange, cell):
    results = []
    wq_vals = [(w, q, cell) for w in wrange for q in qrange]
    pool = Pool(16)

    results.append(pool.map(extinction_wrap, wq_vals))
    return results


def plot_extinction(wrange, qrange, cell, resolution):
    light_line = [(np.linalg.norm(qval)/ev) for q, qval in enumerate(qrange)]
    plt.plot(light_line, 'r--', zorder=1, alpha=0.5)

    raw_results = calculate_extinction(wrange, qrange, cell)
    reshaped_results = np.array(raw_results).reshape((resolution, resolution))
    plt.imshow(reshaped_results, origin='lower', extent=[0, resolution-1, wmin, wmax], aspect='auto', cmap='viridis', zorder=0)

    plt.show()


if __name__ == "__main__":
    a = 15.*10**-9  # lattice spacing
    r = 5.*10**-9  # particle radius
    wp = 6.18  # plasma frequency
    g = 0.0  # losses
    scaling = 1.0
    ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)  # k-> w conversion
    c = 2.997*10**8  # speed of light

    lattice = Honeycomb(a, r, wp, g)
    # intracell = lattice.getExtendedCell()
    # intercell = lattice.getSupercells(1, 1)

    wmin = wp/np.sqrt(2) - 0.5
    wmax = wp/np.sqrt(2) + 0.5

    resolution = 50

    wrange = np.linspace(wmin, wmax, resolution, endpoint=True)
    qrange = lattice.getReciprocalLattice(resolution)

    plot_extinction(wrange, qrange, lattice, resolution)
