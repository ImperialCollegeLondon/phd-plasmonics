#!/usr/bin/python3

"""
Simulating Green's functions for electromagnetic interactions in an array of
plasmonic nanoparticles.
"""


import numpy as np
import scipy as sp
from scipy import special  # used for hankel functions
from scipy import optimize
from matplotlib import pyplot as plt
from multiprocessing import Pool


class Particle:
    """
    Particle class.

    Each particle has a position, radius, plasma frequency and loss.
    """
    def __init__(self, x_pos, y_pos, radius, wp, loss):
        self.R = np.array([x_pos, y_pos])
        self.radius = radius
        self.plasma = wp
        self.loss = loss


def reciprocal_space(spacing, resolution):
    """
    Create set of (x,y) coordinates for path in reciprocal space.

    From K to Gamma to M.
    """
    b = spacing * 3
    K_Gamma_x = np.linspace((4*np.pi)/(3*b), 0, resolution/2, endpoint=False)
    K_Gamma_y = np.zeros(int(resolution/2))

    Gamma_M_x = np.zeros(int(resolution/2))
    Gamma_M_y = np.linspace(0, (2*np.pi)/(np.sqrt(3)*b), resolution/2, endpoint=False)

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

    # xx type interaction
    xx_type = 0.25j * (
    k**2 * (sp.special.hankel1(0, arg))
    - (k/R) * sp.special.hankel1(1, arg)
    + ((k**2 * x**2)/(2*R**2)) * (sp.special.hankel1(2, arg) - sp.special.hankel1(0, arg))
    + ((k * x**2)/(R**3)) * sp.special.hankel1(1, arg)
    )

    # xy type interaction
    xy_type = 0.25j * sp.special.hankel1(2, arg) * ((k**2 * x*y)/(R**2))

    # yy type interaction
    yy_type = 0.25j * (
    k**2 * (sp.special.hankel1(0, arg))
    - (k/R) * sp.special.hankel1(1, arg)
    + ((k**2 * y**2)/(2*R**2)) * (sp.special.hankel1(2, arg) - sp.special.hankel1(0, arg))
    + ((k * y**2)/(R**3)) * sp.special.hankel1(1, arg))

    return np.array([[xx_type, xy_type], [xy_type, yy_type]])


def honeycomb(spacing, radius, wp, loss):
    """
    Honeycomb lattice.

    Create a honeycomb lattice with specific lattice spacing. Also define
    radii, plasma frequency and loss of particles in the lattice.
    """
    particle_coords = []

    particle_coords.append(Particle(spacing, 0, radius, wp, loss).R)
    particle_coords.append(Particle(spacing*0.5, -spacing*np.sqrt(3)/2, radius, wp, loss).R)
    particle_coords.append(Particle(-spacing*0.5, -spacing*np.sqrt(3)/2, radius, wp, loss).R)
    particle_coords.append(Particle(-spacing, 0, radius, wp, loss).R)
    particle_coords.append(Particle(-spacing*0.5, spacing*np.sqrt(3)/2, radius, wp, loss).R)
    particle_coords.append(Particle(spacing*0.5, spacing*np.sqrt(3)/2, radius, wp, loss).R)

    return np.array(particle_coords)


def interactions(intracell, intercell, w, q):
    """
    Interaction matrix.

    Find the matrix of interactions for a 6 site unit cell system. Interactions
    are both between particle in the same cell (INTRAcell) and between
    particles in different cells (INTERcell). Calculated at a particular
    frequency omega (w) and Bloch wavenumber (q).
    """
    H = np.zeros((12, 12), dtype=np.complex_)

    k = w*ev

    # We only need to fill 'top right' diagonal of the matrix since 'opposite
    # direction' interactions are given by the Hermitian conjugate.
    i = 1
    for n in np.arange(len(intracell)):
        for m in np.arange(start=i, stop=len(intracell)):
            H[2*n:2*n+2, 2*m:2*m+2] = sum([green(k, intracell[n] - intracell[m] + inter) * np.exp(-1j * np.dot(q, (intracell[n] - intracell[m] + inter))) for inter in intercell])
            H[2*m:2*m+2, 2*n:2*n+2] = sum([green(k, intracell[m] - intracell[n] + inter) * np.exp(-1j * np.dot(q, (intracell[m] - intracell[n] + inter))) for inter in intercell])
        i += 1

    #H = H + np.conjugate(H).T  # the matrix is symmetrical about the diagonal (check this)

    # Create the diagonal by considering interactions between same particle
    # sites but in different cells. Need to make sure to ignore the (0,0)
    # element in intercell to prevent issues with the Green's function as we
    # have no 'self interaction'.
    for n in np.arange(len(intracell)):
        to_sum = []
        for inter in intercell:
            if np.linalg.norm(inter) != 0:  # ignore (0,0) position
                to_sum.append(green(k, inter) * np.exp(-1j * np.dot(q, inter)))
        H[2*n:2*n+2, 2*n:2*n+2] = sum(to_sum)
    return H


def polar(w, wp, loss, radius):
    """Polarisability function.

    Dynamic polarisability for infinite cylinder.
    """
    k = w*ev
    eps = 1 - (wp**2)/(w**2 + 1j*loss*w)
    static = 2 * np.pi * radius**2 * ((eps-1)/(eps+1))
    return 1/(static/(1 - 1j * (k**2/8) * static))


def supercell(a, cell, t1, t2, max):
    """
    Create a repeated symmetrical list of points for the honeycomb lattice
    supercell structure.

    Returns a list of supercell positions (points) and particle positions
    (particles).
    """
    pos = cell
    points = []
    particles = []

    for n in np.arange(-max, max+1):
        if n < 0:
            for m in np.arange(-n-max, max+1):
                points.append(n*t1+m*t2)
        elif n > 0:
            for m in np.arange(-max, max-n+1):
                points.append(n*t1+m*t2)
        elif n == 0:
            for m in np.arange(-max, max+1):
                points.append(n*t1+m*t2)

    for p in points:
        for g in pos:
            particles.append(p + g)

    return [points, particles]


def plot_interactions(intracell, intercell):
    """
    Check if interactions between supercells are working properly. Also makes
    nice pictures.
    """
    i = 1
    to_plot = []
    for n in np.arange(len(intracell)):
        for m in np.arange(len(intracell)):
            for inter in intercell[0]:
                point = intracell[m] + inter
                to_plot.append([[intracell[n][0], point[0]], [intracell[n][1], point[1]]])
        i += 1

    for i in to_plot:  # plot interactions
        plt.plot(i[0], i[1], zorder=0, alpha=0.1, c='r')
    for i in intercell[1]:  # plot particle locations
        plt.scatter(i[0], i[1], c='k', zorder=1, alpha=0.1)
    plt.show()


def extinction(w, q, intracell, intercell):
    print(w)
    w_plasma = wp
    loss = g
    radius = r
    k = w*ev

    H_matrix = interactions(intracell, intercell, w, q)
    for i in range(12):
        H_matrix[i][i] = H_matrix[i][i] - polar(w, w_plasma, loss, radius)

    return 4*np.pi*k*(sum(1/sp.linalg.eigvals(H_matrix)).imag)


def extinction_wrap(args):
    return extinction(*args)


def calculate_extinction(wrange, qrange, intracell, intercell):
    results = []
    wq_vals = [(w, q, intracell, intercell) for w in wrange for q in qrange]
    pool = Pool(16)

    results.append(pool.map(extinction_wrap, wq_vals))
    return results


if __name__ == "__main__":
    a = 15.*10**-9  # lattice spacing
    r = 5.*10**-9  # particle radius
    wp = 3.5  # plasma frequency
    g = 0.02  # losses
    trans_1 = np.array([3*a, 0])  # 1st Bravais lattice vector
    trans_2 = np.array([3*a/2, np.sqrt(3)*3*a/2])  # 2nd Bravais lattice vector
    ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)  # k->w conversion

    intracell = honeycomb(a, r, wp, g)
    intercell = supercell(a, intracell, trans_1, trans_2, 1)[0]

    wmin = 1.75
    wmax = 3
    resolution = 150

    wrange = np.linspace(wmin, wmax, resolution)
    qrange = reciprocal_space(a, resolution)

    for i in [55,75,95]:
        test = []
        for w in wrange:
            test.append(np.log(np.abs(extinction(w, qrange[i], intracell, intercell))))
        plt.plot(wrange, test)
    plt.show()

    # raw_results = calculate_extinction(wrange, qrange, intracell, intercell)
    #
    # reshaped_results = np.array(raw_results).reshape((resolution, resolution))
    # plt.imshow(reshaped_results, origin='lower', extent=[0, resolution, wmin, wmax], aspect='auto')
    # plt.show()
