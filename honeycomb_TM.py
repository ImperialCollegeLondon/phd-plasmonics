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
    def __init__(self, x_pos, y_pos, radius, wp, loss):
        self.R = np.array([x_pos, y_pos])
        self.radius = radius
        self.plasma = wp
        self.loss = loss


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


def honeycomb_reciprocal_space(spacing, resolution):
    """
    Create set of (x,y) coordinates for path in reciprocal space.

    From K to Gamma to M.
    """
    b = spacing * 3
    K_Gamma_x = np.linspace((4*np.pi)/(3*b), 0, resolution/2, endpoint=False)
    K_Gamma_y = np.zeros(int(resolution/2))

    Gamma_M_x = np.zeros(int(resolution/2))
    Gamma_M_y = np.linspace(0, (2*np.pi)/(np.sqrt(3)*b), resolution/2, endpoint=True)

    q_x = np.concatenate((K_Gamma_x, Gamma_M_x))
    q_y = np.concatenate((K_Gamma_y, Gamma_M_y))

    return np.array(list(zip(q_x, q_y)))


def green(k, distance):
    """
    Green's function interaction.
    """
    x = distance[0]
    y = distance[1]
    R = np.linalg.norm(distance)
    arg = k*R

    result = 0.25j * k**2 * sp.special.hankel1(0, arg)

    return result


def honeycomb_supercell(t1, t2, max):
    """
    Create a repeated symmetrical list of points for the honeycomb lattice
    supercell structure. Returns a list of supercell positions (points).
    """
    points = []

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

    return points


def interactions(intracell, intercell, w, q):
    H = np.zeros((6, 6), dtype=np.complex_)
    k = w*ev

    indices = np.arange(len(intracell))
    for n, m in itertools.combinations(indices, 2):
        H[n][m] = sum([green(k, -intracell[n] + intracell[m] + inter) * np.exp(1j * np.dot(q, -intracell[n] + intracell[m] + inter)) for inter in intercell])
        H[m][n] = sum([green(k, -(-intracell[n] + intracell[m] + inter)) * np.exp(1j * np.dot(q, -(-intracell[n] + intracell[m] +  inter))) for inter in intercell])

    for n in np.arange(len(intracell)):
        to_sum = []
        for inter in intercell:
            if np.linalg.norm(inter) != 0:  # ignore (0,0) position
                to_sum.append(green(k, inter) * np.exp(1j * np.dot(q, inter)))
        H[n][n] = sum(to_sum)

    return H


def polar(w, wp, loss, radius):
    k = w*ev
    eps = 1 - (wp**2)/(w**2 + 1j*loss*w)
    static = np.pi*radius**2 * (eps - 1)
    result = static/(1 - 0.25j * (radius**2 * k**2) * ((eps-1)/(eps+1)))

    return result


def extinction(w, q, intracell, intercell):
    w_plasma = wp
    loss = g
    radius = r
    k = w*ev
    print(w)
    H_matrix = interactions(intracell, intercell, w, q)
    for i in range(len(H_matrix[0])):
        H_matrix[i][i] = H_matrix[i][i] - 1/polar(w, w_plasma, loss, radius)

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
    wp = 6.18  # plasma frequency
    g = 0.08  # losses
    scaling = 1.
    ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)  # k-> w conversion
    c = 2.997*10**8  # speed of light

    trans_1 = np.array([scaling*3*a, 0])  # 1st Bravais lattice vector
    trans_2 = np.array([scaling*3*a/2, scaling*np.sqrt(3)*3*a/2])  # 2nd Bravais lattice vector
    intracell = honeycomb(a, r, wp, g)
    intercell = honeycomb_supercell(trans_1, trans_2, 1)

    wmin = 2
    wmax = 3

    resolution = 100

    wrange = np.linspace(wmin, wmax, resolution, endpoint=True)
    qrange = honeycomb_reciprocal_space(a, resolution)

    light_line = [(np.linalg.norm(qval)/ev) for q, qval in enumerate(qrange)]
    plt.plot(light_line, zorder=1)
    raw_results = calculate_extinction(wrange, qrange, intracell, intercell)
    reshaped_results = np.array(raw_results).reshape((resolution, resolution))
    plt.imshow(reshaped_results, origin='lower', extent=[0, resolution-1, wmin, wmax], aspect='auto', cmap='viridis', zorder=0)
    plt.show()
