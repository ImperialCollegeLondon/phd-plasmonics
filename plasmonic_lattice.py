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
        return (2*np.pi*self.radius**2 * eps)/(1 - 0.25j*np.pi*(k*self.radius)**2 * eps)


class Square(Particle):
    """
    Square lattice class.
    """

    def __init__(self, spacing, radius, wp, loss, neighbours, scaling):
        self.spacing = spacing
        self.scaling = scaling
        self.wp = wp
        self.loss = loss
        self.neighbours = neighbours
        self.t1 = np.array([0, self.scaling*self.spacing])
        self.t2 = np.array([self.scaling*self.spacing, 0])
        Particle.__init__(self, radius, wp, loss)

    def getLatticeVectors(self):
        return [self.t1, self.t2]

    def getUnitCell(self):
        """
        Square has single particle in unit cell at origin
        """
        return [Particle(self.radius, self.wp, self.loss, 0, 0)]

    def getNeighbours(self):
        """
        Function to create square lattice structure, ignoring the origin.
        """

        lattice_points = []
        lattice_range = np.arange(-self.neighbours, self.neighbours+1)
        for (i, j) in itertools.product(lattice_range, repeat=2):
            if i != 0 or j != 0:  # ignore the origin
                lattice_points.append(i*self.t1 + j*self.t2)

        return np.array(lattice_points)

    def getReciprocalLattice(self, size):
        """
        Generate set of reciprocal lattice points from
        Gamma to X to M to Gamma.

        args:
        - size: number of points to create
        """

        Gamma_X_x = np.linspace(0, np.pi/self.spacing, size/3,
                                endpoint=False)
        Gamma_X_y = np.zeros(int(size/3))

        X_M_x = np.ones(int(size/3))*np.pi/self.spacing
        X_M_y = np.linspace(0, np.pi/self.spacing, size/3, endpoint=False)

        M_Gamma_x = np.linspace(np.pi/self.spacing, 0, size/3, endpoint=True)
        M_Gamma_y = np.linspace(np.pi/self.spacing, 0, size/3, endpoint=True)

        q_x = np.concatenate((Gamma_X_x, X_M_x, M_Gamma_x))
        q_y = np.concatenate((Gamma_X_y, X_M_y, M_Gamma_y))

        return np.array(list(zip(q_x, q_y)))

    def getCellSize(self):
        return 1


class Triangle(Particle):
    """
    Square lattice class.
    """

    def __init__(self, spacing, radius, wp, loss, neighbours, scaling):
        self.spacing = spacing
        self.scaling = scaling
        self.wp = wp
        self.loss = loss
        self.neighbours = neighbours
        self.t2 = np.array([self.spacing, 0])
        self.t1 = np.array([self.spacing/2, self.spacing*np.sqrt(3)/2])
        Particle.__init__(self, radius, wp, loss)

    def getLatticeVectors(self):
        return [self.t1, self.t2]

    def getUnitCell(self):
        """
        Square has single particle in unit cell at origin
        """
        return [Particle(self.radius, self.wp, self.loss, 0, 0)]

    def getNeighbours(self):
        """
        Function to create square lattice structure, ignoring the origin.
        """

        lattice_points = []
        lattice_range = np.arange(-self.neighbours, self.neighbours+1)
        for (i, j) in itertools.product(lattice_range, repeat=2):
            if i != 0 or j != 0:  # ignore the origin
                lattice_points.append(i*self.t1 + j*self.t2)

        return np.array(lattice_points)

    def getReciprocalLattice(self, size):
        """
        Create set of (x,y) coordinates for path in reciprocal space.

        From Gamma to M to K to Gamma.
        """

        Gamma_M_x = np.zeros(int(size/3))
        Gamma_M_y = np.linspace(0, (2*np.pi)/(np.sqrt(3)*self.spacing), int(size/3), endpoint=False)

        M_K_x = np.linspace(0, (2*np.pi)/(3*self.spacing), int(size/3), endpoint=False)
        M_K_y = np.ones(int(size/3))*(2*np.pi)/(np.sqrt(3)*self.spacing)

        K_Gamma_x = np.linspace((2*np.pi)/(3*self.spacing), 0, int(size/3), endpoint=True)
        K_Gamma_y = np.linspace((2*np.pi)/(np.sqrt(3)*self.spacing), 0, int(size/3), endpoint=True)

        q_x = np.concatenate((Gamma_M_x, M_K_x, K_Gamma_x))
        q_y = np.concatenate((Gamma_M_y, M_K_y, K_Gamma_y))

        return np.array(list(zip(q_x, q_y)))
        #return np.array(list(zip(np.linspace(-(4*np.pi)/(3*np.sqrt(3)*self.spacing), (4*np.pi)/(3*np.sqrt(3)*self.spacing), size, endpoint=True), np.zeros(size))))


    def getCellSize(self):
        return 1


class SimpleHoneycomb(Particle):
    def __init__(self, spacing, radius, wp, loss, neighbours, scaling):
        self.spacing = spacing
        self.scaling = scaling
        self.wp = wp
        self.loss = loss
        self.neighbours = neighbours
        Particle.__init__(self, radius, wp, loss)

    def getUnitCell(self):
        """Honeycomb has two particle unit cell"""
        particle_list = []
        for x, y in [(0, 0), (self.spacing, 0)]:
            particle_list.append(Particle(self.radius, self.wp, self.loss, x, y))
        return particle_list

    def getNeighbours(self):
        neighbour_list = []
        number = self.neighbours
        t1 = np.array([self.scaling*1.5*self.spacing, self.scaling*self.spacing*np.sqrt(3)/2])
        t2 = np.array([self.scaling*1.5*self.spacing, -self.scaling*self.spacing*np.sqrt(3)/2])

        for n,m in itertools.product(np.arange(-number, number+1), repeat=2):
            neighbour_list.append(n*t1 + m*t2)

        return np.array(neighbour_list)

    def getReciprocalLattice(self, size):
        """
        Create set of (x,y) coordinates for path in reciprocal space.

        From Gamma to M to K to Gamma.
        """

        Gamma_M_x = np.linspace(0, (2*np.pi)/(3*self.spacing), int(size/3), endpoint=False)
        Gamma_M_y = np.zeros(int(size/3))

        M_K_x = np.linspace((2*np.pi)/(3*self.spacing), 0, int(size/3), endpoint=False)
        M_K_y = np.linspace(0, (4*np.pi)/(3*np.sqrt(3)*self.spacing), int(size/3), endpoint=False)

        K_Gamma_x = np.zeros(int(size/3))
        K_Gamma_y = np.linspace((4*np.pi)/(3*np.sqrt(3)*self.spacing), 0, int(size/3), endpoint=True)

        q_x = np.concatenate((Gamma_M_x, M_K_x, K_Gamma_x))
        q_y = np.concatenate((Gamma_M_y, M_K_y, K_Gamma_y))

        return np.array(list(zip(q_x, q_y)))
        #return np.array(list(zip(np.zeros(size), np.linspace((3.5*np.pi)/(3*np.sqrt(3)*self.spacing), (4.5*np.pi)/(3*np.sqrt(3)*self.spacing), size, endpoint=True))))

    def getCellSize(self):
        return 2


class Honeycomb(Particle):
    def __init__(self, spacing, radius, wp, loss, neighbours, scaling):
        self.spacing = spacing
        self.scaling = scaling
        self.neighbours = neighbours
        self.wp = wp
        self.loss = loss
        Particle.__init__(self, radius, wp, loss)

    def getUnitCell(self):
        """Honeycomb has 6 particle unit cell"""
        particle_list = []
        for x, y in [(self.spacing, 0), (self.spacing*0.5, -self.spacing*np.sqrt(3)/2), (-self.spacing*0.5, -self.spacing*np.sqrt(3)/2), (-self.spacing, 0), (-self.spacing*0.5, self.spacing*np.sqrt(3)/2), (self.spacing*0.5, self.spacing*np.sqrt(3)/2)]:
            particle_list.append(Particle(self.radius, self.wp, self.loss, x, y))
        return np.array(particle_list)

    def getNeighbours(self):
        """
        Create a repeated symmetrical list of points for the honeycomb lattice
        supercell structure. Returns a list of supercell positions (points).
        """
        points = []
        number = self.neighbours

        b = 3 * self.spacing * self.scaling

        t1 = np.array([b, 0])  # 1st Bravais lattice vector
        t2 = np.array([b/2, np.sqrt(3)*b/2])  # 2nd Bravais lattice vector

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

        return np.array(points)

    def getReciprocalLattice(self, size):
        """
        Create set of (x,y) coordinates for path in reciprocal space.

        From K to Gamma to M.
        """
        b = 3* self.spacing * self.scaling
        K_Gamma_x = np.linspace((4*np.pi)/(3*b), 0, size/2, endpoint=False)
        K_Gamma_y = np.zeros(int(size/2))

        Gamma_M_x = np.zeros(int(size/2))
        Gamma_M_y = np.linspace(0, (2*np.pi)/(np.sqrt(3)*b), size/2, endpoint=True)

        q_x = np.concatenate((K_Gamma_x, Gamma_M_x))
        q_y = np.concatenate((K_Gamma_y, Gamma_M_y))

        return np.array(list(zip(q_x, q_y)))

    def getCellSize(self):
        return 6


class Extinction:
    def __init__(self, cell, resolution, wmin, wmax):
        self.cell = cell
        self.wmin = wmin
        self.wmax = wmax
        self.resolution = resolution
        self.wrange = np.linspace(wmin, wmax, self.resolution, endpoint=True)
        self.qrange = cell.getReciprocalLattice(self.resolution)

    def calcExtinction(self, w, q):
        """
        Find the extinction at a particular (w, q).
        """
        print(w)
        k = w*ev
        H_matrix = Interaction(q, self.cell).interactionMatrix(w)
        for i in range(len(H_matrix[0])):
            H_matrix[i][i] = H_matrix[i][i] - 1/self.cell.getPolarisability(w)

        return 4*np.pi*k*(sum(1/sp.linalg.eigvals(H_matrix)).imag)

    def _calcExtinction(self, args):
        """
        Wrapper for multiprocessing.
        """
        return self.calcExtinction(*args)

    def loopExtinction(self):
        """
        Method for quickly looping over (w, q) using multiprocessing.

        Calculates the extinction at each (w, q) using calcExtinction() then returns a linear list of extinction values.
        """
        results = []
        wq_vals = [(w, q) for w in self.wrange for q in self.qrange]
        pool = Pool()

        results.append(pool.map(self._calcExtinction, wq_vals))
        return results

    def plotExtinction(self):
        """
        Method for plotting extinction.

        Takes linear list of extinction values from loopExtinction(), reshapes into (size * size) array and plots using imshow().
        """
        light_line = [(np.linalg.norm(qval)/ev) for q, qval in enumerate(self.qrange)]
        plt.plot(light_line, 'r--', zorder=1, alpha=0.5)

        raw_results = self.loopExtinction()
        reshaped_results = np.array(raw_results).reshape((self.resolution, self.resolution))
        plt.imshow(reshaped_results, origin='lower', extent=[0, self.resolution-1, self.wmin, self.wmax], aspect='auto', cmap='viridis', zorder=0)

        plt.show()


class Interaction:
    def __init__(self, q, cell):
        self.q = q
        self.cell = cell

    def green(self, w, distance):
        """
        Green's function interaction.

        Calculate the Green's function tensor between particles which are separated
        by a vector distance at a frequency k. For a 2D Green's function, the
        interactions are modelled with Hankel functions.

        Returns a matrix of the form [[G_xx, G_xy],[G_xy, G_yy]]
        """
        k = w*ev
        x = distance[0]
        y = distance[1]
        R = np.linalg.norm(distance)
        arg = k*R

        xx_type = 0.25j * k**2 * ((y**2/R**2) * sp.special.hankel1(0,arg) + (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg))

        yy_type = 0.25j * k**2 * ((x**2/R**2) * sp.special.hankel1(0,arg) - (x**2 - y**2)/(k*R**3) * sp.special.hankel1(1, arg))

        xy_type = 0.25j * k**2 * x*y/R**2 * sp.special.hankel1(2,arg)

        return np.array([[xx_type, xy_type], [xy_type, yy_type]])

    def interactionMatrix(self, w):
        intracell = self.cell.getUnitCell()
        intercell = self.cell.getNeighbours()
        cell_size = self.cell.getCellSize()
        indices = np.arange(cell_size)

        matrix_size = cell_size*2

        if cell_size == 1:  # No interactions within the cell, only with other cells
            H = sum([self.green(w, inter) * np.exp(1j * np.dot(self.q, inter)) for inter in intercell])

        else:  # Interactions within and with other cells
            H = np.zeros((matrix_size, matrix_size), dtype=np.complex_)

            for n, m in itertools.combinations(indices, 2):
                # Loop over (n, m) = (0, 1), (0, 2)... (1, 2), (1, 3)... (2, 3), (2, 4)...
                # More efficient than considering repeated interactions.
                H[2*n:2*n+2, 2*m:2*m+2] = sum([self.green(w, -intracell[n].pos + intracell[m].pos + inter) * np.exp(1j * np.dot(-self.q, -intracell[n].pos + intracell[m].pos + inter)) for inter in intercell])
                H[2*m:2*m+2, 2*n:2*n+2] = sum([self.green(w, -intracell[m].pos + intracell[n].pos + inter) * np.exp(1j * np.dot(-self.q, -intracell[m].pos + intracell[n].pos + inter)) for inter in intercell])

            for n in indices:
                to_sum = []
                for inter in intercell:
                    if np.linalg.norm(inter) != 0:  # ignore (0,0) position
                        to_sum.append(self.green(w, inter) * np.exp(-1j * np.dot(self.q, inter)))
                H[2*n:2*n+2, 2*n:2*n+2] = sum(to_sum)

        return H

    def eigenproblem(self, w):
        return self.interactionMatrix(w) - np.identity(self.cell.getCellSize()*2)/self.cell.getPolarisability(w)

    def determinant(self, w):
        w_val = w[0] + 1j*w[1]
        result = np.linalg.det(self.eigenproblem(w_val))
        return [result.real, result.imag]


def determinant_solver(w, cell, resolution):
    print(w)
    roots = []
    for q in cell.getReciprocalLattice(resolution):
        array_int = Interaction(q, cell)
        ans = sp.optimize.root(array_int.determinant, w).x
        roots.append(ans)
    return roots


def _determinant_solver(args):
    return determinant_solver(*args)


def dirtyRootFinder(wmin, wmax, guesses, cell, resolution):
    wrange = np.linspace(wmin, wmax, guesses)
    results = []
    values = [([w, 0.01], cell, resolution) for w in wrange]
    pool = Pool()
    results.append(pool.map(_determinant_solver, values))
    pool.close()
    fig, ax = plt.subplots(2)
    for i in range(guesses):
        ax[0].scatter(np.arange(resolution), [results[0][i][j][0] for j in range(resolution)], c='r', s=1)
        ax[1].scatter(np.arange(resolution), [results[0][i][j][1] for j in range(resolution)], c='b', s=1)
    plt.show()


if __name__ == "__main__":
    global ev
    ev = (1.602*10**-19 * 2 * np.pi)/(6.626*10**-34 * 2.997*10**8)  # k-> w conversion
    global c
    c = 2.997*10**8  # speed of light

    lattice_spacing = 15.*10**-9  # lattice spacing
    particle_radius = 5.*10**-9  # particle radius
    plasma_freq = 6.18  # plasma frequency
    loss = 0.05  # losses

    wmin = plasma_freq/np.sqrt(2) - 1.3
    wmax = plasma_freq/np.sqrt(2) + 1.3
    resolution = 300

    lattice = Square(lattice_spacing, particle_radius, plasma_freq, loss, neighbours=1, scaling=1)

    #Extinction(lattice, resolution, wmin, wmax).plotExtinction()
    dirtyRootFinder(wmin, wmax, 4, lattice, resolution)
