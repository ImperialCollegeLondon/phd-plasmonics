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


class Square(Particle): # TODO: update names of getBravais...
    def __init__(self, spacing, radius, wp, loss, neighbours, scaling):
        self.spacing = spacing
        self.scaling = scaling
        self.wp = wp
        self.loss = loss
        self.neighbours = neighbours

        Particle.__init__(self, radius, wp, loss)

    def getLatticeVectors(self):
        a1 = np.array([0, self.scaling*self.spacing])
        a2 = np.array([self.scaling*self.spacing, 0])

        return a1, a2

    def getReciprocalVectors(self):
        a1, a2 = self.getLatticeVectors()
        R = np.array([[0, -1], [1, 0]])

        b1 = 2*np.pi * np.dot(R, a2)/np.dot(a1, np.dot(R, a2))
        b2 = 2*np.pi * np.dot(R, a1)/np.dot(a2, np.dot(R, a1))

        return b1, b2

    def getUnitCell(self):
        """
        Square has single particle in unit cell at origin
        """
        return [Particle(self.radius, self.wp, self.loss, 0, 0)]

    def getNeighbours(self):
        """
        Function to create square lattice structure, ignoring the origin.
        """
        t1, t2 = self.getLatticeVectors()
        lattice_points = []
        lattice_range = np.arange(-self.neighbours, self.neighbours+1)
        for (i, j) in itertools.product(lattice_range, repeat=2):
            if i != 0 or j != 0:  # ignore the origin
                lattice_points.append(i*t1 + j*t2)

        return np.array(lattice_points)

    def getLattice(self, _type, origin):
        """
        Function to create square lattice structure.
        """
        t1, t2 = 0, 0
        if _type is "bravais":
            t1, t2 = self.getLatticeVectors()
        elif _type is "reciprocal":
            t1, t2 = self.getReciprocalVectors()

        lattice_points = []
        lattice_range = np.arange(-self.neighbours, self.neighbours+1)
        for (i, j) in itertools.product(lattice_range, repeat=2):
            if origin is False:
                if i != 0 or j != 0:  # ignore the origin
                    lattice_points.append(i*np.array(t1) + j*np.array(t2))
            elif origin is True:
                lattice_points.append(i*t1 + j*t2)
        return np.array(lattice_points)

    def getBrillouinZone(self, size):
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

    def getSpacing(self):
        return self.spacing


class Triangle(Particle):  # TODO: update names of getBravais...
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

    def getLattice(self, _type='bravais', origin='false'):
        """
        Function to create square lattice structure, ignoring the origin.
        """

        lattice_points = []
        lattice_range = np.arange(-self.neighbours, self.neighbours+1)
        for (i, j) in itertools.product(lattice_range, repeat=2):
            if i != 0 or j != 0:  # ignore the origin
                lattice_points.append(i*self.t1 + j*self.t2)

        return np.array(lattice_points)

    def getBrillouinZone(self, size):
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


class SimpleHoneycomb(Particle):  # TODO: update names of getBravais...
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

    def getLattice(self, _type='bravais', origin='false'):
        neighbour_list = []
        number = self.neighbours
        t1 = np.array([self.scaling*1.5*self.spacing, self.scaling*self.spacing*np.sqrt(3)/2])
        t2 = np.array([self.scaling*1.5*self.spacing, -self.scaling*self.spacing*np.sqrt(3)/2])

        for n,m in itertools.product(np.arange(-number, number+1), repeat=2):
            neighbour_list.append(n*t1 + m*t2)

        return np.array(neighbour_list)

    def getBrillouinZone(self, size):
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

    def getLatticeVectors(self):
        b = 3* self.spacing * self.scaling

        a1 = np.array([b, 0])
        a2 = np.array([b/2, np.sqrt(3)*b/2])

        return a1, a2

    def getReciprocalVectors(self):
        a1, a2 = self.getLatticeVectors()
        R = np.array([[0, -1], [1, 0]])

        b1 = 2*np.pi * np.dot(R, self.a2)/np.dot(self.a1, np.dot(R, self.a2))
        b2 = 2*np.pi * np.dot(R, self.a1)/np.dot(self.a2, np.dot(R, self.a1))

        return b1, b2

    def getUnitCell(self):
        """Honeycomb has 6 particle unit cell"""
        particle_list = []
        for x, y in [(self.spacing, 0), (self.spacing*0.5, -self.spacing*np.sqrt(3)/2), (-self.spacing*0.5, -self.spacing*np.sqrt(3)/2), (-self.spacing, 0), (-self.spacing*0.5, self.spacing*np.sqrt(3)/2), (self.spacing*0.5, self.spacing*np.sqrt(3)/2)]:
            particle_list.append(Particle(self.radius, self.wp, self.loss, x, y))
        return np.array(particle_list)

    def getLattice(self, _type='bravais', origin='false'):
        """
        Create a repeated symmetrical list of points for the honeycomb lattice
        supercell structure. Returns a list of supercell positions (points).
        """
        points = []
        number = self.neighbours
        t1, t2 = 0, 0
        if _type is "bravais":
            t1, t2 = self.getLatticeVectors()
        elif _type is "reciprocal":
            t1, t2 = self.getReciprocalVectors()

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

    def getBrillouinZone(self, size):
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
        self.qrange = cell.getBrillouinZone(self.resolution)

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
        intercell = self.cell.getLattice('bravais', False)
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
                H[2*n:2*n+2, 2*m:2*m+2] = sum([self.green(w, -intracell[n].pos + intracell[m].pos + inter) * np.exp(1j * np.dot(self.q, inter)) for inter in intercell])
                H[2*m:2*m+2, 2*n:2*n+2] = sum([self.green(w, -intracell[m].pos + intracell[n].pos + inter) * np.exp(1j * np.dot(self.q, inter)) for inter in intercell])

            for n in indices:
                to_sum = []
                for inter in intercell:
                    if np.linalg.norm(inter) != 0:  # ignore (0,0) position
                        to_sum.append(self.green(w, inter) * np.exp(1j * np.dot(self.q, inter)))
                H[2*n:2*n+2, 2*n:2*n+2] = sum(to_sum)

        return H

    def eigenproblem(self, w):
        return self.interactionMatrix(w) - np.identity(self.cell.getCellSize()*2)/self.cell.getPolarisability(w)

    def determinant(self, w):
        print(w)
        w_val = w[0] + 1j*w[1]
        result = np.linalg.det(self.eigenproblem(w_val))
        return [result.real, result.imag]


class Ewald:
    def __init__(self, ewald, j_max, q, lattice, position):
        self.q = q
        self.lattice = lattice
        self.pos = position
        self.E = ewald
        self.j_max = j_max

    def ewaldG1(self, w):
        k = w*ev
        a1, a2 = self.lattice.getLatticeVectors()

        area = float(np.cross(a1, a2))
        _sum = 0
        for G_pos in self.lattice.getLattice('reciprocal', True):
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
        for R_pos in self.lattice.getLattice('bravais', True):
            distance = np.linalg.norm(self.pos - R_pos)
            _sum += -(1./(4*np.pi)) * np.exp(1j * np.dot(self.q, R_pos)) * self.integralFunc(distance, w)
        return _sum

    def origSum_incOrigin(self, w):
        return self.ewaldG1(w) + self.ewaldG2(w)

    def ewaldG2_deriv(self, w, type, lattice):
        k = w*ev
        _sum = 0
        for R_pos in lattice:
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
            _sum = _sum*(-1/(4*np.pi))*np.exp(1j*np.dot(self.q, R_pos))
        return _sum

    def ewaldG1_deriv(self, w, type):
        k = w*ev
        a1, a2 = self.lattice.getLatticeVectors()

        area = float(np.cross(a1, a2))
        _sum = 0
        for G_pos in self.lattice.getLattice('reciprocal', True):
            if type is 'xx':
                _sum += -(self.q+G_pos)[0]**2 * (1./area) * (np.exp(1j*np.dot(self.q+G_pos, self.pos)) * np.exp((k**2 - np.linalg.norm(q+G_pos)**2)/(4*self.E**2)))/(np.linalg.norm(self.q+G_pos)**2 - k**2)
            elif type is 'yy':
                _sum += -(self.q+G_pos)[1]**2 * (1./area) * (np.exp(1j*np.dot(self.q+G_pos, self.pos)) * np.exp((k**2 - np.linalg.norm(q+G_pos)**2)/(4*self.E**2)))/(np.linalg.norm(self.q+G_pos)**2 - k**2)
            if type is 'xy':
                _sum += -(self.q+G_pos)[0]*(self.q+G_pos)[1] * (1./area) * (np.exp(1j*np.dot(self.q+G_pos, self.pos)) * np.exp((k**2 - np.linalg.norm(q+G_pos)**2)/(4*self.E**2)))/(np.linalg.norm(self.q+G_pos)**2 - k**2)
        return -_sum

    def totalSum_incOrigin(self, w):
        k = w*ev
        pre_factor = k**2*self.orig(w)
        lattice = self.lattice.getLattice('bravais', True)
        xx_comp = pre_factor + self.ewaldG1_deriv(w, 'xx') + self.ewaldG2_deriv(w, 'xx', lattice)
        xy_comp = self.ewaldG1_deriv(w, 'xy') + self.ewaldG2_deriv(w, 'xy', lattice)
        yy_comp = pre_factor + self.ewaldG1_deriv(w, 'yy') + self.ewaldG2_deriv(w, 'yy', lattice)
        return np.array([[xx_comp, xy_comp], [xy_comp, yy_comp]])

    def t0(self, n, w):
        k = w*ev
        if n == 0:
            return -1 - (1j/np.pi) * sp.special.expi(k**2/(4*self.E**2))
        else:
            return 0

    def t1(self, n, w):
        k = w*ev
        a1, a2 = self.lattice.getLatticeVectors()
        area = float(np.cross(a1, a2))
        _sum = 0
        for G_pos in self.lattice.getLattice('reciprocal', True):
            beta = self.q + G_pos
            beta_n = np.linalg.norm(beta)
            angle = np.angle(beta[0] + 1j*beta[1])  # for some reason no arg func for real vectors, so have to convert to complex then find argument from there
            _sum += ((beta_n/k)**n)/(k**2 - beta_n**2) * np.exp((k**2 - beta_n**2)/(4*self.E**2)) * np.exp(-1j*n*angle)
        return ((4*1j**(n+1))/area) * _sum

    def t2(self, n, w):
        k = w*ev
        reduced_bravais = self.lattice.getLattice('bravais', False)
        _sum = 0
        for R_pos in reduced_bravais:
            alpha = np.angle(R_pos[0] + 1j*R_pos[1])  # for some reason no arg func for real vectors, so have to convert to complex then find argument from there
            _sum += np.exp(1j*np.dot(self.q, R_pos)) * np.exp(-1j*n*alpha) * (np.linalg.norm(R_pos)/k)**n * self.loop_j_func(w, R_pos, n)
        return -2**(n+1)*1j/np.pi * _sum

    def t2_deriv(self, w, _type):
        return -4j*self.ewaldG2_deriv(w, _type, self.lattice.getLattice('bravais', False))

    def loop_j_func(self, w, dist, n):
        k = w*ev
        _sum = 0
        for j in range(0, self.j_max+1):
            _sum += 1/(np.math.factorial(j)) * (k/2)**(2*j) * self.E**(2*n-2*j) * sp.special.expn(j+1-n, np.linalg.norm(dist)**2 * self.E**2)
        return 2*_sum

    def origSum_exOrigin(self, w):
        return -0.25j * (self.t0(0, w) + self.t1(0, w) + self.t2(0, w))

    def dyadicSum_exOrigin(self, w):
        k = w*ev
        pre_factor = k**2*self.origSum_exOrigin(w)
        xx_comp = pre_factor + self.t2_deriv(w, 'xx')
        xy_comp = self.t2_deriv(w, 'xy')
        yy_comp = pre_factor + self.t2_deriv(w, 'yy')
        return np.array([[xx_comp, xy_comp], [xy_comp, yy_comp]])

    def interactionMatrix(self, w):
        #if cell_size == 1:  # No interactions within the cell, only with other cells
        H = self.dyadicSum_exOrigin(w)
        print(H)
        return H

    def eigenproblem(self, w):
        return self.interactionMatrix(w) - np.identity(self.lattice.getCellSize()*2)/self.lattice.getPolarisability(w)

    def determinant(self, w):
        print(w)
        w_val = w[0] + 1j*w[1]
        result = np.linalg.det(self.eigenproblem(w_val))
        return [result.real, result.imag]


def determinant_solver(w, cell, resolution):
    roots = []
    for q in cell.getBrillouinZone(resolution):
        array_int = Interaction(q, cell)
        #array_int = Ewald(2*np.pi/cell.getSpacing(), 5, q, cell, np.array([0,0]))
        ans = sp.optimize.root(array_int.determinant, w).x
        roots.append(ans)
    return roots


def _determinant_solver(args):
    return determinant_solver(*args)


def dirtyRootFinder(wmin, wmax, guesses, cell, resolution):
    wrange = np.linspace(wmin, wmax, guesses)
    results = []
    values = [([w, -0.01], cell, resolution) for w in wrange]
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
    loss = 0.0  # losses

    wmin = plasma_freq/np.sqrt(2) - 1.3
    wmax = plasma_freq/np.sqrt(2) + 1.3

    resolution = 90

    lattice = SimpleHoneycomb(lattice_spacing, particle_radius, plasma_freq, loss, neighbours=2, scaling=1.0)
    # points = lattice.getLattice('bravais', True)
    # plt.scatter([i[0] for i in points],[i[1] for i in points])
    # plt.show()
    #Extinction(lattice, resolution, wmin, wmax).plotExtinction()
    dirtyRootFinder(wmin, wmax, 8, lattice, resolution)
