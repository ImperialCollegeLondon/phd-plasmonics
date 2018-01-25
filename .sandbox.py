#! /usr/bin/python3

import numpy as np
import scipy as sp

from matplotlib import pyplot as plt

fig, ax = plt.subplots(1)

a = 15*10**-9


plt.plot([-np.pi/a, -np.pi/a, np.pi/a, np.pi/a, -np.pi/a],[-np.pi/a, +np.pi/a, np.pi/a, -np.pi/a, -np.pi/a], c="k")
plt.plot([0, np.pi/a, np.pi/a, 0],[0, np.pi/a, 0, 0], c='r')
plt.show()