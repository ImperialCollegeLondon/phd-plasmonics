#! /usr/bin/python3

import numpy as np
import scipy as sp
from scipy import special
from matplotlib import pyplot as plt

plot_range = np.linspace(0.001,1000, 100000)

fig, ax = plt.subplots(2)
ax[0].plot(plot_range, [sp.special.hankel1(0,r).real for r in plot_range], 'b')
ax[1].plot(plot_range, [sp.special.hankel1(0,r).imag for r in plot_range], 'b')
ax[0].plot(plot_range, [sp.special.hankel1(1,r).real for r in plot_range], 'r')
ax[1].plot(plot_range, [sp.special.hankel1(1,r).imag for r in plot_range], 'r')
ax[0].plot(plot_range, [sp.special.hankel1(2,r).imag for r in plot_range], 'g')
ax[1].plot(plot_range, [sp.special.hankel1(2,r).imag for r in plot_range], 'g')
ax[0].legend()
plt.show()
