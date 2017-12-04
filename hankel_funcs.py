#! /usr/bin/python3

import numpy as np
import scipy as sp
from scipy import special
from matplotlib import pyplot as plt

plot_range = np.linspace(0.1,1000, 100000)

fig, ax = plt.subplots(2)
ax[0].plot(plot_range, [sp.special.hankel1(0,r).real for r in plot_range], c='b', alpha = 0.4)
ax[1].plot(plot_range, [sp.special.hankel1(0,r).imag for r in plot_range], c='b', alpha = 0.4)
# ax[0].plot(plot_range, [sp.special.hankel1(1,r).real for r in plot_range], 'r')
# ax[1].plot(plot_range, [sp.special.hankel1(1,r).imag for r in plot_range], 'r')
ax[0].plot(plot_range, [sp.special.hankel1(2,r).real for r in plot_range], c='g', alpha = 0.4)
ax[1].plot(plot_range, [sp.special.hankel1(2,r).imag for r in plot_range], c='g', alpha = 0.4)

ax[0].plot(plot_range, [(sp.special.hankel1(0,r) + sp.special.hankel1(2,r)).real for r in plot_range], c='r')
ax[1].plot(plot_range, [(sp.special.hankel1(0,r) + sp.special.hankel1(2,r)).imag for r in plot_range], c='r')

ax[0].plot(plot_range, [(sp.special.hankel1(0,r) - sp.special.hankel1(2,r)).real for r in plot_range], c='purple')
ax[1].plot(plot_range, [(sp.special.hankel1(0,r) - sp.special.hankel1(2,r)).imag for r in plot_range], c='purple')
plt.legend()
plt.show()
