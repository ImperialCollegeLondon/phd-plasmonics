#! /usr/bin/python3

'''
Quick script to generate .csv files of real and imaginary permittivities for COMSOL.
'''

import numpy as np
from matplotlib import pyplot as plt
import csv

# define frequencies in eV
min_freq = 1
max_freq = 5 

plasma_freq = 3.5
gamma = 0.01

step = 0.05

ELEC_VOLT = 1.6*10**-19/(6.63*10**-34)  # eV conversion to Hz

fig, ax = plt.subplots(2)

perm_re = []
perm_im = []
for w in np.arange(min_freq, max_freq, step):
	perm_re.append([w*ELEC_VOLT, (1- plasma_freq**2/(w**2 - 1j*gamma*w)).real])
	perm_im.append([w*ELEC_VOLT, (1- plasma_freq**2/(w**2 - 1j*gamma*w)).imag])
# ax[0].scatter(np.arange(min_freq, max_freq, step), [i.real for i in permittivity],c='b')
# ax[1].scatter(np.arange(min_freq, max_freq, step), [i.imag for i in permittivity],c='r')

# ax[0].set_title("re")
# ax[1].set_title("im")
# plt.show()

with open("perms_real.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in perm_re:
        writer.writerow([val[0],val[1]])    


with open("perms_imag.csv", "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    for val in perm_im:
        writer.writerow([val[0],val[1]])