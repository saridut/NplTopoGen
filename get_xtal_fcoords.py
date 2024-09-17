#!/usr/bin/env python

import sys
import math
import numpy as np
from _configuration import Configuration
from _config_io import read_ldf

#Read in input LAMMPS data file containing the atom positions
#of the crystal unit cell
ldf_in = sys.argv[1]
config = Configuration()
read_ldf(config, ldf_in)

config.translate(-config.simbox[:,0])
xtal_a = config.simbox[0,1]
xtal_b = config.simbox[1,1]
xtal_c = config.simbox[2,1]
boxl = np.array([xtal_a, xtal_b, xtal_c])

for each in config.atoms.values():
    each['coords']/= boxl
print(xtal_a, xtal_b, xtal_c)
for each in config.atoms.values():
    print(each['coords'])
