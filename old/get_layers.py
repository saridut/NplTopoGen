#!/usr/bin/env python

import math
import numpy as np
from configuration import Configuration
from config_io import *

fn_ldf = '../lmp_run/stress/NPL_ML3_10x10_0_caC2/ML3_10x10_0.lmp'
#Read LAMMPS data file
config = Configuration()
read_ldf(config, fn_ldf)

layer_sep = 0.25*6.174
numML = 3
num_layers = 2*numML + 1

#Z-coordinates of the atoms in the xtal
num_atoms = len(config.atoms)
xtal_atom_types = np.zeros((num_atoms,), dtype=np.int32)
xtal_zcoords = np.zeros((num_atoms,))
for key, val in config.atoms.items():
    xtal_atom_types[key-1] = val['type']
    xtal_zcoords[key-1] = val['coords'][2]

zmin = xtal_zcoords.min(); zmax = xtal_zcoords.max()
layer_locs = np.linspace(zmin, zmax, num_layers)

layers = {}
for i in range(num_layers):
    layers[i+1] = []
for i in range(num_atoms):
    k = np.argmin( np.fabs(layer_locs - xtal_zcoords[i]) )
    layers[k+1].append(i+1)

with open('layers.txt', 'w') as fh:
    fh.write('%d\n'%num_layers)
    for i in range(num_layers):
        fh.write( 'Layer %d %d\n'%(i+1, len(layers[i+1])) )
        buf = '  '.join(['%d'%x for x in layers[i+1]])
        fh.write(buf + '\n')
