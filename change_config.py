#!/usr/bin/env python

'''
Performs general changes with LAMMPS configuration files, e.g., box size,
bounds, etc.

'''

import os
import math
import numpy as np
from configuration import Configuration
from config_io import *

fn = os.path.expanduser('~/Desktop/rst_data.lmp') #Input configuration
fn_ref = os.path.expanduser('NPL_ML3_201x10_15.lmp')
fn_out = 'NPL_ML3_201x10_15_new.lmp' #Output configuration
fn_vel = 'vel.txt'
fn_pos = 'pos.xyz'
xhi = 2222.0
yhi = 2222.0
zhi = 2222.0

#Extract input velocities, if available in the input file
extract_velocities(fn, fn_vel)

#Load current position and unwrap PBC
config = Configuration()
config.add_simbox(0, 10, 0, 10, 0, 10)

read_ldf(config, fn)
#config.unwrap_pbc()

#Move atoms to origin
r = config.get_barycenter()
config.translate(-r, only_atoms=True)

#Save positions
write_xyz(config, fn_pos)

#Read in reference configuration
config.clear()
read_ldf(config, fn_ref)
#Tweak for CVFF impropers
for val in config.improper_coeffs.values():
    val[1] = int(val[1]); val[2] = int(val[2])

#Update positions
read_xyz(config, fn_pos)

##Update velocites
with open(fn_vel, 'r') as fh_vel:
    n = int( fh_vel.readline().strip(' \n') )
    for i in range(n):
        words = fh_vel.readline().strip(' \n').split()
        atm_id = int(words[0])
        v = np.array( [float(words[0]), float(words[1]), float(words[2])] ) 
        config.add_velocity(atm_id, v)

#Update simulation box (Add overrides existing box)
config.add_simbox(0.0, xhi, 0.0, yhi, 0.0, zhi)

#Move atoms to the center of the box and apply PBC
box_center = np.array([xhi/2, yhi/2, zhi/2])
config.translate(box_center, only_atoms=True)
config.apply_pbc()

#Write out
write_ldf(config, fn_out, title='')
