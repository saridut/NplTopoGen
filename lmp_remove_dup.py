#!/usr/bin/env python

'''
Removes duplicate atom types from a lammps data file.

'''
import sys
import math
import numpy as np
from _configuration import Configuration
from _config_io import *
from _geom_utils import *

#-------------------------------------------------------------------------------

config = Configuration()

#dn ='ligands/carboxy_acid/GLC18_usat'
#name = 'GLC18_usat_anion'
dn ='solvents/methanol'
name = 'slvnt_methanol'

fn_in = dn + '/' + name + '_lpg.lmp'
fn_out = dn + '/' + name + '.lmp'
read_ldf(config, fn_in)

config.fit_simbox(sep=0.0)
box_center = config.simbox.mean(axis=1)
config.translate(-box_center) #Bring the box center to the origin
#config.translate(-config.simbox[:,0])

print("Total charge %.8g"%config.get_total_charge())

#config.remove_duplicate_atom_types()
#config.remove_duplicate_x_types('bond')
config.remove_duplicate_types()

write_ldf(config, fn_out, title=name)
#write_xyz(config, nam+'.xyz', title=nam)
