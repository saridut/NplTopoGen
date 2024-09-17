#!/usr/bin/env python

'''
Creates a single NPL crystal slab in a box. The box is commensurate with
the unit cell.

'''
import sys
import math
import numpy as np
from configuration import Configuration
from config_io import *
from geom_utils import *

#-------------------------------------------------------------------------------

config = Configuration()

#Add crystal size.
latis_a = 6.174 #Lattice parameter
nuc_xtal_x = 20 #Number of unit cells along x-direction
nuc_xtal_y = nuc_xtal_x #Number of unit cells along y-direction
numML = 3 #Number of Se monolayers
nam = 'ML%i_%ix%i'%(numML, nuc_xtal_x, nuc_xtal_x)
fn_meta = nam + '_meta.txt'


#Atom types
#All atoms are point particles
atm_t_Cd = config.add_atom_type(112.411, name='Cd')
atm_t_Se = config.add_atom_type(78.96, name='Se')

#Fractional coordinates
uc_fcoords = {
        'Cd_1': [np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.0])],
        'Cd_2': [np.array([0.5, 0.0, 0.5]), np.array([0.0, 0.5, 0.5])],
        'Se_1': [np.array([0.25, 0.25, 0.25]), np.array([0.75, 0.75, 0.25])],
        'Se_2': [np.array([0.75, 0.25, 0.75]), np.array([0.25, 0.75, 0.75])]
        }

#Unit cell
#uc_atm_t = [1, 1, 1, 1, 2, 2, 2, 2] #Atom types
#fcoords = np.array([ 
#   [0.00, 0.00, 0.00],
#   [0.00, 0.50, 0.50],
#   [0.50, 0.00, 0.50],
#   [0.50, 0.50, 0.00],
#   [0.25, 0.25, 0.25],
#   [0.25, 0.75, 0.75],
#   [0.75, 0.25, 0.75],
#   [0.75, 0.75, 0.25],
#   ])

nuc_xtal_z = math.ceil(numML+1/2) #Number of unit cells along z-direction

fh_meta = open(fn_meta, 'w')
fh_meta.write('latis_a = %g\n'%latis_a)
fh_meta.write('nuc_x = %d\n'%nuc_xtal_x)
fh_meta.write('nuc_y = %d\n'%nuc_xtal_y)
fh_meta.write('nuc_z = %d\n'%nuc_xtal_z)
fh_meta.write('num_layers = %d\n'%(2*numML+1))
mdata = []

config.add_simbox(0, 1, 0, 1, 0, 1)

#Add Cd atoms
chge = 1.18
for il in range(numML+1):
    iucz = il//2; iluc = il%2; key = 'Cd_%d'%(iluc+1)
    ilayr = il*2 + 1 #One-based
    for iucy in range(nuc_xtal_y):
        for iucx in range(nuc_xtal_x):
            origin = np.array([iucx, iucy, iucz])
            for each in uc_fcoords[key]:
                pos = latis_a*(origin + each)
                atm_id = config.add_atom(atm_t_Cd, chge, pos)
                mdata.append([atm_id, atm_t_Cd, ilayr, iucx+1, iucy+1, iucz+1])

#Add Se atoms
chge = -1.18
for il in range(numML):
    iucz = il//2; iluc = il%2; key = 'Se_%d'%(iluc+1)
    ilayr = il*2 + 2 #One-based
    for iucy in range(nuc_xtal_y):
        for iucx in range(nuc_xtal_x):
            origin = np.array([iucx, iucy, iucz])
            for each in uc_fcoords[key]:
                pos = latis_a*(origin + each)
                atm_id = config.add_atom(atm_t_Se, chge, pos)
                mdata.append([atm_id, atm_t_Se, ilayr, iucx+1, iucy+1, iucz+1])

fh_meta.write('\n')
mdata = np.asarray(mdata)

for i in range(1, 2*numML+2):
    indx = np.nonzero(mdata[:,2]==i)
    ib = indx[0][0]; ie = indx[0][-1]
    atm_id_ib = mdata[ib,0]; atm_id_ie = mdata[ie,0]
    fh_meta.write('L%d  %d  %d\n'%(i, atm_id_ib, atm_id_ie))

fh_meta.write('\n')
buf = '  '.join( ['%8s'%x for x in ['atm_id','atm_type','layer','ucx', 'ucy', 'ucz']] )
fh_meta.write(buf+'\n')
for i in range(mdata.shape[0]):
    buf = '  '.join( ['%8d'%x for x in mdata[i,:]] )
    fh_meta.write(buf+'\n')

fh_meta.close()

#Set pair coeffs
config.set_pair_coeff(atm_t_Cd, [0.0334, 1.98])
config.set_pair_coeff(atm_t_Se, [0.0296, 5.24])

config.fit_simbox(sep=0.0)
config.translate(-config.simbox[:,0])
config.simbox[0,1] = latis_a*nuc_xtal_x
config.simbox[1,1] = latis_a*nuc_xtal_y

print("Total charge %.2f"%config.get_total_charge())

write_ldf(config, nam+'.lmp', title=nam)
#write_xyz(config, nam+'.xyz', title=nam)
