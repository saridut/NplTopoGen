#!/usr/bin/env python

'''
Creates a n-hexane molecule with TraPPE-UAf force field.

'''
import sys
import os
import math
import numpy as np
from configuration import Configuration
from config_io import *
from geom_utils import *

Nav = 6.02214076e+23 #Avogadro number
kboltz = 3.297623483e-24 * 1e-3 # Unit: kcal/K
econv = kboltz*Nav # Energy conversion factor to kcal/mol. Energies are often
#given in the literature in the form of U/k_b, which is in unit of Kelvin.
#The conversion factor econv gives U in kcal/mol.

config = Configuration()
config.add_simbox(0, 10, 0, 10, 0, 10)

#Atom types (Names and mass from Towhee force field file towhee_ff_TraPPE-UAf)
config.add_atom_type(mass=15.0347, name='CH3*(sp3)') 
config.add_atom_type(mass=14.0268, name='CH2**(sp3)') 

#Pair coeffs (Energy unit: kcal/mol)
#Pair style `lj/cut`
config.set_pair_coeff(1, [98*econv, 3.75])
config.set_pair_coeff(2, [46*econv, 3.95])

#Bond types
#Bond style `harmonic`
bond_length = 1.54
config.add_bond_type(params=[0.5*452900*econv, bond_length])

#Angle types
#Angle style `harmonic`
config.add_angle_type(params=[0.5*62500*econv, 114])

#Dihedral types
#Dihedral style `opls`
C1 = 355.05*econv; C2 = -68.19*econv; C3 = 791.32*econv
config.add_dihedral_type(params=[2*C1, 2*C2, 2*C3, 0.0])

#Atoms
natoms = 6
config.add_atom(1, 0.0, np.zeros((3,)), imol=1)
for i in range(2, natoms+1):
    at = 1 if (i==natoms) else 2
    config.append_atom_bonded(at, 0.0, bond_length, 'efjc', i-1, sep=bond_length)

#Bonds
nbonds = natoms - 1
for i in range(1, nbonds+1):
    config.add_bond(1, i, i+1)

#Angles
nangles = natoms - 2
for i in range(1, nangles+1):
    config.add_angle(1, i, i+1, i+2)

#Dihedrals
ndihedrals = natoms - 3
for i in range(1, ndihedrals+1):
    config.add_dihedral(1, i, i+1, i+2, i+3)

#Bring the barycenter to the origin
r = config.get_barycenter()
config.translate(-r, only_atoms=True) 

#Alter simulation box to have a spacing of 10 A all round and apply PBC
config.fit_simbox(sep=25.0)
config.apply_pbc()

#Write out the configuration
nam = 'nhexane' 
write_ldf(config, nam+'_uo.lmp', title=nam) #uo: structure not energy minimized
