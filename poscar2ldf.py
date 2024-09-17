#!/usr/bin/env python

"""
Converts a POSCAR file to a LAMMPS data file. Note: Contains hard coded values.

"""

import sys
import os
import numpy as np
from _configuration import Configuration
from _config_io import write_ldf, add_molecules
from ligands import acetate

#The POSCAR file contains a CdSe crystal and `N` acetate molecules in a periodic
#box. OPLS-AA forcefield will be included in the LAMMPS data file.

fn_poscar = sys.argv[1]
fn_ldf = os.path.join(os.path.dirname(fn_poscar), 'config.lmp')

print(f"fn_poscar = {fn_poscar}")

basis = np.zeros((3,3))
with open(fn_poscar, 'r') as fh:
    #Skip the first line
    fh.readline()
    scalef = float( fh.readline().strip(" \n") )
    #Basis vectors
    for i in range(3):
        comps = [float(x) for x in fh.readline().strip(" \n").split()]
        basis[i,0] = comps[0]; basis[i,1] = comps[1]; basis[i,2] = comps[2]
        basis *= scalef
    #Element names and population
    elem_names = fh.readline().strip(" \n").split()
    elem_pop = [int(x) for x in fh.readline().strip(" \n").split()]
    num_atoms = sum(elem_pop)
    #Coordinate specification
    coordspec = fh.readline().strip(" \n")
    #Coordinates
    poscar_coords = np.zeros((num_atoms, 3))
    for i in range(num_atoms):
        comps = [float(x) for x in fh.readline().strip(" \n").split()]
        poscar_coords[i,0] = comps[0]
        poscar_coords[i,1] = comps[1]
        poscar_coords[i,2] = comps[2]
    if coordspec[0] in 'CcKk':
        poscar_coords *= scalef
    else:
        poscar_coords = poscar_coords @ basis

config = Configuration()
config.add_simbox(0.0, basis[0,0], 0.0, basis[1,1], 0.0, basis[2,2])

#Add Cd atoms
at_Cd = config.add_atom_type(mass=112.411, name='Cd')
for i in range(elem_pop[0]):
    coords = poscar_coords[i,:]
    config.add_atom(at_Cd, 1.18, coords)
config.set_pair_coeff(at_Cd, [0.0334, 1.98, 10.0])

#Add Se atoms
at_Se = config.add_atom_type(mass=78.96, name='Se')
for i in range(elem_pop[1]):
    offset = elem_pop[0]
    j = i + offset
    coords = poscar_coords[j,:]
    config.add_atom(at_Se, -1.18, coords)
config.set_pair_coeff(at_Se, [0.0296, 5.24, 10.0])

print("Total charge of xtal = %g"%config.get_total_charge())

#Add molecules
num_molecules = 4
g_in = 'inside box '
g_in += ' '.join(['%g'%x for x in config.simbox[:,0]])
g_in += ' '
g_in += ' '.join(['%g'%x for x in config.simbox[:,1]])
molecules = [{'moltem':acetate, 'num':num_molecules, 'constraints':[g_in]}]

add_molecules(config, molecules, packmol_tol=2.0, 
              packmol_path="~/soft/packmol/packmol")

#Update positions of ligand atoms
pcmoff = elem_pop[0] + elem_pop[1] #Offset for all molecules
#POSCAR -> Lammps atom type map
atm_id_map = np.array([3, 4, 1, 2, 5, 6, 7], dtype=np.int32)
for imol in range(1, num_molecules+1):
    ibeg = pcmoff + (imol-1)*acetate.num_atoms
    iend = ibeg + acetate.num_atoms
    pc_mol_coords = poscar_coords[ibeg:iend,:]
    pc_mol_coords = pc_mol_coords[atm_id_map-1,:]
    atm_beg = config.molecules[imol]['atm_beg']
    atm_end = config.molecules[imol]['atm_end']

    for i in range(acetate.num_atoms):
        config.atoms[atm_beg+i]['coords'] = pc_mol_coords[i,:]

config.unwrap_pbc()
for i in range(1, config.num_atoms+1):
    config.add_img_flag(i, np.zeros((3,)))

config.apply_pbc()

#Convert impropers to integers
for val in config.improper_coeffs.values():
    val[1] = int(val[1]); val[2] = int(val[2])

print("Total charge after adding ligands = %g"%config.get_total_charge())

write_ldf(config, fn_ldf)
