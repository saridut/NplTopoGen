#!/usr/bin/env python

import numpy as np
import crystal
from nanoplatelet import NanoPlatelet
from molecule import hexane, toluene, mch
from molecule import acetate, oleate

xtal = crystal.CdSe_zb

#print(xtal.a)
#print(xtal.b)
#print(xtal.c)
#print(xtal.ucell)
#print('num_atom_types: ', xtal.get_num_atom_types())
#for i in [1, 2]:
#    print(f'atom_names[{i}]: ', xtal.get_atom_name(i))
#    print(f'atom_mass[{i}]: ', xtal.get_atom_mass(i))
#    print(f'num_atoms[{i}]: ', xtal.get_num_atoms(i))
#
#for i in range(1, 9):
#    print(xtal.get_atom_type(i))
#
#print(xtal.get_atom_type())
#print(xtal.get_num_atoms())

#NPL without ligands
#npl = NanoPlatelet(xtal, length=1, width=1, unit='lattice', 
#                   num_mono_layers=2, phi=45, charges=[1.18, -1.18],
#                   pair_coeffs=[[0.0334, 1.98], [0.0296, 5.24]]
#                   )
#npl.write('ldf', 'test')
#npl.write('xyz')


#Ligand molecule
#lig_ac = LigandMolecule('C2', 'ligands/carboxy_acid/C2/C2_anion.lmp',
#                     head=2, tail=1, bind_group=[2,3,4])
#lig_ol = LigandMolecule('C18', 'ligands/carboxy_acid/C18_usat/C18_usat_anion.lmp',
#                     head=18, tail=1, bind_group=[18,19,20])

packmol_path='~/soft/packmol/packmol'

#Ligand coated NPL
npl = NanoPlatelet(xtal, length=10, width=5, unit='nm', 
                   num_mono_layers=3, phi=45,
                   charges=[1.18, -1.18],
                   pair_coeffs=[[0.0334, 1.98], [0.0296, 5.24]]
                   )

npl.add_ligands([oleate], [1], dist=0, packmol_tol=0.5,
                packmol_path=packmol_path)

npl.tweak_cvff_impropers()
npl.write('test.lmp')

#hexane = SolventMolecule('hexane', 'solvents/hexane/slvnt_hexane.lmp') #0.6606)
#toluene = SolventMolecule('toluene', 'solvents/toluene/slvnt_toluene.lmp')# 0.86461

solvent = [mch]
npl.solvate(solvent, [1.0], boxx=20.0, packmol_tol=0.5 ,
            packmol_path=packmol_path)

npl.tweak_cvff_impropers()

npl.write('test_wsolvent.lmp', title='Test', fn_mg='molrec.txt')
