#!/usr/bin/env python

import math
import numpy as np
import crystal
from nanoplatelet import NanoPlatelet
from solvents import Toluene, Mch, OleicAcid
from ligands import Acetate, Oleate, Octanoate, Butanoate

xtal = crystal.CdSe_zb

packmol_path='~/soft/packmol/packmol'
tag = 'xtal_ML4_5x5_45'

#Ligand coated NPL
npl = NanoPlatelet(length=50, width=50, unit='ang', 
                   num_mono_layers=4, phi=45, pbc=True, balanced=False)

npl.add_xtal(xtal, charges=[1.18, -1.18], 
             pair_coeffs=[[0.0334, 1.98, 10.0], [0.0296, 5.24, 10.0]])

#fn = '/Users/sdutta/workspace/npl/npl_3_10x10_toluene/octanoate/nosolvent/post.trup.lmp'
#fn_mg = '/Users/sdutta/workspace/npl/npl_3_10x10_toluene/octanoate/nosolvent/lc_ML3_10x10_45_mg.txt'
#fn_pc = '/Users/sdutta/workspace/npl/npl_3_10x10_toluene/octanoate/nosolvent/lc_ML3_10x10_45_pcoeff.lmp'
#npl.read(fn, fn_mg=f"{fn_mg}", fn_pc=f"{fn_pc}")
#npl.velocities.clear()


npl.write(f"{tag}.lmp", title=tag, fn_mg='', with_pc=True)
raise SystemExit()

#Oleate: thickness 22

ligs = [Oleate]
ntyps = np.array([npl.num_atom_types, npl.num_bond_types, npl.num_angle_types,
            npl.num_dihedral_types, npl.num_improper_types], dtype=np.int32)
type_offsets = []
for each in ligs:
    type_offsets.append( tuple(ntyps) )
    new = np.array([each.num_atom_types, each.num_bond_types, each.num_angle_types,
                 each.num_dihedral_types, each.num_improper_types], dtype=np.int32)
    ntyps += new

npl.add_ligands(ligs, [1], type_offsets, offset=6.0, thickness=22.0, packmol_tol=2.0,
                packmol_sidemax=1.0e3, packmol_path=packmol_path)

npl.tweak_cvff_impropers()

npl.write(f"lc_{tag}.lmp", title=f"lc_{tag}", fn_mg=f"lc_{tag}_mg.txt",
          with_pc=False)
npl.gen_ff_pair(fn=f"lc_{tag}_pcoeff.lmp", soften=None)
npl.gen_ff_pair(fn=f"lc_{tag}_pcoeff_soft_lig.lmp", soften='ligands')
#raise SystemExit()

#Add solvent
solvent = [Toluene]
boxx = npl.simbox[0,1] - npl.simbox[0,0]
boxy = npl.simbox[1,1] - npl.simbox[1,0]
boxz = 150.0
dist = 28.0
ntyps = np.array([npl.num_atom_types, npl.num_bond_types, npl.num_angle_types,
            npl.num_dihedral_types, npl.num_improper_types], dtype=np.int32)
type_offsets = []
for comp in solvent:
    type_offsets.append( tuple(ntyps) )
    each = comp[0]
    new = np.array([each.num_atom_types, each.num_bond_types, each.num_angle_types,
                 each.num_dihedral_types, each.num_improper_types], dtype=np.int32)
    ntyps += new

npl.solvate(solvent, [1.0], type_offsets, boxx, boxy, boxz, dist, 
            packmol_tol=0.6, packmol_sidemax=1.0e3, packmol_path=packmol_path)
npl.apply_pbc()
npl.tweak_cvff_impropers()

npl.write(f"slc_{tag}.lmp", title=f"slc_{tag}", fn_mg=f"slc_{tag}_mg.txt",
          with_pc=False)

npl.gen_ff_pair(fn=f"slc_{tag}_pcoeff.lmp", soften=None)
#npl.gen_ff_pair(fn=f"slc_{tag}_pcoeff_soft.lmp", soften='both')
npl.gen_ff_pair(fn=f"slc_{tag}_pcoeff_soft_lig.lmp", soften='ligands')
npl.gen_ff_pair(fn=f"slc_{tag}_pcoeff_soft_sol.lmp", soften='solvent')

