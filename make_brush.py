#!/usr/bin/env python

import math
import numpy as np
from molecule import LigandMolecule
from brush import Brush
from solvents import Hexane, Toluene, Mch, OleicAcid
#from crystal import get_lattice_points

#coords = get_lattice_points('sc', [1.5, 1.5, 1.0], [-2,-2,0], [2,2,0], 'ncn')

#for i,each in enumerate(coords):
#    print(i, each)
#
#raise SystemExit()

packmol_path='~/soft/packmol/packmol'
#packmol_path='~/soft/packmol-20.15.0/packmol'

#Here the bind_group indicates the local atom ids that are bound to the head.
#These atoms must be below the wall.
oa = LigandMolecule('OA', 'ligands/carboxy_acid/C18_usat/C18_usat_acid.lmp',
        head=18, tail=1, bind_group=[19,20,54])
oa.translate_atom(oa.head, np.zeros((3,)))
oa.align(18, 1, np.array([0,0,1]))
oa.fit_simbox()
r = oa.get_gyration_radius(atoms=oa.bind_group)[0]

#d = 4.0 #2*(r+1.5)
#rcut = 2.0
#sigma = 2**(-1/6)*(rcut)
#print(f"sigma = {sigma}, rcut = {rcut}, d = {d}")
#oa.write('oa.lmp')
#raise SystemExit()


tag = 'brush_6x6_oa'
gdens = 0.03 #Grafting density, in #/ang^2
lx = 60     #ly = lx
lz = 100 #75

teth_dist = 5.0 #d
ligands = [oa]
ligand_pop_ratio = np.array([1])

solvent = Hexane #Toluene
solvent_exclude = (0, 10)
    
brush = Brush(lx, lz)

#Ligands
#brush.add_ligands(ligands, ligand_pop_ratio, gdens, teth_dist, 'bcc',
#    packmol_path=None)
brush.add_ligand_one(ligands[0], teth_dist)
print(f"Brush density = {brush.gdens: f}/angstrom^2")
 
#brush.apply_pbc(directions='xy', add_img_flag=True)
#brush.tweak_cvff_impropers()
#brush.adjust_charge()

#brush.write(f"{tag}.lmp", title=f"{tag}", fn_mg=f"{tag}_mg.txt",
#          with_pc=False)
#brush.gen_ff_pair(fn=f"{tag}_pcoeff.lmp", soften=None)
#brush.gen_ff_pair(fn=f"{tag}_pcoeff_soft.lmp", soften='ligands')
#raise SystemExit()

#Solvent
brush.solvate(solvent, solvent_exclude, packmol_tol=2.1,
              packmol_sidemax=1.0e3, packmol_path=packmol_path)

brush.tweak_cvff_impropers()
brush.adjust_charge()
#brush.apply_pbc(directions='xy', add_img_flag=True)

#brush.write(f"slv_{tag}.lmp", title=f"slv_{tag}", fn_mg=f"slv_{tag}_mg.txt",
#          with_pc=False)

#brush.gen_ff_pair(fn=f"slv_{tag}_pcoeff.lmp", soften=None)
#brush.gen_ff_pair(fn=f"slv_{tag}_pcoeff_soft.lmp", soften='both')

#brush.gen_ff_pair(fn=f"slv_{tag}_pcoeff_soft_lig.lmp", soften='ligands')
#brush.gen_ff_pair(fn=f"slv_{tag}_pcoeff_soft_sol.lmp", soften='solvent')

#Piston
pa_eps = 5.29 #0.2384
pa_sigma = 2.629
pa_rcut = None
pa_lpar = 4.0778
pa_mass = 196.967
pa_thickness = 3*pa_lpar
brush.add_piston('top', pa_lpar, pa_thickness, pa_mass,
                pa_eps, pa_sigma, pa_rcut)

brush.apply_pbc(directions='xy', add_img_flag=True)

#brush.write_bond_coeffs(fn=f"slv_{tag}_sbcoeff.lmp", kbond=6.0e7)

brush.gen_ff_pair(fn=f"slv_{tag}_pcoeff.lmp", soften=None)
brush.gen_ff_pair(fn=f"slv_{tag}_pcoeff_soft.lmp", soften='both')
brush.write(f"slv_{tag}.lmp", title=f"slv_{tag}", fn_mg=f"slv_{tag}_mg.txt",
          with_pc=False)
 
#raise SystemExit()


