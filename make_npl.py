#!/usr/bin/env python

import math
import numpy as np
import crystal
from nanoplatelet import NanoPlatelet
from solvents import Toluene, Mch, OleicAcid
from ligands import Acetate, Oleate, Octanoate, Butanoate

xtal = crystal.CdSe_zb

packmol_path='~/soft/packmol/packmol'
#packmol_path='~/soft/packmol-20.15.0/packmol'
tag = 'test'

#Ligand coated NPL
npl = NanoPlatelet(is_slab=False)
npl.add_xtal(xtal, length=50, width=50, num_mono_layers=3, phi=45,
             balanced=False, pbc_xy=False, pbc_z=False, unit='ang',
             charges=[1.18, -1.18], 
             pair_coeffs=[[0.0334, 1.98], [0.0296, 5.24]])

#npl.write(f"{tag}.lmp", title=tag, fn_mg=f"{tag}_mg.txt", with_pc=False)
#raise SystemExit()

ligs = [Octanoate]
npl.add_ligands(ligs, [1], offset=2.0, thickness=16.0, thickness_bm=4,
            packmol_tol=1.0, packmol_sidemax=1.0e3, packmol_path=packmol_path)

npl.tweak_cvff_impropers()

npl.write(f"lc_{tag}.lmp", title=f"lc_{tag}", fn_mg=f"lc_{tag}_mg.txt",
          with_pc=False)
#npl.gen_ff_pair(fn=f"lc_{tag}_pcoeff.lmp", soften=None)
#npl.gen_ff_pair(fn=f"lc_{tag}_pcoeff_soft_lig.lmp", soften='ligands')
raise SystemExit()

#Add solvent
solvent = [Toluene]
boxx = npl.simbox[0,1] - npl.simbox[0,0]
boxy = npl.simbox[1,1] - npl.simbox[1,0]
boxz = 120.0
dist = 10.0

npl.solvate(solvent, [1.0], boxx, boxy, boxz, dist, packmol_tol=2.1, 
            packmol_sidemax=1.0e3, packmol_path=packmol_path)

npl.tweak_cvff_impropers()
npl.adjust_charge()
npl.apply_pbc(directions='xy', add_img_flag=True)

npl.write(f"slc_{tag}.lmp", title=f"slc_{tag}", fn_mg=f"slc_{tag}_mg.txt",
          with_pc=False)

npl.gen_ff_pair(fn=f"slc_{tag}_pcoeff.lmp", soften=None)
#npl.gen_ff_pair(fn=f"slc_{tag}_pcoeff_soft.lmp", soften='both')
npl.gen_ff_pair(fn=f"slc_{tag}_pcoeff_soft_lig.lmp", soften='ligands')
npl.gen_ff_pair(fn=f"slc_{tag}_pcoeff_soft_sol.lmp", soften='solvent')

