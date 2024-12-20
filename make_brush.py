#!/usr/bin/env python

import sys
import math
import numpy as np
from molecule import LigandMolecule
from brush import Brush
from solvents import Hexane, Toluene, Mch, OleicAcid

packmol_path='~/soft/packmol/packmol'

lig_cx = LigandMolecule('C18', 'ligands/carboxy_acid/C18/C18.lmp',
        head=19, tail=18, bind_group=[19])

xa_mass = 12.011
xa_eps = 0.25
xa_rcut = 2.0
xa_sigma = 2**(-1/6)*xa_rcut
xlpar = xa_sigma*2/math.sqrt(2.0) #Lattice parameter for wall atoms (fcc lattice)

xtal_lx = 100
xtal_ly = 100
xtal_lz = 12
box_params = {'sepz': 90}
apl = int(sys.argv[1]) #Area per ligand in ang^2

dn = f"C18-hexane/apl_{apl}"

ligands = [lig_cx]
ligand_pop_ratio = np.array([1])
solvent = Hexane #Toluene
tag = f"brush_C18_apl_{apl:g}_Hexane"
    
brush = Brush(is_slab=True, slab_pos='mid')

#Add crystal
brush.add_xtal(xtal_lx, xtal_ly, xtal_lz, xlpar, xa_mass, xa_eps, xa_sigma,
               xa_rcut, out_dir=dn)

#Add ligands
brush.add_ligands(ligands, ligand_pop_ratio, r0=xa_rcut, balance_charge=False,
                  lattice='bcc', apl=apl, out_dir=dn)
#
#brush.tweak_cvff_impropers()
#brush.adjust_charge()
#brush.apply_pbc(directions='xy', add_img_flag=True)

brush.write(f"{dn}/{tag}.lmp", title=f"{tag}", fn_mg='', with_pc=False)
#brush.gen_ff_pair(fn=f"{tag}_pcoeff.lmp", soften=None)
#brush.gen_ff_pair(fn=f"{tag}_pcoeff_soft.lmp", soften='ligands')

#Solvent
brush.solvate(solvent, box_params, packmol_tol=2.0, packmol_sidemax=1.0e3,
              packmol_path=packmol_path)
#brush.simbox[2,1] += 1; brush.simbox[2,0] = -brush.simbox[2,1]
brush.tweak_cvff_impropers()
brush.adjust_charge()
#brush.apply_pbc(directions='xyz', add_img_flag=True)

#brush.gen_ff_pair(fn=f"slv_{tag}_pcoeff.lmp", soften=None)
#brush.gen_ff_pair(fn=f"slv_{tag}_pcoeff_soft.lmp", soften='both')
#brush.write(f"slv_{tag}.lmp", title=f"slv_{tag}", fn_mg=f"slv_{tag}_mg.txt",
#         with_pc=False)

#Piston
pa_eps = 5.29
pa_sigma = 2.629
pa_rcut = 12.0
pa_lpar = 4.0778
pa_mass = 196.967
pa_thickness = 3*pa_lpar
brush.add_piston('both', pa_lpar, pa_thickness, pa_mass,
                pa_eps, pa_sigma, pa_rcut, out_dir=dn)

brush.apply_pbc(directions='xy', add_img_flag=True)

brush.gen_ff_pair(fn=f"{dn}/slv_{tag}_pcoeff.lmp", soften=None)
brush.gen_ff_pair(fn=f"{dn}/slv_{tag}_pcoeff_soft.lmp", soften='both')
brush.write(f"{dn}/slv_{tag}.lmp", title=f"slv_{tag}", fn_mg=f"{dn}/slv_{tag}_mg.txt",
          with_pc=False)
