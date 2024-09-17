#!/usr/bin/env python

import numpy as np
from molecule import Molecule
from solvents import Toluene, Mch, OleicAcid
from solventbox import SolventBox

packmol_path='~/soft/packmol/packmol'
tag = "toluenebox"

#solutes = [Molecule(name='solute:OleicAcid', 
#            fn='ligands/carboxy_acid/C18_usat/C18_usat_acid.lmp'),
#           Molecule(name='solute:AceticAcid', 
#            fn='ligands/carboxy_acid/C2/C2_acid.lmp')]
#solute_comp_pop = [2, 3]
solutes = []; solute_comp_pop = []

solvent = [Toluene]
system = SolventBox(solvent, [1.0], boxx=50.0, 
                    solutes=solutes, solute_comp_pop=solute_comp_pop,
                    force_neutral=True, 
                    packmol_tol=0.1, packmol_path=packmol_path)

system.tweak_cvff_impropers()

system.write(fn=f"{tag}.lmp", title=f"{tag}", fn_mg=f"{tag}_mg.txt",
             with_pc=False)

system.gen_ff_pair(fn=f"{tag}_pcoeff.lmp", soften=False)
system.gen_ff_pair(fn=f"{tag}_pcoeff_soft.lmp", soften=True)

