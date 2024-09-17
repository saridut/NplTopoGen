#!/usr/bin/env python

import numpy as np
from molecule import Molecule
from solvents import Toluene, Mch, OleicAcid

packmol_path='~/soft/packmol/packmol'

system = Molecule(name='OleicAcid', 
            fn='ligands/carboxy_acid/C18_usat/C18_usat_acid.lmp')

print(system.simbox)

solvent = [Toluene]
system.solvate(solvent, [1.0], boxx=100.0, packmol_tol=0.5 ,
            packmol_path=packmol_path)

system.tweak_cvff_impropers()

system.write('test_solvated.lmp', title='Test', fn_mg='test_molrec.txt')
