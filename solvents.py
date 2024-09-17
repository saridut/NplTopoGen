#!/usr/bin/env python

from molecule import Molecule

#Solvents (with density at 25 C)
Ipa = (Molecule('Ipa', 'solvents/ipa/slvnt_ipa.lmp'), 0.78149)
Mch = (Molecule('Mch', 'solvents/mch/slvnt_mch.lmp'), 0.766)
Hexane = (Molecule('Hexane', 'solvents/hexane/slvnt_hexane.lmp'), 0.65485)
Methanol = (Molecule('Methanol', 'solvents/methanol/slvnt_methanol.lmp'), 0.78633)
Butanol = (Molecule('Butanol', 'solvents/butanol/slvnt_butanol.lmp'), 0.80577)
Acetone = (Molecule('Acetone', 'solvents/acetone/slvnt_acetone.lmp'), 0.78658)
Toluene = (Molecule('Toluene', 'solvents/toluene/slvnt_toluene.lmp'), 0.86224)
OleicAcid = (Molecule('OleicAcid', 
            'ligands/carboxy_acid/C18_usat/C18_usat_acid.lmp'), 0.895)
