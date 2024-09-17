#!/usr/bin/env python

from molecule import LigandMolecule


#Ligands
Acetate = LigandMolecule('Acetate', 
            'ligands/carboxy_acid/C2/C2_anion.lmp',
            head=2, tail=1, bind_group=[2,3,4])
Butanoate = LigandMolecule('Butanoate', 
                'ligands/carboxy_acid/C4/C4_anion.lmp',
                head=4, tail=1, bind_group=[4,5,6])
Hexanoate = LigandMolecule('Butanoate',
                'ligands/carboxy_acid/C6/C6_anion.lmp',
                head=6, tail=1, bind_group=[6,7,8])
Octanoate = LigandMolecule('Octanoate', 
                'ligands/carboxy_acid/C8/C8_anion.lmp',
                head=8, tail=1, bind_group=[8,9,10])
Decanoate = LigandMolecule('Decanoate', 
                'ligands/carboxy_acid/C10/C10_anion.lmp',
                head=10, tail=1, bind_group=[10,11,12])
Dodecanoate = LigandMolecule('Dodecanoate',
                'ligands/carboxy_acid/C12/C12_anion.lmp', 
                head=12, tail=1, bind_group=[12,13,14])
Tetradecanoate = LigandMolecule('Tetradecanoate',
                'ligands/carboxy_acid/C14/C14_anion.lmp', 
                head=14, tail=1, bind_group=[14,15,16])
Hexadecanoate = LigandMolecule('Hexadecanoate',
                'ligands/carboxy_acid/C16/C16_anion.lmp', 
                head=16, tail=1, bind_group=[16,17,18])
Octadecanoate = LigandMolecule('Octadecanoate',
                'ligands/carboxy_acid/C18/C18_anion.lmp', 
                head=18, tail=1, bind_group=[18,19,20])
Icosanoate = LigandMolecule('Icosanoate',
                'ligands/carboxy_acid/C20/C20_anion.lmp', 
                head=20, tail=1, bind_group=[20,21,22])
Oleate = LigandMolecule('Oleate', 
                'ligands/carboxy_acid/C18_usat/C18_usat_anion.lmp',
                head=18, tail=1, bind_group=[18,19,20])

EthaneThiolate = LigandMolecule('EthaneThiolate', 
                    'ligands/thiol/C2/C2_anion.lmp',
                    head=2, tail=1, bind_group=[3])
ButaneThiolate = LigandMolecule('ButaneThiolate', 
                    'ligands/thiol/C4/C4_anion.lmp',
                    head=4, tail=1, bind_group=[5])
HexaneThiolate = LigandMolecule('HexaneThiolate', 
                    'ligands/thiol/C6/C6_anion.lmp',
                    head=6, tail=1, bind_group=[7])
OctaneThiolate = LigandMolecule('OctaneThiolate', 
                    'ligands/thiol/C8/C8_anion.lmp',
                    head=8, tail=1, bind_group=[9])
DecaneThiolate = LigandMolecule('DecaneThiolate', 
                    'ligands/thiol/C10/C10_anion.lmp',
                    head=10, tail=1, bind_group=[11])
DodecaneThiolate = LigandMolecule('DodecaneThiolate', 
                        'ligands/thiol/C12/C12_anion.lmp', 
                        head=12, tail=1, bind_group=[13])
TetradecaneThiolate = LigandMolecule('TetradecaneThiolate',
                        'ligands/thiol/C14/C14_anion.lmp', 
                        head=14, tail=1, bind_group=[15])
HexadecaneThiolate = LigandMolecule('HexadecaneThiolate',
                        'ligands/thiol/C16/C16_anion.lmp',
                        head=16, tail=1, bind_group=[17])
OctadecaneThiolate = LigandMolecule('OctadecaneThiolate',
                        'ligands/thiol/C18/C18_anion.lmp',
                        head=18, tail=1, bind_group=[19])
IcosaneThiolate = LigandMolecule('IcosaneThiolate',
                        'ligands/thiol/C20/C20_anion.lmp',
                        head=20, tail=1, bind_group=[21])
OleylThiolate = LigandMolecule('OleylThiolate',
                    'ligands/thiol/C18_usat/C18_usat_anion.lmp',
                    head=18, tail=1, bind_group=[19])
TwoEht = LigandMolecule('TwoEht', 
                    'ligands/thiol/2-eht/anion.lmp',
                    head=8, tail=1, bind_group=[9])
