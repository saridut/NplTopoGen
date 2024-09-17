#!/usr/bin/env python

'''
Creates a NPL coated with ligands.

'''
import sys
import os
import copy
import subprocess
import math
import numpy as np
from configuration import Configuration
from config_io import *
from geom_utils import *

#-------------------------------------------------------------------------------

#All ligands must be monovalent
ligand_types = {
        1: {'name': 'C8',
            'fn': 'ligands/thiol/C8/C8_anion.lmp',
            'fn_xyz': None, 'nfrac': 1.0, 'chge': 0.0, 'pop': 0, 
            'head': 8, 'tail': 1, 'bndgrp':[9], 'cfg': None} #,
#       2: {'name': 'C18',
#           'fn': 'ligands/carboxy_acid/C18_usat/C18_usat_anion.lmp',
#           'fn_xyz': None, 'nfrac': 0.5, 'chge': 0.0, 'pop': 0, 
#           'head': 18, 'tail': 1, 'bndgrp':[18,19,20], 'cfg': None}
           }

#ligand_types = {
#        1: {'name': 'C2',
#            'fn': 'ligands/carboxy_acid/C2/C2_anion.lmp',
#            'fn_xyz': None, 'nfrac': 1.0, 'chge': 0.0, 'pop': 0, 
#            'head': 2, 'tail': 1, 'bndgrp':[2,3,4], 'cfg': None}
#           }

#ligand_types = {
#        1: {'name': 'C18',
#             'fn': 'ligands/carboxy_acid/C18_usat/C18_usat_anion.lmp',
#             'fn_xyz': None, 'nfrac': 1.0, 'chge': 0.0, 'pop': 0, 
#             'head': 18, 'tail': 1, 'bndgrp':[18,19,20], 'cfg': None}
#           }

with_solvent = True
solvent = {'name': 'hexane', 
            'fn': 'solvents/hexane/slvnt_hexane.lmp',
            'dens': 0.6606, # 0.6606 in g/mL
            'fn_xyz': None, 'pop': 0, 'cfg': None}

tag = '_thC8'
write_molrec = False
write_fn_ldf = True
write_fn_xyz = False

#Crystal
fn_xtal = 'ML5_20x10_0.lmp'
config = Configuration()
read_ldf(config, fn_xtal)

title_npl = 'NPL_' + os.path.basename(fn_xtal).rstrip('.lmp') + tag
fn_ldf_npl = title_npl + '.lmp'
fn_xyz_npl = title_npl + '.xyz'

fn_molrec = 'molrec%s.txt'%tag

#Move the xtal to the origin
r = config.simbox.mean(axis=1)
config.translate(-r)
na_xtal = len(config.atoms)
totchge = config.get_total_charge()
print('totchge : %g'%totchge)

num_ligand_types = len(ligand_types)
lig_size_max = 0.0
for each in ligand_types.values():
    #Ligand configuration: 
    each['cfg'] = Configuration(); cfgl = each['cfg']
    cfgl.add_simbox(0, 1, 0, 1, 0, 1)
    read_ldf(cfgl, each['fn'])
    cfgl.fit_simbox(sep=0.0)
    size = np.amax(cfgl.simbox[:,1] - cfgl.simbox[:,0])
    lig_size_max = max(lig_size_max, size)
    each['chge'] = cfgl.get_total_charge()
    each['pop'] = int(np.rint(totchge*each['nfrac']))
    print('ligand_pop: %s %g'%(each['name'], each['pop']))


num_ligands = 0
for each in ligand_types.values():
    num_ligands += each['pop']

#New simulation box
xtal_xlo = config.simbox[0,0]; xtal_xhi = config.simbox[0,1]
xtal_ylo = config.simbox[1,0]; xtal_yhi = config.simbox[1,1]
xtal_zlo = config.simbox[2,0]; xtal_zhi = config.simbox[2,1]
xtal_lo = np.array([xtal_xlo, xtal_ylo, xtal_zlo])
xtal_hi = np.array([xtal_xhi, xtal_yhi, xtal_zhi])

config.simbox[0,0] -= (3*lig_size_max)
config.simbox[1,0] -= (3*lig_size_max)
config.simbox[2,0] =  config.simbox[0,0]
config.simbox[:,1] = -config.simbox[:,0]

#Region
region = {'inout': 'out', 'lo': xtal_lo-0.1, 'hi': xtal_hi+0.1}
#Overlap params
check_overlap = False
overlap_params = {'key': 'atm_type', 'val': np.array([1,2], dtype=np.int32),
                  'sep': 2.0, 'maxitr' : 1000
                  }
#Add ligands
for key, val in ligand_types.items():
    add_molecules(config, val['cfg'], val['pop'], region = region,
                  check_overlap=check_overlap, overlap_params=overlap_params)
#Adjust charge
chge = config.get_total_charge()
if chge != 0:
    cpa = chge/len(config.atoms)
    for i in range(1, len(config.atoms)+1):
        config.atoms[i]['charge'] -= cpa

#Write molrec
if write_molrec:
    fh_mr = open(fn_molrec, 'w')
    fh_mr.write('%d\n'%num_ligands)
    fh_mr.write('   imol moltyp iatm_beg iatm_end head tail bndgrp_pop bndgrp\n')
    imol = 0; iatm_end = na_xtal
    for key, val in ligand_types.items():
        for i in range(val['pop']):
            imol += 1
            iatm_beg = iatm_end + 1
            iatm_end = iatm_beg + len(val['cfg'].atoms) - 1
            buf = '%7d %6d %8d %8d %4d %4d %5d '%(imol, key, iatm_beg, iatm_end,
                    val['head'], val['tail'], len(val['bndgrp']))
            buf += '     '
            buf += ' '.join('%d'%x for x in val['bndgrp'])
            fh_mr.write(buf+'\n')
    fh_mr.close()


#Solvent
if with_solvent:
    Nav = 6.023*1e23
    #solvent configuration: 
    solvent['cfg'] = Configuration(); cfgs = solvent['cfg']
    cfgs.add_simbox(0, 1, 0, 1, 0, 1)
    read_ldf(cfgs, solvent['fn'])
    chge = cfgs.get_total_charge()
    if chge != 0:
        cpa = chge/len(cfgs.atoms)
        for i in range(1, len(cfgs.atoms)+1):
            cfgs.atoms[i]['charge'] -= cpa
    cfgs.fit_simbox(sep=0.0)

    rho = solvent['dens']*1e-24 #Overall density in g/angstrom^3, 
                                #assuming same as that of the solvent
    #Volume of ligands+solvent
    vol = np.prod(config.simbox[:,1]-config.simbox[:,0]) 
    mass = rho*vol #Overall mass of ligands+solvent
    mass_ligands = 0.0
    for val in ligand_types.values():
        mass_ligands += ( val['pop']*val['cfg'].get_total_mass()/Nav )
    mass_solvent = mass - mass_ligands
    mw_solvent = solvent['cfg'].get_total_mass() #Mol. wt. of solvent
    print('mw_solvent : %g'%mw_solvent)
    print('mass : %g'%mass)
    print('mass_ligands : %g'%mass_ligands)
    print('mass_solvent : %g'%mass_solvent)
    solvent['pop'] = int( (mass_solvent/mw_solvent)*Nav )
    print('solvent_pop : %g'%solvent['pop'])
    add_molecules(config, solvent['cfg'], solvent['pop'], region = region,
                  check_overlap=check_overlap, overlap_params=overlap_params)


#Apply PBC
config.apply_pbc()

#Balance charges
totchge = config.get_total_charge()

#Tweak for CVFF impropers
for val in config.improper_coeffs.values():
    val[1] = int(val[1]); val[2] = int(val[2])

print('Total charge: ', config.get_total_charge())

#Write out the NPL+ligand configuration
if write_fn_ldf :
    print('fn_out_ldf : ', fn_ldf_npl)
    write_ldf(config, fn_ldf_npl, title=title_npl)
if write_fn_xyz :
    print('fn_out_xyz : ', fn_xyz_npl)
    write_xyz(config, fn_xyz_npl, title=title_npl)
