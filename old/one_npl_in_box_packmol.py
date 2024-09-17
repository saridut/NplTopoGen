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
        1: {'name': 'C8_anion',
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
            'dens': 0.6606, #in g/mL
            'fn_xyz': None, 'pop': 0, 'cfg': None}


tag = '_thC8'
write_molrec = True
write_fn_ldf = True
write_fn_xyz = True

#Crystal
fn_xtal = 'ML5_20x10_0.lmp'
#fn_xtal = 'ML3_10x10.lmp'
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
    fn_xyz = 'lg_%s.xyz'%each['name']
    write_xyz(cfgl, fn_xyz, title=fn_xyz)
    each['fn_xyz'] = fn_xyz
    each['pop'] = int(np.rint(totchge*each['nfrac']))


num_ligands = 0
for each in ligand_types.values():
    num_ligands += each['pop']

#Bounding box for PACKMOL
xtal_xlo = config.simbox[0,0]; xtal_xhi = config.simbox[0,1]
xtal_ylo = config.simbox[1,0]; xtal_yhi = config.simbox[1,1]
xtal_zlo = config.simbox[2,0]; xtal_zhi = config.simbox[2,1]

#Reduce by 1 Angstrom along x, y, & z for the inner box.
pm_box_in = np.array([[xtal_xlo-1, xtal_xhi+1],
                      [xtal_ylo-1, xtal_yhi+1],
                      [xtal_zlo-1, xtal_zhi+1]]) 
#Outer box
pm_box_out = copy.deepcopy(pm_box_in)
pm_box_out[0,0] -= (5*lig_size_max)
pm_box_out[1,0] -= (5*lig_size_max)
pm_box_out[2,0] =  pm_box_out[0,0]
pm_box_out[:,1] = -pm_box_out[:,0]

#Make simbox same as the out box
config.simbox = copy.deepcopy(pm_box_out)

#Add ligands
for key, val in ligand_types.items():
    add_molecules(config, val['cfg'], val['pop'])
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
    solvent['cfg'] = Configuration(); cfgl = solvent['cfg']
    cfgl.add_simbox(0, 1, 0, 1, 0, 1)
    read_ldf(cfgl, solvent['fn'])
    cfgl.fit_simbox(sep=0.0)
    fn_xyz = 'slvnt_%s.xyz'%solvent['name']
    write_xyz(cfgl, fn_xyz, title=fn_xyz)
    solvent['fn_xyz'] = fn_xyz

    rho = solvent['dens']*1e-24 #Overall density in g/angstrom^3, 
                                #assuming same as that of the solvent
    #Volume of ligands+solvent
    vol = np.prod(pm_box_out[:,1]-pm_box_out[:,0]) - \
            np.prod(pm_box_in[:,1]-pm_box_in[:,0])
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
    add_molecules(config, solvent['cfg'], solvent['pop'])
    #raise SystemExit()


#Write out packmol file
fn_pm_in = 'inp_npl_pm.txt'
fn_pm_out = 'npl_pm.xyz'
with open(fn_pm_in, 'w') as fh:
    fh.write('tolerance 2.0\n')
    sidemax = np.amax(pm_box_out[:,1]-pm_box_out[:,0])
    fh.write('sidemax %g\n'%sidemax)
    fh.write('seed -1\n')
    fh.write('randominitialpoint\n')
    fh.write('movebadrandom yes\n')
    fh.write('output %s\n'%fn_pm_out)
    fh.write('filetype xyz\n')
    fh.write('\n')
    for each in ligand_types.values():
        fh.write('structure %s\n'%each['fn_xyz'])
        fh.write('    number %d\n'%each['pop'])
        buf = ' '.join([str(x) for x in pm_box_out.flatten('F')])
        fh.write('    inside box %s\n'%buf)
        buf = ' '.join([str(x) for x in pm_box_in.flatten('F')])
        fh.write('    outside box %s\n'%buf)
        fh.write('end structure\n')
    if with_solvent:
        fh.write('structure %s\n'%solvent['fn_xyz'])
        fh.write('    number %d\n'%solvent['pop'])
        buf = ' '.join([str(x) for x in pm_box_out.flatten('F')])
        fh.write('    inside box %s\n'%buf)
        buf = ' '.join([str(x) for x in pm_box_in.flatten('F')])
        fh.write('    outside box %s\n'%buf)
        fh.write('end structure\n')

#Run packmol
args_run = ["~/soft/packmol/packmol < %s"%fn_pm_in]
subprocess.run(args_run, shell=True)

#Read back the packmol output
read_xyz(config, fn_pm_out, offset=na_xtal)
subprocess.run(["rm %s %s"%(fn_pm_out, fn_pm_in)], shell=True)
for each in ligand_types.values():
    subprocess.run(["rm %s"%(each['fn_xyz'])], shell=True)
if with_solvent:
    subprocess.run(["rm %s"%(solvent['fn_xyz'])], shell=True)

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
