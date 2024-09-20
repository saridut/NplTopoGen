#!/usr/bin/env python

import os
import warnings
import subprocess
from pathlib import Path
import numpy as np
#from _geom_utils import *

#-------------------------------------------------------------------------------

def write_xyz(config, fn, title=''):
    '''
    Writes configuration to an XYZ file.  

    '''

    na = config.num_atoms

    with open(fn,'w') as fh:
        fh.write(str(na) + '\n')
        fh.write(title + '\n')

        for i in range(1, na+1):
            at = config.atoms[i]['type']
            atm_nam = config.atom_names[at]
            coords = config.atoms[i]['coords']
            fh.write( '%s  '%atm_nam 
                + '  '.join(['% .15g'%x for x in coords]) + '\n')

#-------------------------------------------------------------------------------

def read_xyz(config, fn, offset=0):
    '''
    Reads atom positions from an XYZ file.
    '''
    
    with open(fn, 'r') as fh:
        lines = fh.readlines()

    na = int(lines[0].strip('\n'))
    assert na <= config.num_atoms
    for i in range(2, 2+na):
        words = lines[i].strip('\n').split()
        coords = np.array([float(x) for x in words[1:4]])
        iatm = offset + i-1
        config.atoms[iatm]['coords'] = np.copy(coords)

#-------------------------------------------------------------------------------

def write_ldf(config, fn, title='', with_pc=True):
    '''
    Writes configuration to a LAMMPS data file (can be imported in Ovito).

    with_pc : bool
        Whether to write pair force field parameters

    '''

    with open(fn,'w') as fh:
        fh.write('#' + title + '\n')

        fh.write('\n')
        fh.write('%d atoms\n'%config.num_atoms)
        fh.write('%d atom types\n'%config.num_atom_types)

        if config.num_bonds > 0:
            fh.write('\n')
            fh.write('%d bonds\n'%config.num_bonds)
            fh.write('%d bond types\n'%config.num_bond_types)

        if config.num_angles > 0:
            fh.write('\n')
            fh.write('%d angles\n'%config.num_angles)
            fh.write('%d angle types\n'%config.num_angle_types)

        if config.num_dihedrals > 0:
            fh.write('\n')
            fh.write('%d dihedrals\n'%config.num_dihedrals)
            fh.write('%d dihedral types\n'%config.num_dihedral_types)

        if config.num_impropers > 0:
            fh.write('\n')
            fh.write('%d impropers\n'%config.num_impropers)
            fh.write('%d improper types\n'%config.num_improper_types)

        fh.write('\n')
        fh.write('%.15g  %.15g  xlo xhi\n'%(config.simbox[0,0], config.simbox[0,1]))
        fh.write('%.15g  %.15g  ylo yhi\n'%(config.simbox[1,0], config.simbox[1,1]))
        fh.write('%.15g  %.15g  zlo zhi\n'%(config.simbox[2,0], config.simbox[2,1]))
        #fh.write('%.15g  %.15g  %.15g  xy xz yz\n'%(config.tilt_factors[0],
        #    config.tilt_factors[1], config.tilt_factors[2])) #Tilt factors

        fh.write('\n')
        fh.write('Masses\n')
        fh.write('\n')
        for iat in range(1,config.num_atom_types+1):
            fh.write('%d  %.8g\n'%(iat, config.atom_mass[iat]))

        if len(config.pair_coeffs) > 0 and with_pc:
            fh.write('\n')
            fh.write('Pair Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.pair_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.pair_coeffs[i])
                fh.write(buf+'\n')

        if len(config.bond_coeffs) > 0 :
            fh.write('\n')
            fh.write('Bond Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.bond_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.bond_coeffs[i])
                fh.write(buf+'\n')

        if len(config.angle_coeffs) > 0 :
            fh.write('\n')
            fh.write('Angle Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.angle_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.angle_coeffs[i])
                fh.write(buf+'\n')

        if len(config.dihedral_coeffs) > 0 :
            fh.write('\n')
            fh.write('Dihedral Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.dihedral_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.dihedral_coeffs[i])
                fh.write(buf+'\n')

        if len(config.improper_coeffs) > 0 :
            fh.write('\n')
            fh.write('Improper Coeffs\n')
            fh.write('\n')
            for i in range(1,len(config.improper_coeffs)+1):
                buf = '%d  '%i + '  '.join(str(x) for x in config.improper_coeffs[i])
                fh.write(buf+'\n')

        fh.write('\n')
        fh.write('Atoms # full\n')
        fh.write('\n')

        #Check if image flags are present (must be present for all atoms or
        #absent for all atoms)
        if 'img_flag' in config.atoms[1]:
            has_img_flag = True
        else :
            has_img_flag = False
        #Write out atom data
        for iatm in range(1, config.num_atoms+1):
            at = config.atoms[iatm]['type']
            chge = config.atoms[iatm]['charge']
            imol = config.atoms[iatm]['imol']
            coords = config.atoms[iatm]['coords']
            buf = '%d  %d  %d  % .15g  '%(iatm, imol, at, chge) \
                + '  '.join( ['% .15g '%x for x in coords] )
            if has_img_flag: 
                img_flag = config.atoms[iatm]['img_flag']
                buf += '  '.join( ['% d'%x for x in img_flag] )
            fh.write(buf+'\n')

        if config.num_bonds > 0:
            fh.write('\n')
            fh.write('Bonds\n')
            fh.write('\n')
            for i in range(1, config.num_bonds+1):
                bt = config.bonds[i]['type']
                atm_i = config.bonds[i]['atm_i']
                atm_j = config.bonds[i]['atm_j']
                buf = '%d  %d  %d  %d\n'%(i, bt, atm_i, atm_j)
                fh.write(buf)

        if config.num_angles > 0:
            fh.write('\n')
            fh.write('Angles\n')
            fh.write('\n')
            for i in range(1, config.num_angles+1):
                ant   = config.angles[i]['type']
                atm_i = config.angles[i]['atm_i']
                atm_j = config.angles[i]['atm_j']
                atm_k = config.angles[i]['atm_k']
                buf = '%d  %d  %d  %d  %d\n'%(i, ant, atm_i, atm_j, atm_k)
                fh.write(buf)

        if config.num_dihedrals > 0:
            fh.write('\n')
            fh.write('Dihedrals\n')
            fh.write('\n')
            for i in range(1, config.num_dihedrals+1):
                dt    = config.dihedrals[i]['type']
                atm_i = config.dihedrals[i]['atm_i']
                atm_j = config.dihedrals[i]['atm_j']
                atm_k = config.dihedrals[i]['atm_k']
                atm_l = config.dihedrals[i]['atm_l']
                buf = '%d  %d  %d  %d  %d  %d\n'%(i, dt, atm_i, atm_j, atm_k, atm_l)
                fh.write(buf)

        if config.num_impropers > 0:
            fh.write('\n')
            fh.write('Impropers\n')
            fh.write('\n')
            for i in range(1, config.num_impropers+1):
                it    = config.impropers[i]['type']
                atm_i = config.impropers[i]['atm_i']
                atm_j = config.impropers[i]['atm_j']
                atm_k = config.impropers[i]['atm_k']
                atm_l = config.impropers[i]['atm_l']
                buf = '%d  %d  %d  %d  %d  %d\n'%(i, it, atm_i, atm_j, atm_k, atm_l)
                fh.write(buf)

        if len(config.velocities) > 0:
            fh.write('\n')
            fh.write('Velocities\n')
            fh.write('\n')
            for i in range(1, len(config.velocities)+1):
                atm_id = i
                v = config.velocities[atm_id]
                buf = '%d  % .15g  % .15g  % .15g\n'%(atm_id, v[0], v[1], v[2])
                fh.write(buf)

#-------------------------------------------------------------------------------

def read_ldf(config, fn):
    '''
    Reads configuration from a LAMMPS data file.

    '''
    atm_toff = config.num_atom_types
    bnd_toff = config.num_bond_types
    ang_toff = config.num_angle_types
    dhd_toff = config.num_dihedral_types
    imp_toff = config.num_improper_types

    atm_idoff = len(config.atoms)
    bnd_idoff = len(config.bonds)
    ang_idoff = len(config.angles)
    dhd_idoff = len(config.dihedrals)
    imp_idoff = len(config.impropers)
    mol_idoff = config.num_molecules

    fh = open(fn,'r')
    #Skip the first line
    line = fh.readline()
    is_header = True #header flag

    while True:
        line = fh.readline()
        #Check EOF
        if not line:
            break
        line = line.strip(' \n')
        #Remove comments 
        m = line.find('#')
        if m != -1:
            line = line[0:m] 
        #Remove blank spaces after removing comment substring
        line = line.strip()
        #Skip blank lines
        if not line:
            continue

        #print(line)
        if line.endswith('atom types'):
            num_atom_types = int(line.split(maxsplit=1)[0])
            for i in range(num_atom_types):
                config.add_atom_type()
        elif line.endswith('bond types'):
            num_bond_types = int(line.split(maxsplit=1)[0])
            for i in range(num_bond_types):
                config.add_bond_type()
        elif line.endswith('angle types'):
            num_angle_types = int(line.split(maxsplit=1)[0])
            for i in range(num_angle_types):
                config.add_angle_type()
        elif line.endswith('dihedral types'):
            num_dihedral_types = int(line.split(maxsplit=1)[0])
            for i in range(num_dihedral_types):
                config.add_dihedral_type()
        elif line.endswith('improper types'):
            num_improper_types = int(line.split(maxsplit=1)[0])
            for i in range(num_improper_types):
                config.add_improper_type()

        elif line.endswith('atoms'):
            num_atoms = int(line.split(maxsplit=1)[0])
        elif line.endswith('bonds'):
            num_bonds = int(line.split(maxsplit=1)[0])
        elif line.endswith('angles'):
            num_angles = int(line.split(maxsplit=1)[0])
        elif line.endswith('dihedrals'):
            num_dihedrals = int(line.split(maxsplit=1)[0])
        elif line.endswith('impropers'):
            num_impropers = int(line.split(maxsplit=1)[0])

        elif line.endswith('xlo xhi'):
            words = line.split()
            config.simbox[0,0] = min(config.simbox[0,0], float(words[0]))
            config.simbox[0,1] = max(config.simbox[0,1], float(words[1]))
        elif line.endswith('ylo yhi'):
            words = line.split()
            config.simbox[1,0] = min(config.simbox[1,0], float(words[0]))
            config.simbox[1,1] = max(config.simbox[1,1], float(words[1]))
        elif line.endswith('zlo zhi'):
            words = line.split()
            config.simbox[2,0] = min(config.simbox[2,0], float(words[0]))
            config.simbox[2,1] = max(config.simbox[2,1], float(words[1]))
        #Tricilinic boxes not considered
        elif line.endswith('xy xz yz'):
            pass
            #config.tilt_factors[0] = float(words[0])
            #config.tilt_factors[1] = float(words[1])
            #config.tilt_factors[2] = float(words[2])

        #Read body section
        elif line == 'Masses':
            fh.readline() #Skip the line following a section header
            for i in range(num_atom_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = atm_toff + int(wrds[0])
                config.set_atom_mass(typ, float(wrds[1]))
                
        elif line == 'Atoms':
            fh.readline() #Skip the line following a section header
            for i in range(num_atoms):
                wrds = fh.readline().rstrip(' \n').split()
                atm_id = atm_idoff + int(wrds[0])
                imol = int(wrds[1])
                if imol !=0:
                    imol += mol_idoff
                typ = atm_toff + int(wrds[2])
                charge = float(wrds[3])
                coords = np.array([float(wrds[4]), float(wrds[5]), float(wrds[6])])
                config.add_atom(typ, charge, coords, imol, atm_id)
                if len(wrds) == 10:
                    img_flag = np.array([int(wrds[7]), int(wrds[8]), int(wrds[9])],
                                        dtype=np.int32)
                    config.add_img_flag(atm_id, img_flag)

        elif line == 'Velocities':
            fh.readline() #Skip the line following a section header
            for i in range(num_atoms):
                wrds = fh.readline().rstrip(' \n').split()
                atm_id = atm_idoff + int(wrds[0])
                vel = np.array([float(wrds[1]), float(wrds[2]), float(wrds[3])])
                config.add_velocity(atm_id, vel)

        elif line == 'Bonds':
            fh.readline() #Skip the line following a section header
            for i in range(num_bonds):
                wrds = fh.readline().rstrip(' \n').split()
                bnd_id = bnd_idoff + int(wrds[0])
                typ = bnd_toff + int(wrds[1])
                config.add_bond(typ, atm_idoff+int(wrds[2]), 
                        atm_idoff+int(wrds[3]), bnd_id)

        elif line == 'Angles':
            fh.readline() #Skip the line following a section header
            for i in range(num_angles):
                wrds = fh.readline().rstrip(' \n').split()
                ang_id = ang_idoff + int(wrds[0])
                typ = ang_toff + int(wrds[1])
                config.add_angle(typ, atm_idoff+int(wrds[2]), 
                    atm_idoff+int(wrds[3]), atm_idoff+int(wrds[4]), ang_id)

        elif line == 'Dihedrals':
            fh.readline() #Skip the line following a section header
            for i in range(num_dihedrals):
                wrds = fh.readline().rstrip(' \n').split()
                dhd_id = dhd_idoff + int(wrds[0])
                typ = dhd_toff + int(wrds[1])
                config.add_dihedral(typ, atm_idoff+int(wrds[2]), 
                    atm_idoff+int(wrds[3]), atm_idoff+int(wrds[4]),
                    atm_idoff+int(wrds[5]), dhd_id)

        elif line == 'Impropers':
            fh.readline() #Skip the line following a section header
            for i in range(num_impropers):
                wrds = fh.readline().rstrip(' \n').split()
                imp_id = imp_idoff + int(wrds[0])
                typ = imp_toff + int(wrds[1])
                config.add_improper(typ, atm_idoff+int(wrds[2]), 
                    atm_idoff+int(wrds[3]), atm_idoff+int(wrds[4]), 
                    atm_idoff+int(wrds[5]), imp_id)

        elif line == 'Pair Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_atom_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = atm_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_pair_coeff(typ, coeffs)

        elif line == 'Bond Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_bond_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = bnd_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_bond_coeff(typ, coeffs)

        elif line == 'Angle Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_angle_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = ang_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_angle_coeff(typ, coeffs)
            
        elif line == 'Dihedral Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_dihedral_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = dhd_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_dihedral_coeff(typ, coeffs)
            
        elif line == 'Improper Coeffs':
            fh.readline() #Skip the line following a section header
            for i in range(num_improper_types):
                wrds = fh.readline().rstrip(' \n').split()
                typ = imp_toff + int(wrds[0])
                coeffs = []
                for each in wrds[1:]:
                    try:
                        coeffs.append(float(each))
                    except ValueError:
                        coeffs.append(each)
                config.set_improper_coeff(typ, coeffs)
        else:
            raise IOError('Unrecognized header line: "%s"'%line)
            
    fh.close()

#-------------------------------------------------------------------------------

def ldf_to_xyz(fn_ldf, fn_xyz, atom_names=None):
    """
    Converts a LAMMPS data file to XYZ file.

    Parameters
    ---------
    fn_ldf : str or pathlib.Path
        Name of the LAMMPS data file to read.
    fn_xyz : str or pathlib.Path
        Name of the XYZ file to write to.
    atom_names : list of tuples
        Mapping of atom type to atom names. E.g. [(1, 'H'), (2, 'Se'), ...].
        If None, the default name is `Xi`, where `i` is the corresponding atom
        type.

    Returns
    -------
    None

    """
    config = Configuration()
    read_ldf(config, fn_ldf)
    config.set_atom_names(atoms_names)
    write_xyz(config, fn_xyz, title='')


#-------------------------------------------------------------------------------

def extract_velocities(fn, fn_out=None):
    '''
    Reads in atom velocities from a LAMMPS data file and optionally writes to a
    file.

    fn : LAMMPS data file
    fn_out : Output file containing atom velocities

    '''
    fh = open(fn,'r')
    #Skip the first line
    line = fh.readline()
    found_vel = False #Is there a velocities section?

    while True:
        line = fh.readline()
        #Check EOF
        if not line:
            break
        line = line.strip(' \n')
        #Remove comments 
        m = line.find('#')
        if m != -1:
            line = line[0:m] 
        #Remove blank spaces after removing comment substring
        line = line.strip()
        #Skip blank lines
        if not line:
            continue

        if line.endswith('atoms'):
            num_atoms = int(line.split(maxsplit=1)[0])
            velocities = np.zeros((num_atoms,3))

        elif line == 'Velocities':
            found_vel = True
            fh.readline() #Skip the line following a section header
            for i in range(num_atoms):
                wrds = fh.readline().rstrip(' \n').split()
                atm_id = int(wrds[0])
                velocities[atm_id-1] = np.array([float(wrds[1]), 
                        float(wrds[2]), float(wrds[3])])
            break

    fh.close()

    if not found_vel:
        print("No velocity data in input file.")
        return None

    if fn_out is not None: 
        with open(fn_out, 'w') as fh_out:
            fh_out.write('%d\n'%num_atoms)
            for i in range(num_atoms):
                v = velocities[i,:]
                buf = '%d % .15g % .15g % .15g\n'%(i+1, v[0], v[1], v[2])
                fh_out.write(buf)

    return velocities

#-------------------------------------------------------------------------------

def add_molecules(config, molecules, packmol_tol=2.0, packmol_sidemax=1.0e3,
                  packmol_path=''):
    """
    Adds `num` molecules following the configuration template `moltem` (an
    instance of Configuration)

    molecules : list
        List of molecules. 

        Each element of `molecules` is a dict with keys 'moltem', 'num',
        'constraints'. `moltem` is an instance of `Configuration` or its
        subclass. `num` is an integer specifying the number of molecules of this
        type to be added. `offsets` is a tuple of five integers specifying the
        offsets of atom types, bond types, angle types, dihedral types, and
        improper types, repectively. `constaints` is a list of strings
        specifying the constrains as required by packmol,
        e.g., `constraints = ['inside cube xmin ymin zmin d',
        'outside sphere a b c d', ...]`.
        An empty constaint list indicates inside the entire simulation box.

        An example of `molecules` may be 
        [{'moltem': acetone, 'num': 20, `offsets`: [0,0,0,0,0], 
        'constraints': ['inside cube x y z d', 
            'outside box xmin ymin zmin xmax ymax zmax', ... ]},
         {'moltem': glycol, 'num': 10,}, `offsets`: [2,4,2,2,0],
         'constraints': ['outside cube x y z d',
            'inside sphere a b c d', ...]},
        ...]
    packmol_tol : float
        Tolerance for Packmol. Default is 2 angstrom.
    packmol_sidemax : float
        Parameter for Packmol. Default is 1000 angstrom.
    packmol_path : str or pathlib.Path or None
        Path to the packmol executable. If None, Packmol will not be used. In
        this case all added molecules will have their atom positions set to
        zero.

    """
    na_ini = config.num_atoms #Number of atoms before adding molecules

    for each in molecules:
        moltem = each['moltem']
        #Offsets for atom types, etc.
        atm_toff = config.num_atom_types
        bnd_toff = config.num_bond_types
        ang_toff = config.num_angle_types
        dhd_toff = config.num_dihedral_types
        imp_toff = config.num_improper_types

        #Add atom type, bond type, etc.
        if each['offsets'][0]==atm_toff:
            for i in range(1,moltem.num_atom_types+1):
                config.add_atom_type(mass=moltem.atom_mass[i])
                typ = each['offsets'][0] + i
                config.set_pair_coeff(typ, moltem.pair_coeffs[i])
        if each['offsets'][1]==bnd_toff:
            for i in range(1,moltem.num_bond_types+1):
                config.add_bond_type(moltem.bond_coeffs[i])
        if each['offsets'][2]==ang_toff:
            for i in range(1,moltem.num_angle_types+1):
                config.add_angle_type(moltem.angle_coeffs[i])
        if each['offsets'][3]==dhd_toff:
            for i in range(1,moltem.num_dihedral_types+1):
                config.add_dihedral_type(moltem.dihedral_coeffs[i])
        if each['offsets'][4]==imp_toff:
            for i in range(1,moltem.num_improper_types+1):
                config.add_improper_type(moltem.improper_coeffs[i])

        #Add atoms, bonds, etc.
        for jmol in range(each['num']):
            #Offsets for atom ids, etc.
            atm_idoff = config.num_atoms
            bnd_idoff = config.num_bonds
            ang_idoff = config.num_angles
            dhd_idoff = config.num_dihedrals
            imp_idoff = config.num_impropers
            mol_idoff = config.num_molecules

            #Positions of atoms of moltem are all set to zero. This will be
            #modified later by packmol.
            for i in range(1, moltem.num_atoms+1):
                atm_id = atm_idoff + i
                mol_id = mol_idoff + 1
                typ = each['offsets'][0] + moltem.atoms[i]['type']
                charge = moltem.atoms[i]['charge']
                coords = np.zeros((3,))
                config.add_atom(typ, charge, coords, mol_id, atm_id)

            for i in range(1, moltem.num_bonds+1):
                bnd_id = bnd_idoff + i
                typ = each['offsets'][1] + moltem.bonds[i]['type']
                atom_i = atm_idoff + moltem.bonds[i]['atm_i']
                atom_j = atm_idoff + moltem.bonds[i]['atm_j']
                config.add_bond(typ, atom_i, atom_j, bnd_id)

            for i in range(1, moltem.num_angles+1):
                ang_id = ang_idoff + i 
                typ = each['offsets'][2] + moltem.angles[i]['type']
                atom_i = atm_idoff + moltem.angles[i]['atm_i']
                atom_j = atm_idoff + moltem.angles[i]['atm_j']
                atom_k = atm_idoff + moltem.angles[i]['atm_k']
                config.add_angle(typ, atom_i, atom_j, atom_k, ang_id)

            for i in range(1, moltem.num_dihedrals+1):
                dhd_id = dhd_idoff + i
                typ = each['offsets'][3] + moltem.dihedrals[i]['type']
                atom_i = atm_idoff + moltem.dihedrals[i]['atm_i']
                atom_j = atm_idoff + moltem.dihedrals[i]['atm_j']
                atom_k = atm_idoff + moltem.dihedrals[i]['atm_k']
                atom_l = atm_idoff + moltem.dihedrals[i]['atm_l']
                config.add_dihedral(typ, atom_i, atom_j, atom_k, atom_l, dhd_id)

            for i in range(1, moltem.num_impropers+1):
                imp_id = imp_idoff + i
                typ = each['offsets'][4] + moltem.impropers[i]['type']
                atom_i = atm_idoff + moltem.impropers[i]['atm_i']
                atom_j = atm_idoff + moltem.impropers[i]['atm_j']
                atom_k = atm_idoff + moltem.impropers[i]['atm_k']
                atom_l = atm_idoff + moltem.impropers[i]['atm_l']
                config.add_improper(typ, atom_i, atom_j, atom_k, atom_l, imp_id)
            #Update molecule population
            config.molecules[mol_id] = {'name': moltem.name, 'atm_beg': atm_idoff+1, 
                                     'atm_end': config.num_atoms}

    #Update atom positions with packmol
    #Write out packmol file
    if packmol_path is None:
        return
    fn_pm_in = Path('inp_pm.txt') #Packmol input file
    fn_pm_out = Path('out_pm.xyz')
    fns_xyz = []
    with open(fn_pm_in, 'w') as fh:
        fh.write('tolerance %g\n'%packmol_tol)
        fh.write('sidemax %g\n'%packmol_sidemax)
        fh.write('seed -1\n')
        fh.write('randominitialpoint\n')
        fh.write('movebadrandom yes\n')
        fh.write('output %s\n'%fn_pm_out)
        fh.write('filetype xyz\n')
        fh.write('\n')
        for each in molecules:
            if each['num'] < 1:
                continue
            moltem = each['moltem']
            fn_xyz = Path('_tmp_'+ moltem.name + '.xyz')
            fns_xyz.append(fn_xyz)
            write_xyz(moltem, fn_xyz)
            fh.write('structure %s\n'%fn_xyz)
            fh.write('  number %d\n'%each['num'])
            for constraint in each['constraints']:
                fh.write('  %s\n'%constraint)
            fh.write('end structure\n')
    
    #Run packmol
    args_run = ["%s < %s"%(packmol_path, fn_pm_in)]
    subprocess.run(args_run, shell=True)
    #Read back the packmol output
    read_xyz(config, fn_pm_out, offset=na_ini)

    fn_pm_in.unlink(missing_ok=True)
    fn_pm_out.unlink(missing_ok=True)
    Path(str(fn_pm_out)+'_FORCED').unlink(missing_ok=True)
    for each in fns_xyz:
        each.unlink(missing_ok=True)


def write_mol_grp(config, fn, title=''):
    """
    Write molecule and group records.

    """
    with open(fn, 'w') as fh:
        fh.write('%s\n'%title)
        fh.write('GROUPS %d\n'%config.num_groups)
        for key,val in config.groups.items():
            fh.write(f"{key}")
            for k, v in val.items():
                fh.write(f" {k}")
                if isinstance(v, range):
                    #fh.write(f" range {v[0]} {v[-1]} {v.step}")
                    fh.write(f" {v[0]}:{v[-1]}:{v.step}")
                else:
                    #fh.write(f" list {len(v)} {' '.join(str(x) for x in v)}")
                    fh.write(f" {len(v)} {' '.join(str(x) for x in v)}")
            fh.write('\n')
        fh.write('\n')
        fh.write('MOLECULES %d\n'%config.num_molecules)
        for key,val in config.molecules.items():
            fh.write(f"  {key} {val['name']} {val['atm_beg']} {val['atm_end']}\n")



def read_mol_grp(config, fn):
    """
    Reads molecule and group records from a file.

    """
    with open(fn, 'r') as fh:
        fh.readline() #Skip title line
        num_groups = int( fh.readline().strip(' \n').split()[1] )
        for i in range(num_groups):
            words = fh.readline().strip(' \n').split()
            num_words = len(words)
            gname = words[0]
            atom_types=None; atoms=None; molecules=None
            indx = 1
            while indx < num_words:
                key = words[indx]
                if ':' in words[indx+1]:
                    fields = [int(x) for x in words[indx+1].split(':')]
                    values = range(fields[0], fields[1]+1, fields[2])
                    indx += 2
                else:
                    n = int(words[indx+1])
                    values = [int(x) for x in words[indx+2:indx+2+n]] 
                    indx += (n+2)
                if key == 'atom_types':
                    atom_types = values
                elif key == 'atoms':
                    atoms = values
                elif key == 'molecules':
                    molecules = values

            config.set_group(gname, atom_types=atom_types, atoms=atoms,
                             molecules=molecules)

        fh.readline() #Skip blank line
        num_molecules = int( fh.readline().strip(' \n').split()[1] )
        for i in range(num_molecules):
            words = fh.readline().strip(' \n').split()
            config.molecules[i+1] = {'name': words[1], 'atm_beg': int(words[2]),
                                     'atm_end': int(words[3])}
