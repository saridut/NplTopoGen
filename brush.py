#!/usr/bin/env python

"""
Class implementing a single NPL crystal.

"""

import warnings
import os
import copy
import math
import numpy as np
from crystal import get_lattice_points
from _configuration import Configuration
from _geom_utils import rotate_vector_axis_angle
from _config_io import read_ldf, write_ldf, write_xyz, write_mol_grp, \
        read_mol_grp, add_molecules


class Brush(Configuration):
    def __init__(self, lx, lz):
        """
        Parameters
        ----------
        lx : float
            Dimension of the simulation box along the x-axis, set equal to the
            dimension along the y-axis. The box bounds are [-lx/2, lx/2]
            along both x & y.
        lz : float
            Dimension of the simulation box along the z-axis. The brush is
            oriented along this direction. Box bounds are [0,lz].

        """
        super().__init__()

        self.xtal = None
        self.lx = lx
        self.lz = lz
        self.gdens = 0.0

#       self._num_atoms_ligands = 0
#       self._num_atom_types_ligands = 0
#       self._num_molecules_ligands = 0
        
        self.add_simbox(-lx/2, lx/2, -lx/2, lx/2, 0, lz)



    def add_ligands(self, ligand_list, ligand_pop_ratio, gdens, teth_dist,
                    lattice, packmol_tol=2.0, packmol_sidemax=1.0e3,
                    packmol_path=''):
        """
        Adds ligands to a bare nanoplatelet.

        Parameters
        ----------

        ligand_list : list of LigandMolecule
            Ligand molecules to add.
        ligand_pop_ratio : list of int
            In case of multiple ligand molecules, the ratio of ligand population
            for each type. Only considers monovalent ligands. E.g., for two
            ligand types in 2:1 ratio, use [2, 1].
        gdens : float
            Nominal grafting number density of ligands per angstrom^2.
        teth_dist : float
            Distance along the z-axis from the bottom of the box to the
            ligand head atoms (the C1 atom).
        lattice : {'sc', 'bcc'}
            Lattice for arranging the ligand molecule graft points
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_sidemax : float
            Parameter for Packmol. Default is 1000 angstrom.
        packmol_path : str or pathlib.Path
            Path to Packmol binary.

        """
        rng = np.random.default_rng()
        num_ligand_types = len(ligand_list)
        lig_pop = np.zeros((num_ligand_types,), dtype=np.int32)

        #Ligand population
        n = gdens*self.lx**2 #Number of ligands from nominal grafting density

        if lattice == 'sc':
            lpar = self.lx/n**0.5
        elif lattice == 'bcc':
            lpar = self.lx/(n/2)**0.5

        #Graft points
        lo = [-self.lx/2, -self.lx/2, teth_dist]
        hi = [ self.lx/2,  self.lx/2, teth_dist]
        graft_points = get_lattice_points(lattice, lpar, lo, hi, 'ppp')
        for each in graft_points:
            each[2] += abs( rng.normal(0.0, 1.0) )
        n = len(graft_points)

        den = ligand_pop_ratio.sum()
        lig_pop = np.rint(n*ligand_pop_ratio/den).astype(np.int32)
        diff = n - lig_pop.sum()

        if diff > 0:
            lig_pop[-1] += diff
        elif diff < 0:
            lig_pop[-1] -= diff
        assert n <= len(graft_points)

        self.gdens = n/self.lx**2

        #n = int( math.ceil(n**0.5)**2 ) #Rounding to nearest squared integer

        aid_beg = self.num_atoms + 1 # First ligand atom id
        aid_end = aid_beg  # Last ligand atom id
        mid_beg = self.num_molecules + 1 #First ligand molecule id
        mid_end = mid_beg  # Last ligand molecule id
        lig_size_max = 0   # Planar size
        for i in range(num_ligand_types):
            ligtem = ligand_list[i]
            pop = lig_pop[i]
            if pop < 1:
                s = input(f"Zero ligands {ligtem.name} ligands. Type `Y` to "
                          f"continue, any other key to exit: ")
                if s != "Y":
                    raise SystemExit("Exiting ...")
            aid_end += ( pop*ligtem.num_atoms - 1 )
            mid_end += (pop - 1)
            print(f"  Number of {ligtem.name} ligand molecules = {pop}")
            ligtem.translate_atom(ligtem.head, np.zeros((3,)))
            coords = ligtem.get_atom_coords()
            lig_size = 2*np.hypot(coords[:,0], coords[:,1]).max()
            lig_size_max = max(lig_size_max, lig_size)

        na_ligands = aid_end - aid_beg + 1
        print('  Total number of ligand molecules = %d'%sum(lig_pop)) 
        print('  Total number of ligand atoms = %d'%na_ligands)
        print('  Maximal ligand planar size = %g'%lig_size_max)

        #Assign types to each ligand
        ligand_types = np.arange(0, n, dtype=np.int32)
        indices = list(range(n))
        for i in range(num_ligand_types):
            indx = rng.choice(indices, size=lig_pop[i], replace=False,
                        shuffle=False)
            for j in indx:
                ligand_types[j] = i
                indices.remove(j)

        #Offsets in types for ligands
        num_types = np.array([self.num_atom_types, self.num_bond_types,
                        self.num_angle_types, self.num_dihedral_types,
                        self.num_improper_types], dtype=np.int32)
        type_offsets = []
        at_beg = self.num_atom_types + 1
        at_end = at_beg
        for each in ligand_list:
            type_offsets.append( tuple(num_types) )
            new_types = np.array([each.num_atom_types, each.num_bond_types,
                    each.num_angle_types, each.num_dihedral_types,
                    each.num_improper_types], dtype=np.int32)
            num_types += new_types
            at_end += (each.num_atom_types - 1)
        
        #Add ligands
        ligands_to_add = []
        for i in range(num_ligand_types):
            elem = {'moltem': ligand_list[i], 'num': lig_pop[i], 
                    'offsets': type_offsets[i]}
            ligands_to_add.append(elem)
        add_molecules(self, ligands_to_add, packmol_path=None)

        #self._num_atoms_ligands = aid_end - aid_beg + 1
        #self._num_atom_types_ligands = at_end - at_beg + 1
        #self._num_molecules_ligands = mid_end - mid_beg + 1

        #Create new group of ligand atoms
        atoms = range(aid_beg, aid_end+1)
        atom_types = range(at_beg, at_end+1)
        molecules = range(mid_beg, mid_end+1)
        self.set_group('ligands', atoms=atoms, atom_types=atom_types,
                       molecules=molecules)

        #Graft ligands
        #Adding grafting point atoms
        #~aid_beg = self.num_atoms + 1
        #~aid_end = aid_beg + len(graft_points) - 1
        #~iat = self.add_atom_type(mass=1.0, name='GP')
        #~eps = 0.0; sigma = 1.0
        #~self.set_pair_coeff(iat, [eps, sigma])
        #~at_beg = iat; at_end = iat

        #~for each in graft_points:
        #~    self.add_atom(iat, 0.0, each)

        #~atoms = range(aid_beg, aid_end+1)
        #~atom_types = range(at_beg, at_end+1)
        #~self.set_group('GraftPoints', atom_types=atom_types, atoms=atoms)

        #Adding bond type: graft points -- ligand heads
        #~bt_gp = self.add_bond_type(params=[60.0, teth_dist])

        #Grafting ligands
        zhat = np.array([0,0,1])
        for k in range(n):
            gpos = graft_points[k].copy()
            #!gpos[2] += 2.0
            ltyp = ligand_types[k]
            moltem = ligand_list[ltyp]
            angle = 2*math.pi*rng.random()
            p = moltem.get_atom_coords([moltem.head])
            moltem.rotate(angle, zhat, p[0:])
            moltem.translate_atom(moltem.head, gpos)

            mol_id = self.groups['ligands']['molecules'][k]
            atm_beg = self.molecules[mol_id]['atm_beg']
            atm_end = self.molecules[mol_id]['atm_end']
            coords = moltem.get_atom_coords()
            #~#Move ligands adjacent to the wall just beyond cutoff
            #~zmin = coords[:,2].min()
            #~dz = 2.0 - zmin #wall cutoff = 2.0 (move wall below)
            #~coords[:,2] += dz
            self.set_atom_coords(range(atm_beg, atm_end+1), coords)
            #Move ligands upwards to bring head to eql dist from graft point
            #~coords[:,2] += teth_dist
            #~self.set_atom_coords(range(atm_beg, atm_end+1), coords)

            #~atm_i = self.groups['GraftPoints']['atoms'][k]
            #~atm_j = atm_beg + moltem.head - 1
            #~self.add_bond(bt_gp, atm_i, atm_j)




    def add_ligand_one(self, ligand, teth_dist):
        """
        Adds a single ligand molecule at the center of the tethering plane.

        Parameters
        ----------

        ligand : LigandMolecule
            Ligand molecule to add.
        teth_dist : float
            Distance along the z-axis from the bottom of the box to the
            ligand head atoms (the C1 atom).
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_sidemax : float
            Parameter for Packmol. Default is 1000 angstrom.
        packmol_path : str or pathlib.Path
            Path to Packmol binary.

        """
        rng = np.random.default_rng()

        #Graft points
        graft_point = np.array([0,0,teth_dist])
        self.gdens = 1/self.lx**2

        aid_beg = self.num_atoms + 1 # First ligand atom id
        aid_end = aid_beg + ligand.num_atoms - 1 # Last ligand atom id
        mid_beg = self.num_molecules + 1 #First ligand molecule id
        mid_end = mid_beg  # Last ligand molecule id

        na_ligands = aid_end - aid_beg + 1
        print('  Total number of ligand molecules = 1') 
        print('  Total number of ligand atoms = %d'%na_ligands)

        #Offsets in types
        num_types = np.array([self.num_atom_types, self.num_bond_types,
                        self.num_angle_types, self.num_dihedral_types,
                        self.num_improper_types], dtype=np.int32)
        type_offsets = tuple(num_types)
        at_beg = self.num_atom_types + 1
        at_end = at_beg + ligand.num_atom_types - 1

        #Add ligands
        ligands_to_add = [{'moltem': ligand, 'num': 1, 'offsets': type_offsets}]
        add_molecules(self, ligands_to_add, packmol_path=None)

        #Create new group of ligand atoms
        atoms = range(aid_beg, aid_end+1)
        atom_types = range(at_beg, at_end+1)
        molecules = range(mid_beg, mid_end+1)
        self.set_group('ligands', atoms=atoms, atom_types=atom_types,
                       molecules=molecules)

        #Grafting ligands
        zhat = np.array([0,0,1])
        angle = 2*math.pi*rng.random()
        p = ligand.get_atom_coords([ligand.head])
        ligand.rotate(angle, zhat, p[0:])
        ligand.translate_atom(ligand.head, graft_point)

        mol_id = self.groups['ligands']['molecules'][0]
        atm_beg = self.molecules[mol_id]['atm_beg']
        atm_end = self.molecules[mol_id]['atm_end']
        coords = ligand.get_atom_coords()
        self.set_atom_coords(range(atm_beg, atm_end+1), coords)



    def solvate(self, solvent, exclude, packmol_tol=2.0,
                packmol_sidemax=1.0e3, packmol_path=''):
        """
        Adds solvent molecules around a nanoplatelet.

        Parameters
        ----------
        solvent : tuple
            Solvent molecule to add. The tuple is (Molecule, density), where density
            is in g/mL.
        exclude : tuple
            exclude[0] and exclude[1], where exclude[1] >= exclude[0], marks a
            slab normal to the z-direction devoid of solvent molecules.
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_sidemax : float
            Parameter for Packmol. Default is 1000 angstrom.
        packmol_path : str or pathlib.Path
            Path to Packmol binary.

        """
        delta = 2.0 #Small gap between periodic images (See Packmol manual)

        #Population of molecules for each component
        volume = self.lx*self.lx*self.lz
        molwt = solvent[0].get_total_mass() # in g/mol
        dens = solvent[1] #Density in g/mL (= g/cc)
        nonsolvent_mass = 0.0
        if 'ligands' in self.groups:
            nonsolvent_atoms = self.groups['ligands']['atoms']
            for each in nonsolvent_atoms:
                at = self.atoms[each]['type']
                nonsolvent_mass += self.atom_mass[at]
        pop = (0.6023*volume*dens - nonsolvent_mass)/molwt
        pop = math.floor(pop)

        aid_beg = self.num_atoms + 1 #First solvent atom id
        aid_end = aid_beg + pop*solvent[0].num_atoms - 1 #Last solvent atom id
        mid_beg = self.num_molecules + 1 #First solvent molecule id
        mid_end = mid_beg + pop - 1#Last solvent molecule id

        na_solvent = aid_end - aid_beg + 1

        print(f"Number of solvent molecules = {pop}")
        print(f"Total number of solvent atoms = {na_solvent}")

        #Offsets in types
        type_offsets = tuple([self.num_atom_types, self.num_bond_types,
                        self.num_angle_types, self.num_dihedral_types,
                        self.num_improper_types])
        at_beg = self.num_atom_types + 1
        at_end = at_beg + solvent[0].num_atom_types - 1

        #Define bounding boxes & add molecules
        oa_bbox_lo = self.simbox[:,0] + delta #Overall bounding box
        oa_bbox_hi = self.simbox[:,1] - delta #Overall bounding box

        if exclude[1] > exclude[0]:
            #Two sub bounding boxes for solvent molecules
            dz_top = self.simbox[2,1] - exclude[1]
            dz_bot = exclude[0] - self.simbox[2,0]
            pop_top = int( pop*dz_top/(dz_top+dz_bot) )
            pop_bot = pop - pop_top

            top_bbox_hi = oa_bbox_hi
            top_bbox_lo = [ oa_bbox_lo[0], oa_bbox_lo[1], exclude[1] ]

            bot_bbox_lo = oa_bbox_lo
            bot_bbox_hi = [ oa_bbox_hi[0], oa_bbox_hi[1], exclude[0] ]

            print(f"Packing solvents \n"
                  f" inside box ({' '.join('%g'%v for v in top_bbox_lo)})"
                  f" ({' '.join('%g'%v for v in top_bbox_hi)}) \n"
                  f" inside box ({' '.join('%g'%v for v in bot_bbox_lo)})"
                  f" ({' '.join('%g'%v for v in bot_bbox_hi)})"
                  )
            g_top = 'inside box ' \
                + ' '.join(['%g'%x for x in top_bbox_lo]) + ' ' \
                + ' '.join(['%g'%x for x in top_bbox_hi])
            elem = {'moltem': solvent[0], 'num': pop_top, 'offsets': type_offsets,
                    'constraints': [g_top]}
            mols_to_add = [elem]

            g_bot = 'inside box ' \
                + ' '.join(['%g'%x for x in bot_bbox_lo]) + ' ' \
                + ' '.join(['%g'%x for x in bot_bbox_hi])
            elem = {'moltem': solvent[0], 'num': pop_bot, 'offsets': type_offsets,
                    'constraints': [g_bot]}
            mols_to_add.append(elem)
        else:
            #A single bounding box for solvent molecules
            print(f"Packing solvents \n"
                  f" inside box ({' '.join('%g'%v for v in oa_bbox_lo)})"
                  f" ({' '.join('%g'%v for v in oa_bbox_hi)})"
                  )
            g = 'inside box ' \
                + ' '.join(['%g'%x for x in oa_bbox_lo]) + ' ' \
                + ' '.join(['%g'%x for x in oa_bbox_hi])
            elem = {'moltem': solvent[0], 'num': pop, 'offsets': type_offsets,
                    'constraints': [g]}
            mols_to_add = [elem]

        add_molecules(self, mols_to_add, packmol_tol, packmol_sidemax,
                      packmol_path)

        #Create new group of solvent atoms
        atoms = range(aid_beg, aid_end+1)
        atom_types = range(at_beg, at_end+1)
        molecules = range(mid_beg, mid_end+1)
        self.set_group('solvent', atoms=atoms, atom_types=atom_types,
                       molecules=molecules)



    def add_piston(self, loc, lpar, thickness, atom_mass, eps, sigma, rcut=None):
        """
        Adds a group of atoms which will act as a piston at the top and bottom
        ends of the simulation box along the z-direction. The simulation box
        dimensions will be enlarged on adding the piston atoms.

        Parameters
        ----------
        loc : {'top', 'bottom', 'both'}
            Location of the piston atoms along the z-direction
        lpar : float
            Lattice constant of the fcc lattice on which the piston atoms
            will be arranged.
        thickness : float
            Thickness of the piston
        atom_mass : float
            Mass of a piston atom
        eps : float
            LJ energy scale
        sigma : float
            LJ length scale
        rcut : float
            LJ interaction cutoff

        """
        delta = 1.0
        if loc == 'top' or loc == 'both':
            lo = [ -self.lx/2, -self.lx/2, self.lz ]
            hi = [  self.lx/2,  self.lx/2, self.lz+thickness ]
            coords = get_lattice_points('fcc', lpar, lo, hi, 'ppc')
            num_surface_atoms = np.count_nonzero(coords[:,2]==hi[2])
            print(f"Top piston: {num_surface_atoms} surface atoms")

            aid_beg = self.num_atoms + 1
            aid_end = aid_beg + len(coords) - 1
            iat = self.add_atom_type(mass=atom_mass, name='PT')
            if rcut is None:
                self.set_pair_coeff(iat, [eps, sigma])
            else:
                self.set_pair_coeff(iat, [eps, sigma, rcut])
            at_beg = iat; at_end = iat

            for each in coords:
                self.add_atom(iat, 0.0, each)

            self.simbox[2,1] += (thickness+delta)

            atoms = range(aid_beg, aid_end+1)
            atom_types = range(at_beg, at_end+1)
            self.set_group('PistonTop', atom_types=atom_types, atoms=atoms)

        if loc == 'bottom' or loc == 'both':
            lo = [ -self.lx/2, -self.lx/2, -thickness ]
            hi = [  self.lx/2,  self.lx/2, 0.0 ]
            coords = get_lattice_points('fcc', lpar, lo, hi, 'ppc')
            num_surface_atoms = np.count_nonzero(coords[:,2]==lo[2])
            print(f"Bottom piston: {num_surface_atoms} surface atoms")

            aid_beg = self.num_atoms + 1
            aid_end = aid_beg + len(coords) - 1
            iat = self.add_atom_type(mass=atom_mass, name='PB')
            if rcut is None:
                self.set_pair_coeff(iat, [eps, sigma])
            else:
                self.set_pair_coeff(iat, [eps, sigma, rcut])
            at_beg = iat; at_end = iat

            for each in coords:
                self.add_atom(iat, 0.0, each)

            self.simbox[2,0] -= (thickness+delta)

            atoms = range(aid_beg, aid_end+1)
            atom_types = range(at_beg, at_end+1)
            self.set_group('PistonBot', atom_types=atom_types, atoms=atoms)



    def adjust_charge(self):
        """
        Tweak charges to make the system electroneutral.

        """
        chge = self.get_total_charge()
        if chge != 0:
            num_atoms_tot = self.num_atoms
            if 'GraftPoints' in self.groups:
                num_atoms_tot -= len(self.groups['GraftPoints']['atoms'])
            cpa = chge/num_atoms_tot
            for i in range(1, self.num_atoms+1):
                self.atoms[i]['charge'] -= cpa
            if 'GraftPoints' in self.groups:
                for i in self.groups['GraftPoints']['atoms']:
                    self.atoms[i]['charge'] = 0.0
        print("Total charge after adjustment = %g"%self.get_total_charge())



    def write_bond_coeffs(self, fn, kbond=None):
        """
        Writes bond coefficients to a file.

        """
        if len(self.bond_coeffs) == 0:
            print("No bond coeffs present. Exiting ...")
            return

        with open(fn, 'w') as fh:
            fh.write('#Bond Coeffs\n')
            fh.write('\n')
            for i in range(1,len(self.bond_coeffs)+1):
                coeffs = self.bond_coeffs[i].copy()
                if kbond:
                    coeffs[0] = kbond
                buf = 'bond_coeff %d  '%i + '  '.join(str(x) for x in coeffs)
                fh.write(buf+'\n')


    def gen_ff_pair(self, fn, soften=None):
        """
        Writes pair interaction coefficients to a file.

        soften : {'ligands', 'solvent', 'both'}
            Soften potential for this group.

        """
        if soften is None:
            with open(fn, 'w') as fh:
                #pair_style lj/cut/coul/long 12.0
                #pair_modify shift yes mix geometric
                #special_bonds lj/coul 0.0 0.0 0.5 angle yes dihedral yes
                for i in range(1, self.num_atom_types+1):
                    buf = f"pair_coeff {i} {i} "
                    buf += ' '.join(str(x) for x in self.pair_coeffs[i])
                    fh.write(buf+"\n")

                at_piston = []; at_non_piston = []
                if 'PistonTop' in self.groups:
                    at_piston += list(self.groups['PistonTop']['atom_types'])
                if 'PistonBot' in self.groups:
                    at_piston += list(self.groups['PistonBot']['atom_types'])
                #for i in range(1, self.num_atom_types+1):
                #    if i not in at_piston:
                #        at_non_piston.append(i)

                #for i in at_piston:
                #    for j in range(1, self.num_atom_types+1):
                #        coeffs_i = self.pair_coeffs[i]
                #        coeffs_j = self.pair_coeffs[j]
                #        eps_i = coeffs_i[0]; sigma_i = coeffs_i[1]
                #        eps_j = coeffs_j[0]; sigma_j = coeffs_j[1]
                #        eps_ij = math.sqrt(eps_i*eps_j)
                #        sigma_ij = math.sqrt(sigma_i*sigma_j)
                #        rcut_ij =  sigma_ij*2**(1/6)
                #        if i < j:
                #            fh.write(f"pair_coeff {i} {j} {eps_ij:g}"
                #                 f" {sigma_ij:g} {rcut_ij:g} \n")
                #        elif i > j:
                #            fh.write(f"pair_coeff {j} {i} {eps_ij:g}"
                #                 f" {sigma_ij:g} {rcut_ij:g} \n")

        else:
            if soften=='ligands' or soften=='solvent':
                soft_atom_types = self.groups[soften]['atom_types']
            if soften=='both':
                soft_atom_types = list(self.groups['ligands']['atom_types']) \
                                + list(self.groups['solvent']['atom_types'])
            with open(fn, 'w') as fh:
                #pair_style lj/cut/coul/long/soft 2 0.5 10.0 12.0
                #pair_modify shift yes
                #special_bonds lj/coul 0.0 0.0 0.5 angle yes dihedral yes
                #Pair coeffs for self types (I == J)
                for i in range(1, self.num_atom_types+1):
                    lamda = 0.0 if i in soft_atom_types else 1.0
                    coeffs = self.pair_coeffs[i]
                    eps = coeffs[0]; sigma = coeffs[1]
                    #cutoffs = ' '.join(str(x) for x in coeffs[2:])
                    fh.write(f"pair_coeff {i} {i} {eps} {sigma} {lamda}\n")

                #Pair coeffs for cross types (I <= J) using geometric mixing rule
                #Ignores explicit cutoff, will use the global cutoff
                lamda = 1.0
                for i in range(1, self.num_atom_types):
                    for j in range(i+1, self.num_atom_types+1):
                        if (i in soft_atom_types) and (j in soft_atom_types):
                            lamda = 0.0
                        else:
                            lamda = 1.0
                        coeffs_i = self.pair_coeffs[i]
                        coeffs_j = self.pair_coeffs[j]
                        eps_i = coeffs_i[0]; sigma_i = coeffs_i[1]
                        eps_j = coeffs_j[0]; sigma_j = coeffs_j[1]
                        eps_ij = math.sqrt(eps_i*eps_j)
                        sigma_ij = math.sqrt(sigma_i*sigma_j)
                        fh.write(f"pair_coeff {i} {j} {eps_ij:g}"
                                 f" {sigma_ij:g} {lamda} \n")



    def write(self, fn='', title='', fn_mg='', with_pc=True):
        """
        Parameters
        ----------
        fn : str or pathlib.Path
            Output file name with extension `.lmp` or `.xyz`.
        title : str
            Title to appear on the first line
        fn_mg : str or pathlib.Path
            Name of file containing molecule and group data. Not written if an
            empty string or contains only spaces.
        with_pc : bool
            Include pair coefficients in Lammps data file? 
        Returns
        -------
        None

        """
        root, ext = os.path.splitext(fn); ft = ext.lower().lstrip('.')
        if ft == 'lmp':
            write_ldf(self, fn, title=title, with_pc=with_pc)
        elif ft == 'xyz':
            write_xyz(self, fn, title=title)
        else:
            raise ValueError("Bad file type")
        if len(fn_mg.strip()) != 0:
            write_mol_grp(self, fn_mg, title)



    def read(self, fn_ldf, fn_mg='', fn_pc=''):
        """
        Reads nanoplatelet configuration from a file.

        Parameters
        ----------
        fn_ldf : str or pathlib.Path
            Lammps data file
        fn_mg : str or pathlib.Path
            Name of file containing molecule and group data. Not read if an
            empty string or contains only spaces.
        fn_pc : str or pathlib.Path
            Name of file containing pair coefficients. Cross coefficients not
            read. Not read if an empty string or contains only spaces. Pair
            coefficients already present in `fn_ldf` will be overwritten.

        Returns
        -------
        None

        """
        #Read LAMMPS data file
        read_ldf(self, fn_ldf) 
        #Read molecules/group records
        if len(fn_mg.strip()) != 0:
            read_mol_grp(self, fn_mg)
        #Read pair coeffs
        if len(fn_pc.strip()) != 0:
            with open(fn_pc, 'r') as fh:
                lines = fh.readlines()
            for each in lines:
                words = each.strip(' \n').split()
                at_i = int(words[1]); at_j = int(words[2])
                if at_i == at_j:
                    params = [float(x) for x in words[3:]]
                    self.set_pair_coeff(at_i, params)

        #if 'ligands' in self.groups:
        #    grp_lig = self.groups['ligands']
        #    self._num_atoms_ligands = len(grp_lig['atoms'])
        #    self._num_atom_types_ligands = len(grp_lig['atom_types'])
        #    self._num_molecules_ligands = len(grp_lig['molecules'])



