#!/usr/bin/env python

"""
Class implementing a box of solvent molecules.

"""

import os
import warnings
import math
import numpy as np
from _configuration import Configuration
from _geom_utils import rotate_vector_axis_angle
from _config_io import read_ldf, write_ldf, write_xyz, write_mol_grp, \
        add_molecules


class SolventBox(Configuration):
    """
    Class implementing a box of solvent molecules. 

    """
    def __init__(self, solvent, molfrac, boxx, boxy=None, boxz=None,
                solutes=[], solute_comp_pop=[], force_neutral=False, 
                 packmol_tol=2.0, packmol_path=''):
        """
        
        Parameters
        ----------
        solvent : list
            Solvent molecules to add. Each element of the list is a tuple 
            (Molecule, density), where density is in g/mL.
        molfrac : list of int
            Mole fraction of each component of the solvent. Use [1] for a pure
            solvent.
        boxx : float
            Length of the simulation box along the x-direction (angstrom).
        boxy : float
            Length of the simulation box along the y-direction. If None, is set
            equal to `boxx` (angstrom).
        boxz : float
            Length of the simulation box along the z-direction. If None, is set
            equal to `boxx` (angstrom).
        solutes : list of Molecule
            Solute molecules to add.
        solute_comp_pop : list of int
            Number of solute molecules to add for each entry in `solutes`.
        force_neutral : bool
            Force the system to be neutral by subtracting residual charge from
            each atom.
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_path : str or pathlib.Path
            Path to Packmol binary.

        """
        super().__init__()
        if boxy is None:
            boxy = boxx
        if boxz is None:
            boxz = boxx

        #Set the current simbox
        self.simbox[0,0] = -boxx/2; self.simbox[0,1] = boxx/2
        self.simbox[1,0] = -boxy/2; self.simbox[1,1] = boxy/2
        self.simbox[2,0] = -boxz/2; self.simbox[2,1] = boxz/2
        volume = (self.simbox[:,1]-self.simbox[:,0]).prod()
        
        #Outer bounding box: simbox box slightly reduced by delta
        delta = 2.0 #(As recommended on Packmol user's manual)
        bbox_out_lo = self.simbox[:,0] + delta
        bbox_out_hi = self.simbox[:,1] - delta

        print(f"Packing solvents \n"
              f" inside box ({' '.join('%g'%v for v in bbox_out_lo)})"
              f" ({' '.join('%g'%v for v in bbox_out_hi)})"
              )

        num_solute_components = len(solutes)
        num_solvent_components = len(solvent)
        solvent_comp_pop = [] #Population of molecules for each solvent component

        #Solute groups
        na_solute = 0; nm_solute = 0; at_solute = 0
        for i in range(num_solute_components):
            na_solute_beg = na_solute + 1
            nm_solute_beg = nm_solute + 1
            at_solute_beg = at_solute + 1
            if solute_comp_pop[i] > 0:
                na_solute += ( solute_comp_pop[i]*solutes[i].num_atoms )
                nm_solute += solute_comp_pop[i]
                at_solute += solutes[i].num_atom_types
                #Create a separate groups for each component
                self.set_group(solutes[i].name, 
                    atoms=range(na_solute_beg, na_solute+1),
                    atom_types=range(at_solute_beg, at_solute+1),
                    molecules=range(nm_solute_beg, nm_solute+1))
        #Create a separate groups for all solutes combined
        if num_solute_components > 0:
            self.set_group('Solute', atoms=range(1, na_solute+1), 
                atom_types=range(1, at_solute+1), 
                molecules=range(1, nm_solute+1))
        print('Total number of solute atom types = %d'%at_solute)
        print('Total number of solute atoms = %d'%na_solute)
        print('Total number of solute molecules = %d'%sum(solute_comp_pop)) 

        #Calculate population of molecules for each solvent component
        na_solvent = na_solute; nm_solvent = nm_solute; at_solvent = at_solute
        for i in range(num_solvent_components):
            na_solvent_beg = na_solvent + 1
            nm_solvent_beg = nm_solvent + 1
            at_solvent_beg = at_solvent + 1
            molwt = solvent[i][0].get_total_mass()
            ndens = solvent[i][1]*0.6023/molwt #number per angstrom^3
            pop = molfrac[i]*ndens*volume
            if pop < 1:
                s = input('Zero molecules of solvent component %s. Type "Y" to '
                          'continue, any other key to exit: '%solvent[i][0].name)
                if s != "Y":
                    raise SystemExit("Exiting ...")
            solvent_comp_pop.append( int(np.rint(pop)) )
            print('  Number of %s molecules = %g'%(solvent[i][0].name,
                                                 solvent_comp_pop[i]))
            if solvent_comp_pop[i] > 0:
                na_solvent += ( solvent_comp_pop[i]*solvent[i][0].num_atoms )
                nm_solvent += solvent_comp_pop[i]
                at_solvent += solvent[i][0].num_atom_types
                #Create a separate groups for each component
                self.set_group(solvent[i][0].name, 
                    atoms=range(na_solvent_beg, na_solvent+1),
                    atom_types=range(at_solvent_beg, at_solvent+1),
                    molecules=range(nm_solvent_beg, nm_solvent+1))
        #Create a separate groups for solvent (all components)
        self.set_group('Solvent', 
            atoms=range(na_solute+1, na_solvent+1), 
            atom_types=range(at_solute+1, at_solvent+1), 
            molecules=range(nm_solute+1, nm_solvent+1))
        print('Total number of solvent atom types = %d'%(at_solvent-at_solute))
        print('Total number of solvent atoms = %d'%(na_solvent-na_solute))
        print('Total number of solvent molecules = %d'%sum(solvent_comp_pop)) 

        #Add molecules
        mols_to_add = []
        for i in range(num_solute_components):
            g_in = 'inside box '
            g_in += ' '.join(['%g'%x for x in bbox_out_lo])
            g_in += ' '
            g_in += ' '.join(['%g'%x for x in bbox_out_hi])
            elem = {'moltem': solutes[i], 'num': solute_comp_pop[i],
                    'constraints': [g_in]}
            mols_to_add.append(elem)

        for i in range(num_solvent_components):
            g_in = 'inside box '
            g_in += ' '.join(['%g'%x for x in bbox_out_lo])
            g_in += ' '
            g_in += ' '.join(['%g'%x for x in bbox_out_hi])
            elem = {'moltem': solvent[i][0], 'num': solvent_comp_pop[i],
                    'constraints': [g_in]}
            mols_to_add.append(elem)

        add_molecules(self, mols_to_add, packmol_tol=packmol_tol,
                      packmol_path=packmol_path)

        #Adjust charge
        if force_neutral:
            chge = self.get_total_charge()
            if chge != 0:
                cpa = chge/len(self.atoms)
                for i in range(1, len(self.atoms)+1):
                    self.atoms[i]['charge'] -= cpa
        print("Total charge after adding ligands = %g"%self.get_total_charge())

        

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



    def tweak_cvff_impropers(self):
        """
        Ensure that CVFF improper coefficients are integers.

        """
        for val in self.improper_coeffs.values():
            val[1] = int(val[1]); val[2] = int(val[2])



    def gen_ff_pair(self, fn, soften=False):
        """
        Writes pair interaction coefficients to a file.

        soften : bool
            Generate softened version of the pair potential.

        Notes
        -----
        For `soften == True`, the following pair style holds:
         pair_style lj/cut/coul/long/soft 2 0.5 10.0 12.0
         pair_modify shift yes mix geometric
         special_bonds lj/coul 0.0 0.0 0.5 angle yes dihedral yes

        For `soften == False`, the following pair style holds:
         pair_style lj/cut/coul/long 12.0
         pair_modify shift yes mix geometric
         special_bonds lj/coul 0.0 0.0 0.5 angle yes dihedral yes

        """
        with open(fn, 'w') as fh:
            #Pair coeffs for self types (I == J)
            lamda = 0.0 if soften else ''
            for i in range(1, self.num_atom_types+1):
                coeffs = self.pair_coeffs[i]
                eps = coeffs[0]; sigma = coeffs[1]
                if len(coeffs) > 2:
                    cutoffs = ' '.join(str(x) for x in coeffs[2:])
                else:
                    cutoffs = ''
                fh.write(f"pair_coeff {i} {i} {eps} {sigma} {lamda}"
                         f" {cutoffs}\n")


        
