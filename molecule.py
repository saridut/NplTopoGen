#!/usr/bin/env python

import os
import warnings
import math
import numpy as np
from copy import deepcopy
from _geom_utils import fix_axis_angle, rotate_vector_axis_angle
from _configuration import Configuration
from _config_io import read_ldf, write_ldf, write_xyz, write_mol_grp, \
        add_molecules


class Molecule(Configuration):
    """
    Class implementing a molecule.

    """
    def __init__(self, name, fn) :
        """
        Parameters
        ----------
        name : str
            Name of the solvent molecule.
        fn : str or pathlib.Path
            Path of the file containing the molecular configuration as a LAMMPS
            data file.

        """
        super().__init__()

        read_ldf(self, fn)
        self.name = name

        #Find an OBB for the atoms
        na = len(self.atoms)
        positions = np.zeros((na,3))
        for i in range(1, na+1):
            positions[i-1,:] = self.atoms[i]['coords']
        com = positions.mean(axis=0)
        positions -= np.tile(com, (na,1))
        cov = np.matmul(positions.T, positions)/na
        eigvals, eigvecs = np.linalg.eigh(cov, UPLO='L')
        positions = positions @ eigvecs

        for i in range(1, na+1):
            self.atoms[i]['coords'][:] = positions[i-1,:]

        #Update molecule population
            self.molecules[1] = {'name': self.name, 'atm_beg': 1,
                                 'atm_end': self.num_atoms}
        #Place the molecule in the positive quadrant, with its bottom, left, back
        #corner to the origin.
        self.fit_simbox(sep=0.0)
        r = self.simbox[:,0]
        self.translate(-r)


    def translate_atom(self, atm_id, pos, out_coords=None):
        """
        Translate a molecule such that an atom is brought to a given point.
        Note that the box around the molecule will not be translated. Call
        the `fit_simbox` method to set a new box around the molecule.

        Parameters
        ----------
        atm_id : int
            Atom id 
        pos : (3,) ndarray of floats
            Point where atom `atm_id` should be moved.
        out_coords : (n,3) ndarray of floats
            If not None, the atom coordinates after translation will be placed
            here. The original coordinates will remain unchanged.

        Returns
        -------
        None

        """
        assert atm_id in self.atoms.keys()
        dr = pos - self.atoms[atm_id]['coords']
        if out_coords is None:
            self.translate(dr, only_atoms=True)
            self.fit_simbox()
        else:
            self.translate(dr, only_atoms=True, out_coords=out_coords)

            


    def rotate(self, angle, axis, pivot, out_coords=None):
        """
        Rotates a molecule rigidly by an angle about an axis.
        Note that the box around the molecule will not be translated. Call
        the `fit_simbox` method to set a new box around the molecule.

        Parameters
        ----------
        angle : float
            Angle of rotation (in radian) 
        axis : (3,) ndarray of floats
            Axis of rotation (need not be a unit vector)
        pivot : (3,) ndarray of floats
            Point in space about which to rotate
        out_coords : (n,3) ndarray of floats
            If not None, the atom coordinates after rotation will be placed
            here. The original coordinates will remain unchanged.

        Returns
        -------
        None

        """
        uhat, theta = fix_axis_angle(axis, angle, normalize=True)
        n = self.num_atoms
        coords = np.zeros((n,3))
        for i in range(n):
            coords[i,:] = self.atoms[i+1]['coords'] - pivot

        if out_coords is None:
            rotated_coords = rotate_vector_axis_angle(coords, uhat, theta)
            for i in range(n):
                self.atoms[i+1]['coords'] = rotated_coords[i,:] + pivot
            self.fit_simbox()
        else:
            out_coords = rotate_vector_axis_angle(coords, uhat, theta)
            for i in range(n):
                out_coords[i,:] += pivot


    def align(self, atm_1, atm_2, axis):
        """
        Rotates a molecule rigidly to align the line through two atoms along a
        direction. The two atoms need not be bonded. The pivot point for 
        rotation will be the position of `atm_1`.
        Note that the box around the molecule will not be translated. Call
        the `fit_simbox` method to set a new box around the molecule.

        Parameters
        ----------
        atm_1 : int
            Atom id
        atm_2 : int
            Atom id
        axis : (3,) ndarray of floats
            Direction along which to align (need not be a unit vector)

        Returns
        -------
        None

        """
        pivot = self.atoms[atm_1]['coords']
        phat = axis/np.linalg.norm(axis) #Direction unit vector
        v = self.atoms[atm_2]['coords'] - self.atoms[atm_1]['coords']
        vhat = v/np.linalg.norm(v) #Line unit vector
        u = np.cross(vhat, phat); uhat = u/np.linalg.norm(u)
        ctheta = np.dot(vhat, phat); theta = math.acos(ctheta)
        self.rotate(theta, uhat, pivot)
        self.fit_simbox()


    def get_gyration_radius(self, atoms=None):
        """
        Parameters
        ----------
        atoms : None | iterable of ints
            Atom ids. If ``None``, calculate for all atoms.

        Returns
        -------
        float

        """
        if atoms is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atoms
        gyr_ten = np.zeros((3,3))
        c = self.get_com(aids); totmass = 0.0
        for i in aids:
            at = self.atoms[i]['type']
            mass = self.atom_mass[at]
            totmass += mass
            r = self.atoms[i]['coords'] - c
            gyr_ten[0,0] += (mass*r[0]*r[0]) 
            gyr_ten[0,1] += (mass*r[0]*r[1]) 
            gyr_ten[0,2] += (mass*r[0]*r[2]) 
            gyr_ten[1,1] += (mass*r[1]*r[1]) 
            gyr_ten[1,2] += (mass*r[1]*r[2]) 
            gyr_ten[2,2] += (mass*r[2]*r[2]) 
        gyr_ten /= totmass
        gyr_ten[1,0] = gyr_ten[0,1]
        gyr_ten[2,0] = gyr_ten[0,2]
        gyr_ten[2,1] = gyr_ten[1,2]
        rg = math.sqrt(gyr_ten[0,0]+gyr_ten[1,1]+gyr_ten[2,2])
        return (rg, gyr_ten[0,0]**0.5, gyr_ten[1,1]**0.5, gyr_ten[2,2]**0.5,
                gyr_ten)





    def write(self, fn='', title='', fn_mg=''):
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
            write_ldf(self, fn, title=title)
        elif ft == 'xyz':
            write_xyz(self, fn, title=title)
        else:
            raise ValueError("Bad file type")
        if len(fn_mg.strip()) != 0:
            write_mol_grp(self, fn_mg, title)



    def solvate(self, solvent, molfrac, boxx, boxy=None, boxz=None, 
                    packmol_tol=2.0, packmol_path=''):
        """
        Adds solvent molecules around a molecule.

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
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_path : str or pathlib.Path
            Path to Packmol binary.

        """
        print('Solvating ...')
        if boxy is None:
            boxy = boxx
        if boxz is None:
            boxz = boxx
        #Set current bounding box center at the origin
        r = self.simbox.mean(axis=1)
        self.translate(-r)
        cboxx = self.simbox[0,1] - self.simbox[0,0]
        cboxy = self.simbox[1,1] - self.simbox[1,0]
        cboxz = self.simbox[2,1] - self.simbox[2,0]
        if (boxx < cboxx) or (boxy < cboxy) or (boxz < cboxz):
            warnings.warn(
                f"\nFinal box size smaller than current box size:\n"
                f"  Final: ({boxx:g}, {boxy:g}, {boxz:g})\n"
                f"  Current: ({cboxx:g}, {cboxy:g}, {cboxz:g}).\n"
                f"  Final box will be set to a cube with side equal to the "
                f"longest extent of current box size."
                )
            boxx_ = max(cboxx, cboxy, cboxz); boxy_ = boxx_; boxz_ = boxx_
        else:
            boxx_ = boxx; boxy_ = boxy; boxz_ = boxz

        #Set the current simbox
        self.simbox[0,0] = -boxx_/2; self.simbox[0,1] = boxx_/2
        self.simbox[1,0] = -boxy_/2; self.simbox[1,1] = boxy_/2
        self.simbox[2,0] = -boxz_/2; self.simbox[2,1] = boxz_/2
        volume = (self.simbox[:,1]-self.simbox[:,0]).prod()

        #Outer bounding box: simbox box slightly reduced by delta
        delta = 2.0 #(As recommended on Packmol user's manual)
        bbox_out_lo = self.simbox[:,0] + delta
        bbox_out_hi = self.simbox[:,1] - delta

        print(f"Packing solvents \n"
              f" inside box ({' '.join('%g'%v for v in bbox_out_lo)})"
              f" ({' '.join('%g'%v for v in bbox_out_hi)})"
              )

        num_components = len(solvent)
        comp_pop = [] #Population of molecules for each component

        #Calculate population of each component molecule
        na_solvent = 0
        for i in range(num_components):
            molwt = solvent[i][0].get_total_mass()
            ndens = solvent[i][1]*0.6023/molwt #number per angstrom^3
            pop = molfrac[i]*ndens*volume
            if pop < 1:
                s = input('Zero molecules of solvent component %s. Type "Y" to '
                          'continue, any other key to exit: '%solvent[i][0].name)
                if s != "Y":
                    raise SystemExit("Exiting ...")
            comp_pop.append( int(np.rint(pop)) )
            na_solvent += ( comp_pop[i]*solvent[i][0].num_atoms )
            print('  Number of %s molecules = %g'%(solvent[i][0].name,
                                                 comp_pop[i]))
        print('Total number of solvent molecules = %d'%sum(comp_pop)) 
        print('Total number of solvent atoms = %d'%na_solvent)

        #Add solvent molecules
        mols_to_add = []
        for i in range(num_components):
            g_in = 'inside box '
            g_in += ' '.join(['%g'%x for x in bbox_out_lo])
            g_in += ' '
            g_in += ' '.join(['%g'%x for x in bbox_out_hi])
            elem = {'moltem': solvent[i][0], 'num': comp_pop[i],
                    'constraints': [g_in]}
            mols_to_add.append(elem)
        add_molecules(self, mols_to_add, packmol_tol=packmol_tol,
                      packmol_path=packmol_path)



class LigandMolecule(Molecule):
    """
    Class implementing a ligand molecule.

    """
    def __init__(self, name, fn, head, tail, bind_group) :
        """
        Parameters
        ----------
        name : str
            Name of the ligand molecule.
        fn : str or pathlib.Path
            Path of the file containing the ligand configuration.
        head : int
            Atom id of the ligand head. The id is local, i.e. with respect to
            the ligand molecule. Ligand head is the first atom after the binding
            group in the ligand main chain.
        tail : int
            Atom id of the ligand tail. The id is local, i.e. with respect to
            the ligand molecule. Ligand tail is the last atom in the ligand main
            chain.
        bind_group : sequence of ints
            Atom ids of the binding group of the ligand. The ids are local, i.e.
            with respect to the ligand molecule.

        """
        super().__init__(name, fn)
        self.head = head
        self.tail = tail
        self.bind_group = deepcopy(bind_group)
        self.charge = self.get_total_charge()
