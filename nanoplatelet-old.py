#!/usr/bin/env python

"""
Class implementing a single NPL crystal.

"""

import os
import warnings
import math
import numpy as np
from _configuration import Configuration
from _geom_utils import rotate_vector_axis_angle
from _config_io import read_ldf, write_ldf, write_xyz, write_mol_grp, \
        add_molecules


class NanoPlatelet(Configuration):
    def __init__(self, xtal, length, width, unit='nm', num_mono_layers=3, 
                 phi=45, pbc=False, charges=None, pair_coeffs=None):
        """
        Parameters
        ----------
        xtal : Crystal
            Instance of `Crystal`
        length : float
            Length of the nanoplatelet (NPL)
        width : float
            Width of the nanoplatelet (NPL)
        unit : {'ang' | 'nm' | 'lattice'}
            Unit of `length` and `width`. Note that the unit associated with
            `xtal` is always in angstrom.
        num_mono_layers : int
            Number of monolayers.
        phi : float
            Angle between the lengthwise edge and (110) direction in degrees.
        pbc : bool
            Whether to apply periodic boundary conditions along the planar
            directions. Note that the thickness direction is never periodic.
            If `pbc = True`, `phi` is neglected.
        charges : sequence of floats
            Charge for each atom type
        pair_coeffs : sequence of lists
            Parameters for pair interactions 

        """
        super().__init__()

        self.xtal = xtal
        self.length = length
        self.width = width
        self.unit = unit
        self.num_mono_layers = num_mono_layers
        self.phi = phi
        self.pbc = pbc

        self._num_atoms_xtal = 0
        self._num_atom_types_xtal = 0
        self._bbox_xtal = None
        self._num_atoms_ligands = 0
        self._num_atom_types_ligands = 0
        self._num_molecules_ligands = 0
        self._bbox_ligands = None

        print('Building crystal ...')

        if self.unit == 'nm':
            hlen = 10.0*self.length/2.0; hwid = 10.0*self.width/2.0
        elif self.unit == 'ang':
            hlen = self.length/2.0; hwid = self.width/2.0
        elif self.unit == 'lattice':
            hlen = self.xtal.a*self.length/2.0
            hwid = self.xtal.b*self.width/2.0
        height = xtal.c*self.num_mono_layers/2.0
        
        #Atom types
        num_atom_types = self.xtal.get_num_atom_types()
        for i in range(1, num_atom_types+1):
            name = self.xtal.get_atom_name(i); mass = self.xtal.get_atom_mass(i)
            self.add_atom_type(mass, name)
        
        #Bounding box centered at the origin
        hlen_bbox = math.hypot(hlen, hwid) #Half length of the bounding box
        #Number of unit cells along x, y, and z directions
        nx = math.ceil(hlen_bbox/self.xtal.a) + 2 #add two additonal unit cells,
                                                  #just in case
        ny = nx
        nz =  math.ceil((self.num_mono_layers+1)/2)
        nc = (2*nx)*(2*ny)*nz #Total number of unit cells
        na_uc = self.xtal.get_num_atoms() #Number of atoms per unit cell
        na_all = nc*na_uc #Total number of atoms
        at_uc = self.xtal.get_atom_type() #Type of all atoms per unit cell
        
        coords = np.zeros((na_all,3), dtype=np.float64)
        atm_types = np.zeros((na_all,), dtype=np.int32)
        icoord = 0
        for k in range(0,nz):
            for j in range(-ny,ny):
                for i in range(-nx,nx):
                    ibeg = icoord; iend = icoord + na_uc
                    origin = [i*self.xtal.a, j*self.xtal.b, k*self.xtal.c]
                    coords[ibeg:iend,:] = np.tile(origin,(na_uc,1)) + self.xtal.ucell
                    atm_types[ibeg:iend,] = at_uc
                    icoord += na_uc

        #Rotate the lattice points about z-axis
        axis = np.array([0, 0, 1])
        coords = rotate_vector_axis_angle(coords, axis, math.radians(self.phi-45))

        #Add atoms
        for iatm in range(na_all):
            pos = coords[iatm,:]
            if (abs(pos[0]) <= hlen) and (abs(pos[1]) <= hwid) \
                    and (pos[2] <= height):
                at = atm_types[iatm]
                chge = charges[at-1]
                self.add_atom(at, chge, pos)

        #Set pair coeffs
        for i in range(1, self.num_atom_types+1):
            self.set_pair_coeff(i, pair_coeffs[i-1])
        
        #Wrap in a simulation box and center at the origin
        self.fit_simbox(sep=0.0)
        r = self.simbox.sum(axis=1)/2
        self.translate(-r)

        self._num_atoms_xtal = self.num_atoms
        self._num_atom_types_xtal = self.num_atom_types
        self._bbox_xtal = self.simbox.copy()

        #Create new group of crystal atoms
        self.set_group('xtal', atoms=range(1, self._num_atoms_xtal+1),
                       atom_types=range(1,self._num_atom_types_xtal+1))

        print(f"NPL crystal bounding box ="
              f" ({' '.join('%g'%v for v in self._bbox_xtal[:,0])})"
              f" ({' '.join('%g'%v for v in self._bbox_xtal[:,1])})"
              )
        print("Number of NPL crystal atom types = %d"%self._num_atom_types_xtal)
        print("Number of NPL crystal atoms = %d"%self._num_atoms_xtal)
        print("Charge of the NPL crystal = %g"%self.get_total_charge())
        


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



    def add_ligands(self, ligand_list, ligand_pop_ratio, dist, 
                    packmol_tol=2.0, packmol_path=''):
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
        dist : float
            Distance around the nanoplatelet within which the ligands will be
            placed. This value is ignored if smaller than the largest extent of
            the ligands.
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_path : str or pathlib.Path
            Path to Packmol binary.

        """
        totchge = self.get_total_charge()
        
        print('Adding ligands ...')

        num_ligand_types = len(ligand_list)
        lig_charge = []; lig_pop = []; lig_size_max = 0.0

        for each in ligand_list:
            size = np.amax(each.simbox[:,1] - each.simbox[:,0])
            lig_size_max = max(lig_size_max, size)
            lig_charge.append(each.get_total_charge())
        print('  Maximum ligand size = %g'%lig_size_max)
        if dist < lig_size_max:
            s = input(f"  `dist` = {dist} is smaller than the largest ligand"
                      f" extent = {lig_size_max}. Do you want to enter a"
                      f" different value for `dist`? [N/value] ")
            if s == 'N':
                dist_ = dist
            else:
                dist_ = float(s)
        else:
            dist_ = dist

        #Ratio common factor
        cf = abs(totchge) / abs( np.dot(lig_charge, ligand_pop_ratio) )
        na_ligands = 0
        for i in range(num_ligand_types):
            pop = ligand_pop_ratio[i]*cf
            if pop < 1:
                s = input('Zero ligands %s ligands. Type "Y" to '
                          'continue, any other key to exit: '%ligand_list[i].name)
                if s != "Y":
                    raise SystemExit("Exiting ...")
            lig_pop.append( int(np.rint(pop)) )
            na_ligands += ( lig_pop[i]*ligand_list[i].num_atoms )
            print('  Number of %s ligand molecules = %g'%(
                    ligand_list[i].name, lig_pop[i]))
        print('  Total number of ligand molecules = %d'%sum(lig_pop)) 
        print('  Total number of ligand atoms = %d'%na_ligands)

        #Inner bounding box: xtal bounding box slightly increased by delta
        delta = 1.0
        bbox_in_lo = self.simbox[:,0] - delta
        bbox_in_hi = self.simbox[:,1] + delta
        
        #Enlarge the current simbox
        self.simbox[:,0] -= dist_
        self.simbox[:,1] += dist_

        #Outer bounding box: simbox box slightly reduced by delta
        delta = 2.0 #(As recommended on Packmol user's manual)
        bbox_out_lo = self.simbox[:,0] + delta
        bbox_out_hi = self.simbox[:,1] - delta
        
        #Add ligands
        ligands_to_add = []
        for i in range(num_ligand_types):
            g_in = 'inside box '
            g_in += ' '.join(['%g'%x for x in bbox_out_lo])
            g_in += ' '
            g_in += ' '.join(['%g'%x for x in bbox_out_hi])
            g_out = 'outside box '
            g_out += ' '.join(['%g'%x for x in bbox_in_lo])
            g_out += ' '
            g_out += ' '.join(['%g'%x for x in bbox_in_hi])
            elem = {'moltem': ligand_list[i], 'num': lig_pop[i],
                    'constraints': [g_in, g_out]}
            ligands_to_add.append(elem)
        add_molecules(self, ligands_to_add, packmol_tol=packmol_tol,
                      packmol_path=packmol_path)
        #Adjust charge
        chge = self.get_total_charge()
        if chge != 0:
            cpa = chge/len(self.atoms)
            for i in range(1, len(self.atoms)+1):
                self.atoms[i]['charge'] -= cpa
        print("Total charge after adding ligands = %g"%self.get_total_charge())

        self._num_atoms_ligands = self.num_atoms - self._num_atoms_xtal
        self._num_atom_types_ligands = \
                self.num_atom_types - self._num_atom_types_xtal
        self._num_molecules_ligands = self.num_molecules
        self._bbox_ligands = self.simbox.copy()

        #Create new group of ligand atoms
        atoms = range(self._num_atoms_xtal+1, self.num_atoms+1)
        atom_types = range(self._num_atom_types_xtal+1, self.num_atom_types+1)
        molecules = range(1, self._num_molecules_ligands+1)
        self.set_group('ligands', atoms=atoms, atom_types=atom_types,
                       molecules=molecules)



    def solvate(self, solvent, molfrac, boxx, boxy=None, boxz=None, 
                    dist=0.0, packmol_tol=2.0, packmol_path=''):
        """
        Adds solvent molecules around a nanoplatelet.

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
        dist : float
            Distance from the crystal surface beyond which the solvent molecules
            will be placed.
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
        #Current bounding box
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
        #Inner bounding box: xtal bounding box slightly increased by delta
        delta = 1.0
        bbox_in_lo = self._bbox_xtal[:,0] - dist - delta
        bbox_in_hi = self._bbox_xtal[:,1] + dist + delta

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
              f" outside box ({' '.join('%g'%v for v in bbox_in_lo)})"
              f" ({' '.join('%g'%v for v in bbox_in_hi)})\n"
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
            g_out = 'outside box '
            g_out += ' '.join(['%g'%x for x in bbox_in_lo])
            g_out += ' '
            g_out += ' '.join(['%g'%x for x in bbox_in_hi])
            elem = {'moltem': solvent[i][0], 'num': comp_pop[i],
                    'constraints': [g_in, g_out]}
            mols_to_add.append(elem)
        add_molecules(self, mols_to_add, packmol_tol=packmol_tol,
                      packmol_path=packmol_path)
        #Adjust charge
        chge = self.get_total_charge()
        if chge != 0:
            cpa = chge/self.num_atoms
            for i in range(1, self.num_atoms+1):
                self.atoms[i]['charge'] -= cpa
        print("Total charge after solvation = %g"%self.get_total_charge())

        #Create new group of solvent atoms
        atoms = range(self._num_atoms_xtal+self._num_atoms_ligands+1, 
                      self.num_atoms+1)
        atom_types = range(
                self._num_atom_types_xtal+self._num_atom_types_ligands+1,
                self.num_atom_types+1)
        molecules = range(self._num_molecules_ligands+1, self.num_molecules+1)
        self.set_group('solvent', atoms=atoms, atom_types=atom_types,
                       molecules=molecules)


    def tweak_cvff_impropers(self):
        """
        Ensure that CVFF improper coefficients are integers.

        """
        for val in self.improper_coeffs.values():
            val[1] = int(val[1]); val[2] = int(val[2])


    def gen_ff_pair(self, fn, soften=None):
        """
        Writes pair interaction coefficients to a file.

        soften : {'ligands', 'solvent', 'both'}
            Soften potential for this group.

        """
        xtal_atom_types = sorted(list(self.groups['xtal']['atom_types']))
        if soften is None:
            with open(fn, 'w') as fh:
                #pair_style lj/cut/coul/long 12.0
                #pair_modify shift yes mix geometric
                #special_bonds lj/coul 0.0 0.0 0.5 angle yes dihedral yes
                for i in range(1, self.num_atom_types+1):
                    buf = f"pair_coeff {i} {i} "
                    buf += ' '.join(str(x) for x in self.pair_coeffs[i])
                    fh.write(buf+"\n")
                for ii in range(len(xtal_atom_types)-1):
                    i = xtal_atom_types[ii]
                    for jj in range(ii+1, len(xtal_atom_types)):
                        j = xtal_atom_types[jj]
                        coeffs_i = self.pair_coeffs[i]
                        coeffs_j = self.pair_coeffs[j]
                        eps_i = coeffs_i[0]; sigma_i = coeffs_i[1]
                        eps_j = coeffs_j[0]; sigma_j = coeffs_j[1]
                        eps_ij = math.sqrt(eps_i*eps_j)
                        #Arithmetic mixing for between xtal types
                        sigma_ij = (sigma_i+sigma_j)/2.0
                        fh.write(f"pair_coeff {i} {j} {eps_ij:g} {sigma_ij:g}\n")
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
                    cutoffs = ' '.join(str(x) for x in coeffs[2:])
                    fh.write(f"pair_coeff {i} {i} {eps} {sigma} {lamda}"
                             f" {cutoffs}\n")
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
                        #Arithmetic mixing for between xtal types
                        if (i in xtal_atom_types) and (j in xtal_atom_types):
                            sigma_ij = (sigma_i+sigma_j)/2.0
                        else:
                            sigma_ij = math.sqrt(sigma_i*sigma_j)
                        fh.write(f"pair_coeff {i} {j} {eps_ij:g}"
                                 f" {sigma_ij:g} {lamda} \n")



        
class SolvatedNanoPlatelets(Configuration):
    """
    Class implementing nanoplatelets coated with ligands and immersed in a box
    of solvent molecules.

    """
    def __init__(self, npl,  use_packmol=False):
        """
        Parameters
        ----------
        npl : NanoPlatelet
            Instance of `NanoPlatelet`.
        use_packmol : bool
            Whether to use PackMol to place the ligands around the nanoplatelet.

        """
        raise NotImplementedError()
        #super().__init__()
