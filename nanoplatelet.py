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
        read_mol_grp, add_molecules


class NanoPlatelet(Configuration):
    def __init__(self, is_slab):
        """
        Parameters
        ----------
        is_slab : bool
            Whether the crystal is a slab or not. A slab is periodic in the x &
            y directions and centered in the box parallel to the xy-plane.
            
        """
        super().__init__()
        self.is_slab = is_slab
        self.xtal = None
        self.num_mono_layers = 0
        self._bbox_xtal = None

        

    def add_xtal(self, xtal, length, width, num_mono_layers, phi=45,
                balanced=False, pbc_xy=False, pbc_z=False, unit='ang',
                charges=None, pair_coeffs=None):
        """
        Adds a crystal core.

        Parameters
        ----------
        xtal : Crystal
            Instance of `Crystal`
        length : float
            Length of the nanoplatelet (NPL) in angstrom
        width : float
            Width of the nanoplatelet (NPL) in angstrom
        num_mono_layers : int
            Number of monolayers.
        phi : float
            Angle between the lengthwise edge and (110) direction in degrees.
        balanced : bool
            Whether there are same number of monolayers for both atomic species
            in the crystal.
        pbc_xy : bool
            Whether the crystal is periodic along the planar (x & y) directions.
            If `pbc_xy = True`, `phi` is neglected.
        pbc_z : bool
            Whether the crystal is periodic along the perpendicular (z)
            direction. If `pbc_z = True`, `balanced` must be true and
            `num_mono_layers` must be a positive multiple of two.
        unit : {'ang' | 'lattice'}
            Unit of `length` and `width`. 'lattice' units will be converted to
            angstrom.
        charges : sequence of floats
            Charge for each atom type
        pair_coeffs : sequence of lists
            Parameters for pair interactions 

        """
        print('Building crystal ...')

        self.xtal = xtal
        self.num_mono_layers = num_mono_layers
        #Atom types
        num_atom_types_xtal = self.xtal.get_num_atom_types()
        for i in range(1, num_atom_types_xtal+1):
            name = self.xtal.get_atom_name(i)
            mass = self.xtal.get_atom_mass(i)
            self.add_atom_type(mass, name)
        
        #Repeat unit spanning the x-y plane
        ru_coords = []; ru_at = [] #Type of all atoms per unit cell
        if balanced:
            imax = 2*num_mono_layers
        else:
            imax = 2*num_mono_layers + 1
        for i in range(1, imax+1):
            pos, typ = self.xtal.get_atoms_in_layer(i)
            ru_coords.extend(pos); ru_at.extend(typ)
        ru_coords = np.asarray(ru_coords)
        ru_at = np.asarray(ru_at)

        num_atoms_xtal = 0
        if pbc_xy:
            if unit == 'ang':
                nx = math.floor(length/self.xtal.a)
                ny = math.floor(width/self.xtal.b)
            elif unit == 'lattice':
                nx = int(length); ny = int(width)

            for j in range(ny):
                for i in range(nx):
                    origin = np.array([i*self.xtal.a, j*self.xtal.b, 0])
                    for pos, at in zip(ru_coords, ru_at):
                        self.add_atom(at, charges[at-1], pos+origin)
                        num_atoms_xtal += 1

            #Wrap in a simulation box and center at the origin
            self.fit_simbox()
            self.simbox[:,0] = 0.0
            self.simbox[0,1] = nx*self.xtal.a
            self.simbox[1,1] = ny*self.xtal.b
            if pbc_z:
                self.simbox[2,1] = (num_mono_layers//2)*self.xtal.c
            r = self.simbox.sum(axis=1)/2
            self.translate(-r)
            #r = self.get_barycenter()
            #self.translate(-r, only_atoms=True)
        else:
            if unit == 'ang':
                hlen = length/2; hwid = width/2
            elif unit == 'lattice':
                hlen = self.xtal.a*length/2
                hwid = self.xtal.b*width/2
        
            #Bounding box centered at the origin
            hlen_bbox = math.hypot(hlen, hwid) #Half length of the bounding box
            #Number of unit cells along x, y, and z directions
            nx = math.ceil(hlen_bbox/self.xtal.a) + 2 #add two additonal unit cells,
                                                      #just in case
            ny = nx
            nc = (2*nx)*(2*ny) #Total number of unit cells
            na_ru = ru_coords.shape[0] #Number of atoms per repeat unit
            na_all = nc*na_ru #Total number of atoms

            coords = np.zeros((na_all,3), dtype=np.float64)
            atm_types = np.zeros((na_all,), dtype=np.int32)
            icoord = 0
            blc = np.array([-nx*self.xtal.a, -ny*self.xtal.b, 0]) #Bottom left corner
            for j in range(2*ny):
                for i in range(2*nx):
                    ibeg = icoord; iend = icoord + na_ru
                    origin = blc + np.array([i*self.xtal.a, j*self.xtal.b, 0])
                    coords[ibeg:iend,:] = np.tile(origin,(na_ru,1)) + ru_coords
                    atm_types[ibeg:iend,] = ru_at
                    icoord += na_ru

            #Rotate the lattice points about z-axis
            axis = np.array([0, 0, 1])
            coords = rotate_vector_axis_angle(coords, axis, math.radians(phi-45))

            #Add atoms
            for iatm in range(na_all):
                pos = coords[iatm,:]
                if (abs(pos[0]) < hlen) and (abs(pos[1]) < hwid) :
                    at = atm_types[iatm]
                    chge = charges[at-1]
                    self.add_atom(at, chge, pos)
                    num_atoms_xtal += 1

            #Wrap in a simulation box and center at the origin
            self.fit_simbox(sep=0)
            if pbc_z:
                self.simbox[2,1] = self.simbox[2,0] \
                        + (num_mono_layers//2)*self.xtal.c
            r = self.simbox.sum(axis=1)/2
            self.translate(-r)

        #Set pair coeffs
        for i in range(1, num_atom_types_xtal+1):
            self.set_pair_coeff(i, pair_coeffs[i-1])

        self._bbox_xtal = self.simbox.copy()
        #Create new group of crystal atoms
        self.set_group('xtal', atoms=range(1, num_atoms_xtal+1),
            atom_types=range(1, num_atom_types_xtal+1))

        print(f"NPL crystal bounding box ="
              f" ({' '.join('%g'%v for v in self._bbox_xtal[:,0])})"
              f" ({' '.join('%g'%v for v in self._bbox_xtal[:,1])})"
              )
        print("Number of NPL crystal atom types = %d"%num_atom_types_xtal)
        print("Number of NPL crystal atoms = %d"%num_atoms_xtal)
        print("Charge of the NPL crystal = %g"%self.get_total_charge())



    def add_ligands(self, ligand_list, ligand_pop_ratio, offset,
                    thickness, thickness_bm, packmol_tol=2.0,
                    packmol_sidemax=1.0e3, packmol_path=''):
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
        offset : float
            Distance from the crystal surface beyond which ligand
            molecules will be placed.
        thickness : float
            Width of the region around the nanoplatelet where the ligands will
            be placed. The ligands are placed within a distance of `offset` and
            `offset+thickness` around the nanoplatelet.
        thickness_bm : float
            Width of the region confining only the atoms of the binding moeity.
            Must be smaller than `thickness`.
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_sidemax : float
            Parameter for Packmol. Default is 1000 angstrom.
        packmol_path : str or pathlib.Path
            Path to Packmol binary.

        """
        totchge = self.get_total_charge()
        
        print('Adding ligands ...')

        num_ligand_types = len(ligand_list)
        lig_charge = np.zeros((num_ligand_types,))
        lig_pop = np.zeros((num_ligand_types,), dtype=np.int32)

        aid_beg = self.num_atoms + 1 # First ligand atom id
        aid_end = aid_beg  # Last ligand atom id
        mid_beg = self.num_molecules + 1 #First ligand molecule id
        mid_end = mid_beg  # Last ligand molecule id
        lig_size_max = 0

        for i in range(num_ligand_types):
            each = ligand_list[i]
            size = np.amax(each.simbox[:,1] - each.simbox[:,0])
            lig_size_max = max(lig_size_max, size)
            lig_charge[i] = each.get_total_charge()
        print('  Maximum ligand size = %g'%lig_size_max)
        if thickness < lig_size_max:
            s = input(f"  `thickness` = {thickness} is smaller than the largest"
                      f" ligand extent = {lig_size_max}. Do you want to enter a"
                      f" different value for `thickness`? [N/value] ")
            if s == 'N':
                thickness_ = thickness
            else:
                thickness_ = float(s)
        else:
            thickness_ = thickness

        #Ratio common factor
        cf = abs(totchge) / abs( np.dot(lig_charge, ligand_pop_ratio) )

        for i in range(num_ligand_types):
            pop = ligand_pop_ratio[i]*cf
            if pop < 1:
                s = input('Zero ligands %s ligands. Type "Y" to '
                          'continue, any other key to exit: '%ligand_list[i].name)
                if s != "Y":
                    raise SystemExit("Exiting ...")
            lig_pop[i] = int(np.rint(pop))
            aid_end += ( lig_pop[i]*ligand_list[i].num_atoms - 1 )
            mid_end += (lig_pop[i] - 1)
            print('  Number of %s ligand molecules = %g'%(
                    ligand_list[i].name, lig_pop[i]))
        print('  Total number of ligand molecules = %d'%sum(lig_pop)) 
        print('  Total number of ligand atoms = %d'%(aid_end-aid_beg+1))

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

        delta = 1.0 #Small gap between periodic images (See Packmol manual)
        if self.is_slab:
            lig_pop_above = lig_pop//2
            lig_pop_below = lig_pop - lig_pop_above

            #Enlarge the current simbox (along z)
            self.simbox[2,0] = self._bbox_xtal[2,0] - offset - thickness_
            self.simbox[2,1] = self._bbox_xtal[2,1] + offset + thickness_

            #Bounding box above the crystal
            bbox_above_lo = [self._bbox_xtal[0,0] + delta, 
                             self._bbox_xtal[1,0] + delta,
                             self._bbox_xtal[2,1] + delta + offset]

            bbox_above_hi = [self._bbox_xtal[0,1] - delta, 
                             self._bbox_xtal[1,1] - delta,
                             self._bbox_xtal[2,1] + offset + thickness_]
            #Bbox for ligand binding group
            bbox_above_ba = bbox_above_lo + bbox_above_hi
            bbox_above_ba[5] = bbox_above_lo[2] + thickness_bm

            #Bounding box below the crystal
            bbox_below_lo = [self._bbox_xtal[0,0] + delta, 
                             self._bbox_xtal[1,0] + delta,
                             self._bbox_xtal[2,0] - offset - thickness_]

            bbox_below_hi = [self._bbox_xtal[0,1] - delta, 
                             self._bbox_xtal[1,1] - delta,
                             self._bbox_xtal[2,0] - delta - offset]
            #Bbox for ligand binding group
            bbox_below_ba = bbox_below_lo + bbox_below_hi
            bbox_below_ba[2] = bbox_below_hi[2] - thickness_bm

            #Add ligands
            ligands_to_add = []
            for i in range(num_ligand_types):
                g_above = 'inside box '
                g_above += ' '.join([str(x) for x in bbox_above_lo])
                g_above += ' '
                g_above += ' '.join([str(x) for x in bbox_above_hi])
                g_above += '\n  atoms ' + \
                        ' '.join([str(x) for x in ligand_list[i].bind_group])
                g_above += '\n    inside box ' + \
                        ' '.join([str(x) for x in bbox_above_ba])
                g_above += '\n  end atoms'
                elem = {'moltem': ligand_list[i], 'num': lig_pop_above[i],
                        'offsets': type_offsets[i], 'constraints': [g_above]}
                ligands_to_add.append(elem)

                g_below = 'inside box '
                g_below += ' '.join([str(x) for x in bbox_below_lo])
                g_below += ' '
                g_below += ' '.join([str(x) for x in bbox_below_hi])
                g_below += '\n  atoms ' + \
                        ' '.join([str(x) for x in ligand_list[i].bind_group])
                g_below += '\n    inside box ' + \
                        ' '.join([str(x) for x in bbox_below_ba])
                g_below += '\n  end atoms'
                elem = {'moltem': ligand_list[i], 'num': lig_pop_below[i],
                        'offsets': type_offsets[i], 'constraints': [g_below]}
                ligands_to_add.append(elem)
        else:
            #Enlarge the current simbox
            self.simbox[:,0] = self._bbox_xtal[:,0] - offset - thickness_
            self.simbox[:,1] = self._bbox_xtal[:,1] + offset + thickness_

            #Inner bounding box: xtal bounding box slightly increased by delta
            bbox_in_lo = self._bbox_xtal[:,0] - offset - delta
            bbox_in_hi = self._bbox_xtal[:,1] + offset + delta
            bbox_in_bm = bbox_in_lo.tolist() + bbox_in_hi.tolist()

            #Outer bounding box: simbox box slightly reduced by delta
            bbox_out_lo = self._bbox_xtal[:,0] - offset - thickness_
            bbox_out_hi = self._bbox_xtal[:,1] + offset + thickness_
            bbox_out_bm = (bbox_in_lo - thickness_bm).tolist() + \
                            (bbox_in_hi + thickness_bm).tolist() 
            
            #Add ligands
            ligands_to_add = []
            for i in range(num_ligand_types):
                g_in = 'inside box '
                g_in += ' '.join([str(x) for x in bbox_out_lo])
                g_in += ' '
                g_in += ' '.join([str(x) for x in bbox_out_hi])
                g_out = 'outside box '
                g_out += ' '.join([str(x) for x in bbox_in_lo])
                g_out += ' '
                g_out += ' '.join([str(x) for x in bbox_in_hi])
                g_bm  = '  atoms ' + \
                        ' '.join([str(x) for x in ligand_list[i].bind_group])
                g_bm += '\n    inside box ' + \
                        ' '.join([str(x) for x in bbox_out_bm])
                g_bm += '\n    outside box ' + \
                        ' '.join([str(x) for x in bbox_in_bm])
                g_bm += '\n  end atoms'
                elem = {'moltem': ligand_list[i],
                        'num': lig_pop[i],
                        'offsets': type_offsets[i],
                        'constraints': [g_in, g_out, g_bm]}
                ligands_to_add.append(elem)

        add_molecules(self, ligands_to_add, packmol_tol, packmol_sidemax,
                      packmol_path)
        print("Total charge after adding ligands = %g"%self.get_total_charge())

        #Create new group of ligand atoms
        atoms = range(aid_beg, aid_end+1)
        atom_types = range(at_beg, at_end+1)
        molecules = range(mid_beg, mid_end+1)
        self.set_group('ligands', atoms=atoms, atom_types=atom_types,
                       molecules=molecules)



    def solvate(self, solvent, molfrac, boxx, boxy=None, boxz=None, 
                dist=0.0, packmol_tol=2.0, packmol_sidemax=1.0e3,
                packmol_path=''):
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
        packmol_sidemax : float
            Parameter for Packmol. Default is 1000 angstrom.
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

        #Set the current simbox
        self.simbox[0,0] = -boxx_/2; self.simbox[0,1] = boxx_/2
        self.simbox[1,0] = -boxy_/2; self.simbox[1,1] = boxy_/2
        self.simbox[2,0] = -boxz_/2; self.simbox[2,1] = boxz_/2

        num_components = len(solvent)

        aid_beg = self.num_atoms + 1 #First solvent atom id
        aid_end = aid_beg  #Last solvent atom id
        mid_beg = self.num_molecules + 1 #First solvent molecule id
        mid_end = mid_beg  #Last solvent molecule id

        #Offsets in types
        num_types = np.array([self.num_atom_types, self.num_bond_types,
                        self.num_angle_types, self.num_dihedral_types,
                        self.num_improper_types], dtype=np.int32)
        type_offsets = []
        at_beg = self.num_atom_types + 1
        at_end = at_beg
        for comp in solvent:
            type_offsets.append( tuple(num_types) )
            each = comp[0]
            new_types = np.array([each.num_atom_types, each.num_bond_types,
                    each.num_angle_types, each.num_dihedral_types,
                    each.num_improper_types], dtype=np.int32)
            num_types += new_types
            at_end += (each.num_atom_types - 1)

        delta = 1.0 #Small gap between periodic images (See Packmol manual)
        if self.is_slab:
            volume_above = (self.simbox[0,1]-self.simbox[0,0]) \
                          *(self.simbox[1,1]-self.simbox[1,0]) \
                          *(self.simbox[2,1]-self._bbox_xtal[2,1])
            volume_below = (self.simbox[0,1]-self.simbox[0,0]) \
                          *(self.simbox[1,1]-self.simbox[1,0]) \
                          *(self._bbox_xtal[2,0]-self.simbox[2,0])

            #Population of molecules for each component
            comp_pop_above = np.zeros((num_components,), dtype=np.int32) 
            comp_pop_below = np.zeros((num_components,), dtype=np.int32) 
            #Calculate population of each component molecule
            for i in range(num_components):
                molwt = solvent[i][0].get_total_mass()
                ndens = solvent[i][1]*0.6023/molwt #number per angstrom^3
                pop = molfrac[i]*ndens*volume_above
                if pop < 1:
                    s = input('Zero molecules of solvent component %s. Type "Y" to '
                              'continue, any other key to exit: '%solvent[i][0].name)
                    if s != "Y":
                        raise SystemExit("Exiting ...")
                comp_pop_above[i] = int(np.rint(pop))

                pop = molfrac[i]*ndens*volume_below
                if pop < 1:
                    s = input('Zero molecules of solvent component %s. Type "Y" to '
                              'continue, any other key to exit: '%solvent[i][0].name)
                    if s != "Y":
                        raise SystemExit("Exiting ...")
                comp_pop_below[i] = int(np.rint(pop))

                mid_end = mid_end + comp_pop_above[i] + comp_pop_below[i] - 1
                aid_end = aid_end + (comp_pop_above[i]+comp_pop_below[i]) \
                            *solvent[i][0].num_atoms - 1
                print('  Number of %s molecules = %g'%(solvent[i][0].name,
                                            comp_pop_above[i]+comp_pop_below[i]))
            print('Total number of solvent molecules = %d'
                  %sum(comp_pop_above+comp_pop_below))
            print('Total number of solvent atoms = %d'%(aid_end-aid_beg+1))

            
            #Bounding box above the crystal
            bbox_above_lo = [self._bbox_xtal[0,0] + delta, 
                             self._bbox_xtal[1,0] + delta,
                             self._bbox_xtal[2,1] + delta + dist]

            bbox_above_hi = [self._bbox_xtal[0,1] - delta, 
                             self._bbox_xtal[1,1] - delta,
                             self.simbox[2,1] - delta]
            #Bounding box below the crystal
            bbox_below_lo = [self._bbox_xtal[0,0] + delta, 
                             self._bbox_xtal[1,0] + delta,
                             self.simbox[2,0] + delta]

            bbox_below_hi = [self._bbox_xtal[0,1] - delta, 
                             self._bbox_xtal[1,1] - delta,
                             self._bbox_xtal[2,0] - delta - dist]

            print(f"Packing solvents \n"
                  f" inside box ({' '.join('%g'%v for v in bbox_above_lo)})"
                  f" ({' '.join('%g'%v for v in bbox_above_hi)})\n"
                  f" inside box ({' '.join('%g'%v for v in bbox_below_lo)})"
                  f" ({' '.join('%g'%v for v in bbox_below_hi)})"
                  )

            mols_to_add = []
            for i in range(num_components):
                g_above = 'inside box '
                g_above += ' '.join(['%g'%x for x in bbox_above_lo])
                g_above += ' '
                g_above += ' '.join(['%g'%x for x in bbox_above_hi])
                elem = {'moltem': solvent[i][0], 'num': comp_pop_above[i],
                        'offsets': type_offsets[i], 'constraints': [g_above]}
                mols_to_add.append(elem)

                g_below = 'inside box '
                g_below += ' '.join(['%g'%x for x in bbox_below_lo])
                g_below += ' '
                g_below += ' '.join(['%g'%x for x in bbox_below_hi])
                elem = {'moltem': solvent[i][0], 'num': comp_pop_below[i],
                        'offsets': type_offsets[i], 'constraints': [g_below]}
                mols_to_add.append(elem)
        else:
            volume = (self.simbox[:,1]-self.simbox[:,0]).prod()
            #Population of molecules for each component
            comp_pop = np.zeros((num_components,), dtype=np.int32) 
            #Calculate population of each component molecule
            for i in range(num_components):
                molwt = solvent[i][0].get_total_mass()
                ndens = solvent[i][1]*0.6023/molwt #number per angstrom^3
                pop = molfrac[i]*ndens*volume
                if pop < 1:
                    s = input('Zero molecules of solvent component %s. Type "Y" to '
                              'continue, any other key to exit: '%solvent[i][0].name)
                    if s != "Y":
                        raise SystemExit("Exiting ...")
                comp_pop[i] = int(np.rint(pop))
                mid_end += (comp_pop[i] - 1)
                aid_end += ( comp_pop[i]*solvent[i][0].num_atoms - 1 )
                print('  Number of %s molecules = %g'%(solvent[i][0].name,
                                                     comp_pop[i]))
            print('Total number of solvent molecules = %d'%sum(comp_pop)) 
            print(f"Total number of solvent atoms = {aid_end-aid_beg+1}")

            #Inner bounding box: xtal bounding box slightly increased by delta
            bbox_in_lo = self._bbox_xtal[:,0] - dist - delta
            bbox_in_hi = self._bbox_xtal[:,1] + dist + delta

            #Outer bounding box: simbox box slightly reduced by delta
            bbox_out_lo = self.simbox[:,0] + delta
            bbox_out_hi = self.simbox[:,1] - delta

            print(f"Packing solvents \n"
                  f" outside box ({' '.join('%g'%v for v in bbox_in_lo)})"
                  f" ({' '.join('%g'%v for v in bbox_in_hi)})\n"
                  f" inside box ({' '.join('%g'%v for v in bbox_out_lo)})"
                  f" ({' '.join('%g'%v for v in bbox_out_hi)})"
                  )

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
                elem = {'moltem': solvent[i][0],
                        'num': comp_pop[i],
                        'offsets': type_offsets[i],
                        'constraints': [g_in, g_out]}
                mols_to_add.append(elem)

        add_molecules(self, mols_to_add, packmol_tol, packmol_sidemax,
                      packmol_path)

        #Create new group of solvent atoms
        atoms = range(aid_beg, aid_end+1)
        atom_types = range(at_beg, at_end+1)
        molecules = range(mid_beg, mid_end+1)
        self.set_group('solvent', atoms=atoms, atom_types=atom_types,
                       molecules=molecules)



    def adjust_charge(self):
        """
        Tweak charges to make the system electroneutral.

        """
        chge = self.get_total_charge()
        if chge != 0:
            num_atoms_tot = self.num_atoms
            cpa = chge/num_atoms_tot
            for i in range(1, self.num_atoms+1):
                self.atoms[i]['charge'] -= cpa
        print("Total charge after adjustment = %g"%self.get_total_charge())



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
            Name of file containing pair coefficients. Not read if an
            empty string or contains only spaces. Pair coefficients already 
            present in `fn_ldf` will be overwritten.

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
        #Xtal/ligand data
        if 'xtal' in self.groups:
            grp_xtal = self.groups['xtal']
            self._num_atoms_xtal = len(grp_xtal['atoms'])
            self._num_atom_types_xtal = len(grp_xtal['atom_types'])

        self._bbox_xtal = np.array([[np.inf,np.NINF],[np.inf,np.NINF],
                                    [np.inf,np.NINF]])
        for iatm in self.groups['xtal']['atoms']:
            coords = self.atoms[iatm]['coords']
            self._bbox_xtal[:,0] = np.minimum(self._bbox_xtal[:,0], coords)
            self._bbox_xtal[:,1] = np.maximum(self._bbox_xtal[:,1], coords)

        #if 'ligands' in self.groups:
        #    grp_lig = self.groups['ligands']
        #    self._num_atoms_ligands = len(grp_lig['atoms'])
        #    self._num_atom_types_ligands = len(grp_lig['atom_types'])
        #    self._num_molecules_ligands = len(grp_lig['molecules'])



        
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
