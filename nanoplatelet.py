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
            This is nelected if `self.is_slab` is true.
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
        #Xtal atom types
        at_beg = self.num_atom_types + 1 
        at_end = at_beg + xtal.get_num_atom_types() - 1
        for i in range(at_beg, at_end+1):
            name = xtal.get_atom_name(i)
            mass = xtal.get_atom_mass(i)
            self.add_atom_type(mass, name)

        #Set pair coeffs
        for i, val in enumerate(pair_coeffs):
            self.set_pair_coeff(at_beg+i, val)

        #Xtal atom ids
        aid_beg = self.num_atoms + 1 # First xtal atom id
        aid_end = aid_beg
        
        #Repeat unit spanning the x-y plane
        ru_coords = []; ru_at = [] #Type of all atoms per unit cell
        if balanced:
            imax = 2*num_mono_layers
        else:
            imax = 2*num_mono_layers + 1
        for i in range(1, imax+1):
            pos, typ = xtal.get_atoms_in_layer(i)
            ru_coords.extend(pos); ru_at.extend(typ)
        ru_coords = np.asarray(ru_coords)
        ru_at = np.asarray(ru_at)

        na = 0
        if pbc_xy:
            if unit == 'ang':
                nx = math.floor(length/xtal.a)
                ny = math.floor(width/xtal.b)
            elif unit == 'lattice':
                nx = int(length); ny = int(width)

            for j in range(ny):
                for i in range(nx):
                    origin = np.array([i*xtal.a, j*xtal.b, 0])
                    for pos, at in zip(ru_coords, ru_at):
                        self.add_atom(at, charges[at-1], pos+origin)
                        na += 1
            #Wrap in a simulation box and center at the origin
            self.fit_simbox()
            self.simbox[:,0] = 0.0
            self.simbox[0,1] = nx*xtal.a
            self.simbox[1,1] = ny*xtal.b
            if pbc_z:
                self.simbox[2,1] = (num_mono_layers//2)*xtal.c
            r = self.simbox.sum(axis=1)/2
            self.translate(-r)
        else:
            if unit == 'ang':
                hlen = length/2; hwid = width/2
            elif unit == 'lattice':
                hlen = xtal.a*length/2
                hwid = xtal.b*width/2
        
            #Bounding box centered at the origin
            hlen_bbox = math.hypot(hlen, hwid) #Half length of the bounding box
            #Number of unit cells along x, y, and z directions
            nx = math.ceil(hlen_bbox/xtal.a) + 2 #add two additonal unit cells,
                                                      #just in case
            ny = nx
            nc = (2*nx)*(2*ny) #Total number of unit cells
            na_ru = ru_coords.shape[0] #Number of atoms per repeat unit
            na_all = nc*na_ru #Total number of atoms

            coords = np.zeros((na_all,3), dtype=np.float64)
            atm_types = np.zeros((na_all,), dtype=np.int32)
            icoord = 0
            blc = np.array([-nx*xtal.a, -ny*xtal.b, 0]) #Bottom left corner
            for j in range(2*ny):
                for i in range(2*nx):
                    ibeg = icoord; iend = icoord + na_ru
                    origin = blc + np.array([i*xtal.a, j*xtal.b, 0])
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
                    na += 1
            #Wrap in a simulation box and center at the origin
            self.fit_simbox(sep=0)
            if pbc_z:
                self.simbox[2,1] = self.simbox[2,0] \
                        + (num_mono_layers//2)*xtal.c
            r = self.simbox.sum(axis=1)/2
            self.translate(-r)

        aid_end += (na - 1) #Id of the last xtal atom

        #Atoms ids of the top & bottom crystal surface. Atoms on each surface
        #are of the same type.
        xtal_coords = self.get_atom_coords(list(range(aid_beg, aid_end+1)))
        zcoords = xtal_coords[:,2]
        zlo = zcoords.min(); zhi = zcoords.max()

        top_mask = np.isclose(zcoords, zhi)
        aids_xst = (aid_beg + np.nonzero(top_mask)[0]).tolist()
        at_xst = self.atoms[aids_xst[0]]['type']

        bot_mask = np.isclose(zcoords, zlo)
        aids_xsb = (aid_beg + np.nonzero(bot_mask)[0]).tolist()
        at_xsb = self.atoms[aids_xsb[0]]['type']

        #Add groups
        self.set_group('Xtal', atoms=range(aid_beg, aid_end+1),
                atom_types=range(at_beg, at_end+1))
        self.set_group('XtalSurfTop', atom_types=range(at_xst, at_xst+1),
                        atoms=aids_xst)
        self.set_group('XtalSurfBot', atom_types=range(at_xsb, at_xsb+1),
                        atoms=aids_xsb)



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

        lig_size_max = 0
        num_ligand_types = len(ligand_list)
        lig_charge = np.zeros((num_ligand_types,))
        lig_pop = np.zeros((num_ligand_types,), dtype=np.int32)
        lig_pop_top = np.zeros((num_ligand_types,), dtype=np.int32)
        lig_pop_bot = np.zeros((num_ligand_types,), dtype=np.int32)

        aid_beg = self.num_atoms + 1 # First ligand atom id
        aid_end = aid_beg  # Last ligand atom id
        mid_beg = self.num_molecules + 1 #First ligand molecule id
        mid_end = mid_beg  # Last ligand molecule id
        aids_lt = []; aids_lb = []; aidsc = aid_beg #Lists & counter
        mids_lt = []; mids_lb = []; midsc = mid_beg #Lists & counter

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
            if self.is_slab:
                lig_pop_top[i] = int(np.rint(pop))//2
                v = list(range(lig_pop_top[i]*ligand_list[i].num_atoms))
                aids = [aidsc+i for i in v]
                aids_lt.extend(aids)
                aidsc += len(v)

                v = list(range(lig_pop_top[i]))
                mids = [midsc+i for i in v]
                mids_lt.extend(mids)
                midsc += len(v)

                lig_pop_bot[i] = pop - lig_pop_top[i]
                v = list(range(lig_pop_bot[i]*ligand_list[i].num_atoms))
                aids = [aidsc+i for i in v]
                aids_lb.extend(aids)
                aidsc += len(v)

                v = list(range(lig_pop_bot[i]))
                mids = [midsc+i for i in v]
                mids_lb.extend(mids)
                midsc += len(v)
                print(f"  Number of {ligand_list[i].name} molecules"
                      f" = {lig_pop_top[i]+lig_pop_bot[i]}")
            else:
                lig_pop[i] = int(np.rint(pop))
                aid_end += ( lig_pop[i]*ligand_list[i].num_atoms - 1 )
                mid_end += (lig_pop[i] - 1)
                print(f"  Number of {ligand_list[i].name} molecules"
                      f" = {lig_pop[i]}")
        if self.is_slab:
            print(f"  Total number of ligand molecules"
                  f" = {len(mids_lt)+len(mids_lb)}") 
            print(f"  Total number of ligand atoms"
                  f" = {len(aids_lt)+len(aids_lb)}")
            #raise SystemExit()
        else:
            print(f"  Total number of ligand molecules = {sum(lig_pop)}") 
            print(f"  Total number of ligand atoms = {aid_end-aid_beg+1}")

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

        #Xtal bounding box
        rlo, rhi = self.get_bbox(self.groups['Xtal']['atoms'])
        bbox_xtal = np.asarray((rlo,rhi)).transpose()

        delta = 1.0 #Small gap between periodic images (See Packmol manual)
        if self.is_slab:
            #Enlarge the current simbox (along z)
            self.simbox[2,0] = bbox_xtal[2,0] - offset - thickness_
            self.simbox[2,1] = bbox_xtal[2,1] + offset + thickness_

            #Bounding box above the crystal
            bbox_top_lo = [bbox_xtal[0,0] + delta, 
                           bbox_xtal[1,0] + delta,
                           bbox_xtal[2,1] + delta + offset]

            bbox_top_hi = [bbox_xtal[0,1] - delta, 
                           bbox_xtal[1,1] - delta,
                           bbox_xtal[2,1] + offset + thickness_]
            #Bbox for ligand binding group
            bbox_top_ba = bbox_top_lo + bbox_top_hi
            bbox_top_ba[5] = bbox_top_lo[2] + thickness_bm

            #Bounding box below the crystal
            bbox_bot_lo = [bbox_xtal[0,0] + delta, 
                           bbox_xtal[1,0] + delta,
                           bbox_xtal[2,0] - offset - thickness_]

            bbox_bot_hi = [bbox_xtal[0,1] - delta, 
                           bbox_xtal[1,1] - delta,
                           bbox_xtal[2,0] - delta - offset]
            #Bbox for ligand binding group
            bbox_bot_ba = bbox_bot_lo + bbox_bot_hi
            bbox_bot_ba[2] = bbox_bot_hi[2] - thickness_bm

            #Add ligands
            ligands_to_add = []
            for i in range(num_ligand_types):
                g_top = 'inside box '
                g_top += ' '.join([str(x) for x in bbox_top_lo])
                g_top += ' '
                g_top += ' '.join([str(x) for x in bbox_top_hi])
                g_top += '\n  atoms ' + \
                        ' '.join([str(x) for x in ligand_list[i].bind_group])
                g_top += '\n    inside box ' + \
                        ' '.join([str(x) for x in bbox_top_ba])
                g_top += '\n  end atoms'
                elem = {'moltem': ligand_list[i], 'num': lig_pop_top[i],
                        'offsets': type_offsets[i], 'constraints': [g_top]}
                ligands_to_add.append(elem)

                g_bot = 'inside box '
                g_bot += ' '.join([str(x) for x in bbox_bot_lo])
                g_bot += ' '
                g_bot += ' '.join([str(x) for x in bbox_bot_hi])
                g_bot += '\n  atoms ' + \
                        ' '.join([str(x) for x in ligand_list[i].bind_group])
                g_bot += '\n    inside box ' + \
                        ' '.join([str(x) for x in bbox_bot_ba])
                g_bot += '\n  end atoms'
                elem = {'moltem': ligand_list[i], 'num': lig_pop_bot[i],
                        'offsets': type_offsets[i], 'constraints': [g_bot]}
                ligands_to_add.append(elem)

            #Create ligand group
            atoms = range(aid_beg, aidsc)
            atom_types = range(at_beg, at_end+1)
            molecules = range(mid_beg, midsc)

            self.set_group('LigandsTop', atoms=aids_lt, atom_types=atom_types,
                       molecules=mids_lt)
            self.set_group('LigandsBot', atoms=aids_lb, atom_types=atom_types,
                       molecules=mids_lb)
            self.set_group('Ligands', atoms=atoms, atom_types=atom_types,
                           molecules=molecules)
        else:
            #Enlarge the current simbox
            self.simbox[:,0] = bbox_xtal[:,0] - offset - thickness_
            self.simbox[:,1] = bbox_xtal[:,1] + offset + thickness_

            #Inner bounding box: xtal bounding box slightly increased by delta
            bbox_in_lo = bbox_xtal[:,0] - offset - delta
            bbox_in_hi = bbox_xtal[:,1] + offset + delta
            bbox_in_bm = bbox_in_lo.tolist() + bbox_in_hi.tolist()

            #Outer bounding box: simbox box slightly reduced by delta
            bbox_out_lo = bbox_xtal[:,0] - offset - thickness_
            bbox_out_hi = bbox_xtal[:,1] + offset + thickness_
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

            #Create ligand group
            atoms = range(aid_beg, aid_end+1)
            atom_types = range(at_beg, at_end+1)
            molecules = range(mid_beg, mid_end+1)
            self.set_group('Ligands', atoms=atoms, atom_types=atom_types,
                           molecules=molecules)

        add_molecules(self, ligands_to_add, packmol_tol, packmol_sidemax,
                      packmol_path)
        print("Total charge after adding ligands = %g"%self.get_total_charge())




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

        #Xtal bounding box
        rlo, rhi = self.get_bbox(self.groups['Xtal']['atoms'])
        bbox_xtal = np.asarray((rlo,rhi)).reshape((3,2))

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
            bbox_above_lo = [bbox_xtal[0,0] + delta, 
                             bbox_xtal[1,0] + delta,
                             bbox_xtal[2,1] + delta + dist]

            bbox_above_hi = [bbox_xtal[0,1] - delta, 
                             bbox_xtal[1,1] - delta,
                             self.simbox[2,1] - delta]
            #Bounding box below the crystal
            bbox_below_lo = [bbox_xtal[0,0] + delta, 
                             bbox_xtal[1,0] + delta,
                             self.simbox[2,0] + delta]

            bbox_below_hi = [bbox_xtal[0,1] - delta, 
                             bbox_xtal[1,1] - delta,
                             bbox_xtal[2,0] - delta - dist]

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
            bbox_in_lo = bbox_xtal[:,0] - dist - delta
            bbox_in_hi = bbox_xtal[:,1] + dist + delta

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
        self.set_group('Solvent', atoms=atoms, atom_types=atom_types,
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
            if soften=='Ligands' or soften=='Solvent':
                soft_atom_types = self.groups[soften]['atom_types']
            if soften=='both':
                soft_atom_types = list(self.groups['Ligands']['atom_types']) \
                                + list(self.groups['Solvent']['atom_types'])
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
                #Remove comments and split
                stripped = each.strip(' \n')
                if stripped.startswith('#'):
                    continue
                else:
                    non_comment = each.split("#",1)[0]
                    if non_comment != '':
                        words = non_comment.split()
                    else:
                        continue
                at_i = int(words[1]); at_j = int(words[2])
                if at_i == at_j:
                    params = [float(x) for x in words[3:]]
                    self.set_pair_coeff(at_i, params)
        #Xtal/ligand data
        if 'Xtal' in self.groups:
            grp_xtal = self.groups['Xtal']
            self._num_atoms_xtal = len(grp_xtal['atoms'])
            self._num_atom_types_xtal = len(grp_xtal['atom_types'])

        self._bbox_xtal = np.array([[np.inf,np.NINF],[np.inf,np.NINF],
                                    [np.inf,np.NINF]])
        for iatm in self.groups['Xtal']['atoms']:
            coords = self.atoms[iatm]['coords']
            self._bbox_xtal[:,0] = np.minimum(self._bbox_xtal[:,0], coords)
            self._bbox_xtal[:,1] = np.maximum(self._bbox_xtal[:,1], coords)



        
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
