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
        read_mol_grp, write_grp_lammps, add_molecules


class Brush(Configuration):
    def __init__(self, is_slab, slab_pos):
        """
        The simulation box is centered in the xy-plane with its bottom surface
        at z = 0.

        Parameters
        ----------
        is_slab : bool
            Whether the crystal is a slab or not. A slab is periodic in the x &
            y directions and parallel to the xy-plane.
        zpos : {'bot', 'mid'}
            Whether the slab is at the bottom or at the middle of the box. If `bot`, the
            lower z-bound of the simulation box will be at z = 0. For all other
            cases, the simulation box is centered at the origin = (0,0,0) and
            the crystal is placed at the center of the box.
            
        """
        super().__init__()
        self.is_slab = is_slab
        self.slab_pos = slab_pos
        self.apl = None   #Area per ligand
        self.xlpar = None



    def add_xtal(self, lx, ly, lz, xlpar, xa_mass, xa_eps, xa_sigma, xa_rcut,
                 out_dir=None):
        """
        Adds a crystal normal to the z-axis onto whose surface(s) the ligands will
        be grafted.

        Parameters
        ----------
        lx : float
            Extent of the crystal along the x-direction
        ly : float
            Extent of the crystal along the y-direction
        lz : float
            Extent of the crystal along the z-direction
        xlpar : float
            Lattice parameter of the crystal
        xa_mass : float
            Mass of the wall atoms
        xa_eps : float
            LJ epsilon parameter for the wall atoms
        xa_sigma : float
            LJ sigma parameter for the wall atoms
        xa_rcut : float or None
            LJ curoff distance for the wall atoms

        """
        if out_dir is None:
            odir = os.getcwd()
        else:
            path = os.path.expanduser(out_dir)
            if os.path.exists(path):
                odir = path
            else:
                warnings.warn(f"Directory {out_dir} does not exist, using"
                              " current working directory.") 
                odir = os.getcwd()

        self.xlpar = xlpar

        hlx = 0.5*self.xlpar*math.ceil(lx/self.xlpar)
        hly = 0.5*self.xlpar*math.ceil(ly/self.xlpar)
        hlz = 0.5*self.xlpar*math.ceil(lz/self.xlpar)
        if self.is_slab and (self.slab_pos == 'bot'):
            lo = [-hlx, -hly, 0]; hi = [hlx, hly, 2*hlz]
        else:
            lo = [-hlx, -hly, -hlz]; hi = [hlx, hly, hlz]

        xa_coords = get_lattice_points('fcc', self.xlpar, lo, hi, boundary='ppc')

        if self.is_slab:
            self.add_simbox(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
            print(f"Surface area = {(hi[0]-lo[0])*(hi[1]-lo[1]):g} A^2",
                    file=open(f"{odir}/info.txt", 'a'))
        else:
            center = xa_coords.mean(axis=0)
            xa_coords[:,:] -= center
            lb = xa_coords.min(axis=0)
            ub = xa_coords.max(axis=0)
            self.add_simbox(lb[0], ub[0], lb[1], ub[1], lb[2], ub[2])

        #Crystal atoms types: Assuming only a single atom type
        at_beg = self.num_atom_types + 1 
        at_end = at_beg
        iat = self.add_atom_type(mass=xa_mass, name='XA')
        if xa_rcut is None:
            self.set_pair_coeff(iat, [xa_eps, xa_sigma])
        else:
            self.set_pair_coeff(iat, [xa_eps, xa_sigma, xa_rcut])

        #Crystal atoms ids
        aid_beg = self.num_atoms + 1 # First crystal atom id
        aid_end = aid_beg + xa_coords.shape[0] - 1 #Last crystal atom id

        #Add atom positions
        for each in xa_coords:
            self.add_atom(iat, 0.0, each)

        #Atoms ids of the top & bottom crystal surface
        xa_zcoords = xa_coords[:,2]
        zlo = xa_zcoords.min(); zhi = xa_zcoords.max()

        top_mask = np.isclose(xa_zcoords, zhi)
        aids_xst = (aid_beg + np.nonzero(top_mask)[0]).tolist()
        ats_xst = []
        for aid in aids_xst:
            at = self.atoms[aid]['type']
            if at not in ats_xst:
                ats_xst.append(at)

        bot_mask = np.isclose(xa_zcoords, zlo)
        aids_xsb = (aid_beg + np.nonzero(bot_mask)[0]).tolist()
        ats_xsb = []
        for aid in aids_xsb:
            at = self.atoms[aid]['type']
            if at not in ats_xsb:
                ats_xsb.append(at)

        #Add groups
        atoms = range(aid_beg, aid_end+1)
        atom_types = range(at_beg, at_end+1)
        self.set_group('Xtal', atom_types=atom_types, atoms=atoms)
        self.set_group('XtalSurfTop', atom_types=ats_xst, atoms=aids_xst)
        self.set_group('XtalSurfBot', atom_types=ats_xsb, atoms=aids_xsb)



    def add_ligands(self, ligands, pop_ratio, r0, balance_charge, lattice, apl,
                    out_dir=None):
        """
        Adds ligands to a bare crystal. The resulting configuration will have
        an enlarged simulation box to include the ligands.

        Parameters
        ----------

        ligands: list of LigandMolecule
            Ligand molecules to add.
        pop_ratio : list of int
            In case of multiple ligand molecules, the ratio of ligand population
            for each type. Only considers monovalent ligands. E.g., for two
            ligand types in 2:1 ratio, use [2, 1].
        r0 : float
            Distance of the ligand heads from the crystal surface
        balance_charge : bool
            Determine ligand population from neutralizing the total charge on
            the crystal.
        lattice : {'sc', 'bcc', None}
            Lattice for arranging the ligand molecule graft points. If None,
            ligands are distributed randomly.
        apl : float or None
            Area per ligand in angstrom^2. Ignored if `balance_charge` is True.

        """
        if out_dir is None:
            odir = os.getcwd()
        else:
            path = os.path.expanduser(out_dir)
            if os.path.exists(path):
                odir = path
            else:
                warnings.warn(f"Directory {out_dir} does not exist, using"
                              " current working directory.") 
                odir = os.getcwd()

        rng = np.random.default_rng()

        #Add ligand molecule types
        for each in ligands:
            self.add_molecule_type(each)

        #Atom types for ligands
        ats_ligands = self._get_ligand_atom_types(ligands)

        #Type offsets for ligands
        type_offsets = self._get_ligand_type_offsets(ligands)

        #Ligand population
        if balance_charge:
            pop = self._ligpop_from_charge(ligands, pop_ratio)
        else:
            pop = self._ligpop_from_area(ligands, pop_ratio, apl, out_dir)

        #Assign ligands to surfaces
        surfaces = self._ligands_to_surfaces(ligands, pop, r0, lattice)

        #Add ligand molecules
        mid_beg = self.num_molecules + 1
        for each in surfaces.values():
            normal = each['normal']
            pop = each['pop']
            gp_indx = self._choose(pop)
            for i, lg in enumerate(ligands):
                mrec = {'moltem': lg, 'num': pop[i], 'offsets': type_offsets[i]}
                mids = add_molecules(self, [mrec], packmol_path=None)
                mid_end = mids[0][-1]
                gpi = gp_indx[i]
                lg.align(lg.head, lg.tail, normal)
                coords = np.zeros((lg.num_atoms,3))
                for j,mid in enumerate(mids[0]): #mids is a list of ranges
                    graft_point = each['graft_points'][gpi[j],:]
                    angle = 2*math.pi*rng.random()
                    p = lg.get_atom_coords([lg.head])
                    lg.rotate(angle, normal, p[0:], coords)
                    dr = graft_point - coords[lg.head-1,:]
                    coords += dr
                    atm_beg = self.molecules[mid]['atm_beg']
                    atm_end = self.molecules[mid]['atm_end']
                    self.set_atom_coords(range(atm_beg, atm_end+1), coords)

        #Create group for ligands
        aid_beg = self.molecules[mid_beg]['atm_beg']
        aid_end = self.molecules[mid_end]['atm_end']
        self.set_group('Ligands', atom_types=ats_ligands, 
                       atoms=range(aid_beg, aid_end+1),
                       molecules=range(mid_beg, mid_end+1))

        #Create group for ligand heads
        aids_head = []
        for mid in self.groups['Ligands']['molecules']:
            name = self.molecules[mid]['name']
            aid = self.molecules[mid]['atm_beg'] \
                    + self.molecule_types[name].head - 1
            aids_head.append(aid)
        self.set_group('LigandHeads', atoms=aids_head)
        write_grp_lammps(self, f"{odir}/GrpLigHead.lmp", 'LigandHeads', 'LigHead')

        #Update simulation box
        if self.is_slab:
            lo, hi = self.get_bbox(self.groups['Ligands']['atoms'])
            if self.slab_pos == 'bot':
                self.simbox[2,1] = hi[2]
            else:
                self.simbox[2,1] = hi[2]; self.simbox[2,0] = -self.simbox[2,1]
        else:
            self.fit_simbox()



    def _get_ligand_atom_types(self, ligands):
        """
        Atom types of ligand atoms.

        """
        at_beg = self.num_atom_types + 1
        at_end = at_beg
        for each in ligands:
            at_end += (each.num_atom_types - 1)
        return range(at_beg, at_end+1)


    def _get_ligand_type_offsets(self, ligands):
        """
        Type offsets for each ligand species.

        """
        num_types = np.array([self.num_atom_types, self.num_bond_types,
                            self.num_angle_types, self.num_dihedral_types,
                            self.num_improper_types], dtype=np.int32)
        type_offsets = []
        for each in ligands:
            type_offsets.append( tuple(num_types) )
            new_types = np.array([each.num_atom_types, each.num_bond_types,
                            each.num_angle_types, each.num_dihedral_types,
                            each.num_improper_types], dtype=np.int32)
            num_types += new_types
        return type_offsets


    def _choose(self, pop):
        """
        Assign types to each ligand molecule graft point (needed for random
        mixture of different ligand molecules)

        Parameters
        ----------
        pop : int, array-like
            Population of each ligand species

        Returns
        -------
        1-D numpy array of ints

        """
        rng = np.random.default_rng()
        pop_tot = sum(pop)
        indices = list(range(pop_tot))
        chosen = []
        for i,n in enumerate(pop):
            indx = rng.choice(indices, size=n, replace=False, shuffle=False)
            chosen.append(indx) 
            for j in indx:
                indices.remove(j)
        return chosen


    def _ligpop_from_area(self, ligands, pop_ratio, apl, out_dir):
        """
        Determines population of each ligand species based on surface area.

        """
        #Crystal bounding box 
        lo, hi = self.get_bbox(self.groups['Xtal']['atoms'])
        #Total surface area
        if self.is_slab:
            area_tot = (hi[0]-lo[0])*(hi[1]-lo[1])
            if self.slab_pos != 'bot':
                area_tot *= 2
        else:
            dr = hi - lo
            area_tot = 2*(dr[0]*dr[1] + dr[1]*dr[2] + dr[2]*dr[0])
        #Population of each ligand species
        num_ligands = area_tot/apl #This is a float
        s = sum(pop_ratio); pop_frac = [x/s for x in pop_ratio]
        pop = [round(num_ligands*f) for f in pop_frac]
        self.apl = area_tot/sum(pop)
        print(f"Area/ligand = {self.apl:f} A^2\n"
              f"Brush density = {1/self.apl:f}/A^2", file=open(f"{out_dir}/info.txt", 'a'))
        return pop


    def _ligpop_from_charge(self, ligands, pop_ratio):
        """
        Determines population of each ligand species to balance the charge of
        the crystal.

        """
        #Crystal bounding box 
        lo, hi = self.get_bbox(self.groups['Xtal']['atoms'])
        #Total surface area
        if self.is_slab:
            area_tot = (hi[0]-lo[0])*(hi[1]-lo[1])
            if self.slab_pos != 'bot':
                area_tot *= 2
        else:
            dr = hi - lo
            area_tot = 2*(dr[0]*dr[1] + dr[1]*dr[2] + dr[2]*dr[0])

        totchg = self.get_total_charge()
        ligchg = [each.get_total_charge() for each in ligands]
        s = sum(pop_ratio); pop_frac = [x/s for x in pop_ratio]
        pop_tot = abs(totchg)/sum([abs(f*c) for f,c in zip(pop_frac,ligchg)])
        pop = [round(pop_tot*f) for f in pop_frac]

        residual_charge = totchg + sum([p*c for p,c in zip(pop,ligchg)])
        print (f"Residual charge after adding ligands = {residual_charge}")

        self.apl = area_tot/sum(pop)
        print(f"Area/ligand = {self.apl:f} A^2\n"
              f"Brush density = {1/self.apl:f}/A^2")
        return pop


    def _get_graft_points(self, n, lo, hi, lattice, boundary):
        """

        """
        rng = np.random.default_rng()

        if math.isclose(hi[0], lo[0]):
            area = (hi[1]-lo[1])*(hi[2]-lo[2])
        elif math.isclose(hi[1], lo[1]):
            area = (hi[2]-lo[2])*(hi[0]-lo[0])
        elif math.isclose(hi[2], lo[2]):
            area = (hi[0]-lo[0])*(hi[1]-lo[1])

        apl = area/n
        if lattice == 'sc':
            lpar = math.sqrt(apl)
        elif lattice == 'bcc':
            lpar = math.sqrt(2*apl)

        sites = get_lattice_points(lattice, lpar, lo, hi, boundary='ppp',
                                    num_sites=n)
        graft_points = rng.choice(sites, size=n, replace=False)
        return graft_points


    def _ligands_to_surfaces(self, ligands, pop, r0, lattice):
        """

        """
        surfaces = {} #Keys: 'name', Values= {'normal', 'pop', 'graft_points'}
        #Crystal bounding box 
        xtal_lo, xtal_hi = self.get_bbox(self.groups['Xtal']['atoms'])
        #Bounding box for ligand surface area
        if self.is_slab:
            if self.slab_pos == 'bot':
                lo = [self.simbox[0,0], self.simbox[1,0], xtal_hi[2]+r0]
                hi = [self.simbox[0,1], self.simbox[1,1], xtal_hi[2]+r0]
                gp = self._get_graft_points(sum(pop), lo, hi, lattice, 'ppp')
                surfaces['top'] = {'normal': np.array([0,0,1]), 'pop': pop,
                                    'graft_points': gp}
            else:
                #Top surface
                lo = [self.simbox[0,0], self.simbox[1,0], xtal_hi[2]+r0]
                hi = [self.simbox[0,1], self.simbox[1,1], xtal_hi[2]+r0]
                pop_top = [x//2 for x in pop]
                n = sum(pop_top)
                gp = self._get_graft_points(n, lo, hi, lattice, 'ppp')
                surfaces['top'] = {'normal': np.array([0,0,1]),
                                   'pop': pop_top, 'graft_points': gp}
                #Bottom surface
                lo = [self.simbox[0,0], self.simbox[1,0], xtal_lo[2]-r0]
                hi = [self.simbox[0,1], self.simbox[1,1], xtal_lo[2]-r0]
                pop_bot = [x-y for x,y in zip(pop,pop_top)]
                n = sum(pop_bot)
                gp = self._get_graft_points(n, lo, hi, lattice, 'ppp')
                surfaces['bot'] = {'normal': np.array([0,0,-1]),
                                   'pop': pop_bot, 'graft_points': gp}
        else:
            dx = xtal_hi[0] - xtal_lo[0] + 2*r0
            dy = xtal_hi[1] - xtal_lo[1] + 2*r0
            dz = xtal_hi[2] - xtal_lo[2] + 2*r0
            areas = [2*dy*dz, 2*dx*dz, 2*dx*dy]
            area_tot = sum(areas)

            ind = np.argsort(areas)
            pop_ = [[], [], []]
            pop_[ind[0]] = [int(areas[ind[0]]*x/area_tot) for x in pop]
            pop_[ind[1]] = [int(areas[ind[1]]*x/area_tot) for x in pop]
            pop_[ind[2]] = [z-x-y for 
                            x,y,z in zip(pop_[ind[0]], pop_[ind[1]], pop)]
            popx = pop_[0]; popy = pop_[1]; popz = pop_[2]

            #Surface normal to the x-axis: Left 
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_lo[0]-r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_left = [x//2 for x in popx]
            n = sum(pop_left)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['left'] = {'normal': np.array([-1,0,0]),
                                'pop': pop_left, 'graft_points': gp}

            #Surface normal to the x-axis: Right 
            lo = [xtal_hi[0]+r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_right = [x-y for x,y in zip(popx,pop_left)]
            n = sum(pop_right)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['right'] = {'normal': np.array([1,0,0]),
                                'pop': pop_right, 'graft_points': gp}

            #Surface normal to the y-axis: Front
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_lo[1]-r0, xtal_hi[2]+r0]
            pop_front = [x//2 for x in popy]
            n = sum(pop_front)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['front'] = {'normal': np.array([0,-1,0]),
                                'pop': pop_front, 'graft_points': gp}

            #Surface normal to the y-axis: Back
            lo = [xtal_lo[0]-r0, xtal_hi[1]+r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_back = [x-y for x,y in zip(popy,pop_front)]
            n = sum(pop_back)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['back'] = {'normal': np.array([0,1,0]),
                                'pop': pop_back, 'graft_points': gp}

            #Surface normal to the z-axis: Bottom
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_lo[2]-r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_lo[2]-r0]
            pop_bot = [x//2 for x in popz]
            n = sum(pop_bot)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['bot'] = {'normal': np.array([0,0,-1]),
                               'pop': pop_bot, 'graft_points': gp}

            #Surface normal to the z-axis: Top
            lo = [xtal_lo[0]-r0, xtal_lo[1]-r0, xtal_hi[2]+r0]
            hi = [xtal_hi[0]+r0, xtal_hi[1]+r0, xtal_hi[2]+r0]
            pop_top = [x-y for x,y in zip(popz,pop_bot)]
            n = sum(pop_top)
            gp = self._get_graft_points(n, lo, hi, lattice, 'nnn')
            surfaces['top'] = {'normal': np.array([0,0,1]),
                               'pop': pop_top, 'graft_points': gp}
        return surfaces



    def add_ligand_one(self, ligand, r0, both_sides):
        """
        Adds a single ligand molecule at the center of the tethering plane.

        Parameters
        ----------

        ligand : LigandMolecule
            Ligand molecule to add.
        r0 : float
            Distance along the z-axis from the slab surface to the ligand head.
        both_sides : bool
            Whether to add ligands on both the top & bottom surfaces of the slab

        """
        if not self.is_slab:
            raise SystemExit("Not a slab. Exiting ...")
        else:
            if self.slab_pos=='bot' and both_sides:
                raise SystemExit("Cannot put ligands on both sides for a slab "
                            "positioned at the bottom of the box. Exiting ...")

        num_ligands = 2 if both_sides else 1
        
        self.apl = (self.simbox[0,1]-self.simbox[0,0]) \
                    *(self.simbox[1,1]-self.simbox[1,0])

        #Add ligand molecule types
        self.add_molecule_type(ligand)

        aid_beg = self.num_atoms + 1 # First ligand atom id
        mid_beg = self.num_molecules + 1 #First ligand molecule id
        aid_end = aid_beg + num_ligands*ligand.num_atoms - 1 # Last ligand atom id
        mid_end = mid_beg + num_ligands - 1 # Last ligand molecule id

        na_ligands = aid_end - aid_beg + 1
        print(f"  Total number of ligand molecules = {num_ligands}") 
        print(f"  Total number of ligand atoms = {na_ligands}")

        #Offsets in types
        num_types = np.array([self.num_atom_types, self.num_bond_types,
                        self.num_angle_types, self.num_dihedral_types,
                        self.num_improper_types], dtype=np.int32)
        type_offsets = tuple(num_types)
        at_beg = self.num_atom_types + 1
        at_end = at_beg + ligand.num_atom_types - 1

        #Add ligands
        ligands_to_add = [{'moltem': ligand, 'num': num_ligands,
                           'offsets': type_offsets}]
        add_molecules(self, ligands_to_add, packmol_path=None)

        #Create new group of ligand atoms
        atoms = range(aid_beg, aid_end+1)
        atom_types = range(at_beg, at_end+1)
        molecules = range(mid_beg, mid_end+1)
        self.set_group('Ligands', atoms=atoms, atom_types=atom_types,
                       molecules=molecules)

        #Grafting ligands
        lo, hi = self.get_bbox(self.groups['XtalSurfTop']['atoms'])
        graft_point = np.array([0,0, hi[2]])

        rng = np.random.default_rng()
        zhat = np.array([0,0,1])

        angle = 2*math.pi*rng.random()
        p = ligand.get_atom_coords([ligand.head])
        ligand.rotate(angle, zhat, p[0:])
        ligand.translate_atom(ligand.head, graft_point)

        mol_id = self.groups['Ligands']['molecules'][0]
        atm_beg = self.molecules[mol_id]['atm_beg']
        atm_end = self.molecules[mol_id]['atm_end']
        coords = ligand.get_atom_coords()
        self.set_atom_coords(range(atm_beg, atm_end+1), coords)

        if both_sides:
            angle = 2*math.pi*rng.random()
            p = ligand.get_atom_coords([ligand.head])
            ligand.rotate(angle, zhat, p[0:])
            ligand.translate_atom(ligand.head, graft_point)

            mol_id = self.groups['Ligands']['molecules'][1]
            atm_beg = self.molecules[mol_id]['atm_beg']
            atm_end = self.molecules[mol_id]['atm_end']
            coords = ligand.get_atom_coords()
            coords[:,2] *= -1
            self.set_atom_coords(range(atm_beg, atm_end+1), coords)



    def solvate(self, solvent, box_params, packmol_tol=2.0,
                packmol_sidemax=1.0e3, packmol_path=None):
        """
        Adds solvent molecules around a nanoplatelet.

        Parameters
        ----------
        solvent : tuple
            Solvent molecule to add. The tuple is (Molecule, density), where density
            is in g/mL.
        box_params : dict
            Parameters for changing the simulation box.
        packmol_tol : float
            Tolerance for Packmol. Default is 2 angstrom.
        packmol_sidemax : float
            Parameter for Packmol. Default is 1000 angstrom.
        packmol_path : str or pathlib.Path or None
            Path to the packmol executable. If None, Packmol will not be used. In
            this case all added molecules will have their atom positions set to
            zero.

        """
        delta = 2.0 #Small gap between periodic images (See Packmol manual)

        #Change simulation box size
        if 'boxx' in box_params:
            if not self.is_slab:
                self.simbox[0,1] = box_params['boxx']/2
                self.simbox[0,0] = -self.simbox[0,1]
        if 'boxy' in box_params:
            if not self.is_slab:
                self.simbox[1,1] = box_params['boxy']/2
                self.simbox[1,0] = -self.simbox[1,1]
        if 'boxz' in box_params:
            if self.is_slab and self.slab_pos=='bot':
                self.simbox[2,1] = box_params['boxz']
            else:
                self.simbox[2,1] = box_params['boxz']/2
                self.simbox[2,0] = -self.simbox[2,1]

        if 'Xtal' in self.groups:
            xtal_lo, xtal_hi = self.get_bbox(self.groups['Xtal']['atoms'])
            if 'sepx' in box_params:
                if not self.is_slab:
                    self.simbox[0,1] = xtal_hi[0] + box_params['sepx']
                    self.simbox[0,0] = -self.simbox[0,1]
            if 'sepy' in box_params:
                if not self.is_slab:
                    self.simbox[1,1] = xtal_hi[1] + box_params['sepy']
                    self.simbox[1,0] = -self.simbox[1,1]
            if 'sepz' in box_params:
                if self.is_slab and self.slab_pos=='bot':
                    self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
                else:
                    self.simbox[2,1] = xtal_hi[2] + box_params['sepz']
                    self.simbox[2,0] = -self.simbox[2,1]
            if 'make_cubic' in box_params and box_params['make_cubic']:
                if self.is_slab and \
                        math.isclose(self.simbox[0,1], self.simbox[1,1]):
                    if self.slab_pos == 'bot':
                        self.simbox[2,1] = 2*self.simbox[0,1]
                    else:
                        self.simbox[2,1] = self.simbox[0,1]
                        self.simbox[2,0] = -self.simbox[2,1]
                else:
                    v = self.simbox[:,1].max()
                    self.simbox[:,1] = v
                    self.simbox[:,0] = -self.simbox[:,1]

        #Add solvent molecule type
        self.add_molecule_type(solvent[0])

        #Solvent volume = box volume - crystal volume (in angstrom^3)
        volume = np.prod(self.simbox[:,1]-self.simbox[:,0])
        if 'Xtal' in self.groups:
            xtal_vol = (xtal_hi-xtal_lo).prod()
            volume -= xtal_vol

        #Population of molecules 
        molwt = solvent[0].get_total_mass() # in g/mol
        dens = solvent[1] #Density in g/mL
        nonsolvent_mass = 0.0
        if 'Ligands' in self.groups:
            nonsolvent_atoms = self.groups['Ligands']['atoms']
            for each in nonsolvent_atoms:
                at = self.atoms[each]['type']
                nonsolvent_mass += self.atom_mass[at]
        pop = (0.6023*volume*dens - nonsolvent_mass)/molwt
        pop = math.floor(pop)

        aid_beg = self.num_atoms + 1 #First solvent atom id
        aid_end = aid_beg + pop*solvent[0].num_atoms - 1 #Last solvent atom id
        mid_beg = self.num_molecules + 1 #First solvent molecule id
        mid_end = mid_beg + pop - 1#Last solvent molecule id

        #Offsets in types
        type_offsets = tuple([self.num_atom_types, self.num_bond_types,
                        self.num_angle_types, self.num_dihedral_types,
                        self.num_improper_types])
        at_beg = self.num_atom_types + 1
        at_end = at_beg + solvent[0].num_atom_types - 1

        #Define bounding boxes & add molecules
        oa_bbox_lo = self.simbox[:,0] + delta #Overall bounding box
        oa_bbox_hi = self.simbox[:,1] - delta #Overall bounding box

        if self.is_slab:
            if self.slab_pos == 'bot':
                #A single bounding box for solvent molecules
                bbox_lo = [oa_bbox_lo[0], oa_bbox_lo[1], xtal_hi[2]]
                bbox_hi = oa_bbox_hi
                print(f"Packing solvents \n"
                      f" inside box ({' '.join(['%g'%v for v in bbox_lo])})"
                      f" ({' '.join(['%g'%v for v in bbox_hi])})"
                      )
                g = 'inside box ' \
                    + ' '.join(['%g'%v for v in bbox_lo]) + ' ' \
                    + ' '.join(['%g'%v for v in bbox_hi])
                elem = {'moltem': solvent[0], 'num': pop, 'offsets': type_offsets,
                        'constraints': [g]}
                mols_to_add = [elem]
            else:
                #Two bounding boxes for the top & bottom halves
                pop_top = int(0.5*pop); pop_bot = pop - pop_top

                top_bbox_hi = oa_bbox_hi
                top_bbox_lo = [ oa_bbox_lo[0], oa_bbox_lo[1], xtal_hi[2] ]

                bot_bbox_lo = oa_bbox_lo
                bot_bbox_hi = [ oa_bbox_hi[0], oa_bbox_hi[1], xtal_lo[2] ]

                print(f"Packing solvents \n"
                      f" inside box ({' '.join(['%g'%v for v in top_bbox_lo])})"
                      f" ({' '.join(['%g'%v for v in top_bbox_hi])}) \n"
                      f" inside box ({' '.join(['%g'%v for v in bot_bbox_lo])})"
                      f" ({' '.join(['%g'%v for v in bot_bbox_hi])})"
                      )
                g_top = 'inside box ' \
                    + ' '.join(['%g'%v for v in top_bbox_lo]) + ' ' \
                    + ' '.join(['%g'%v for v in top_bbox_hi])
                elem = {'moltem': solvent[0], 'num': pop_top, 'offsets': type_offsets,
                        'constraints': [g_top]}
                mols_to_add = [elem]

                g_bot = 'inside box ' \
                    + ' '.join(['%g'%v for v in bot_bbox_lo]) + ' ' \
                    + ' '.join(['%g'%v for v in bot_bbox_hi])
                elem = {'moltem': solvent[0], 'num': pop_bot, 'offsets': type_offsets,
                        'constraints': [g_bot]}
                mols_to_add.append(elem)
        else:
            #Add solvent molecules to the region between two bounding boxes
            in_bbox_lo = xtal_lo; in_bbox_hi = xtal_hi
            out_bbox_lo = oa_bbox_lo; out_bbox_hi = oa_bbox_hi
            
            print(f"Packing solvents \n"
                  f" inside box ({' '.join(['%g'%v for v in out_bbox_lo])})"
                  f" ({' '.join(['%g'%v for v in out_bbox_hi])}) \n"
                  f" outside box ({' '.join(['%g'%v for v in in_bbox_lo])})"
                  f" ({' '.join(['%g'%v for v in in_bbox_hi])})"
                  )
            g_in = 'inside box ' \
                + ' '.join(['%g'%v for v in out_bbox_lo]) + ' ' \
                + ' '.join(['%g'%v for v in out_bbox_hi])

            g_out = 'outside box ' \
                + ' '.join(['%g'%v for v in in_bbox_lo]) + ' ' \
                + ' '.join(['%g'%v for v in in_bbox_hi])

            elem = {'moltem': solvent[0], 'num': pop, 'offsets': type_offsets,
                    'constraints': [g_in, g_out]}
            mols_to_add = [elem]

        add_molecules(self, mols_to_add, packmol_tol, packmol_sidemax,
                      packmol_path)
        #Create new group of solvent atoms
        atoms = range(aid_beg, aid_end+1)
        atom_types = range(at_beg, at_end+1)
        molecules = range(mid_beg, mid_end+1)
        self.set_group('Solvent', atoms=atoms, atom_types=atom_types,
                       molecules=molecules)



    def deform(self, A, waveform):
        """

        Parameters
        ----------
        A : float
            Amplitude of the sine or cosine function
        waveform : {'sin', 'cos'}

        """
        lx = self.simbox[0,1] - self.simbox[0,0]
        if waveform == 'sin':
            for i in range(1, self.num_atoms+1):
                x = self.atoms[i]['coords'][0]
                dz = A*math.sin(2*math.pi*x/lx)
                self.atoms[i]['coords'][2] += dz
        elif waveform == 'cos':
            for i in range(1, self.num_atoms+1):
                x = self.atoms[i]['coords'][0]
                dz = A*math.cos(2*math.pi*x/lx)
                self.atoms[i]['coords'][2] += dz



    def add_piston(self, loc, lpar, thickness, atom_mass, eps, sigma, 
                   rcut=None, out_dir=None):
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
        if out_dir is None:
            odir = os.getcwd()
        else:
            path = os.path.expanduser(out_dir)
            if os.path.exists(path):
                odir = path
            else:
                warnings.warn(f"Directory {out_dir} does not exist, using"
                              " current working directory.") 
                odir = os.getcwd()

        delta = 1.0
        if loc == 'top' or loc == 'both':
            lo = [self.simbox[0,0], self.simbox[1,0], self.simbox[2,1] ]
            hi = [self.simbox[0,1], self.simbox[1,1], self.simbox[2,1]+thickness]
            coords = get_lattice_points('fcc', lpar, lo, hi, 'ppc')
            num_surface_atoms = np.count_nonzero(coords[:,2]==hi[2])
            print(f"Top piston: {num_surface_atoms} surface atoms", 
                  file=open(f"{odir}/info.txt", 'a'))
            print(f"#variable PtNa equal {num_surface_atoms}", 
                  file=open(f"{odir}/GrpLigHead.lmp", 'a'))

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
            lo = [self.simbox[0,0], self.simbox[1,0], self.simbox[2,0]-thickness]
            hi = [self.simbox[0,1], self.simbox[1,1], self.simbox[2,0]]
            coords = get_lattice_points('fcc', lpar, lo, hi, 'ppc')
            num_surface_atoms = np.count_nonzero(coords[:,2]==lo[2])
            print(f"Bottom piston: {num_surface_atoms} surface atoms",
                    file=open(f"{odir}/info.txt", 'a'))
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
            aids_charged = []
            for i in range(1, self.num_atoms+1):
                q = self.atoms[i]['charge']
                if not math.isclose(q, 0.0):
                    aids_charged.append(i)
            na_charged = len(aids_charged)
            cpa = chge/na_charged
            for i in aids_charged:
                self.atoms[i]['charge'] -= cpa
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

        soften : {'Ligands', 'Solvent', 'both'}
            Soften potential for this group.

        """
        ats_xtal = list(self.groups['Xtal']['atom_types'])
        if soften is None:
            with open(fn, 'w') as fh:
                #pair_style lj/cut/coul/long 12.0
                #pair_modify shift yes
                for i in range(1, self.num_atom_types+1):
                    coeffs = self.pair_coeffs[i]
                    eps = coeffs[0]; sigma = coeffs[1]
                    if sigma == 0: sigma = 2.5
                    if len(coeffs) > 2 :
                        rcuts = ' '.join([f"{x:g}" for x in coeffs[2:]])
                    else:
                        rcuts = ''
                    fh.write(f"pair_coeff {i} {i} {eps:g} {sigma:g} {rcuts}\n")

                    for j in range(i+1, self.num_atom_types+1):
                        coeffs_i = self.pair_coeffs[i]
                        coeffs_j = self.pair_coeffs[j]
                        eps_i = coeffs_i[0]; sigma_i = coeffs_i[1]
                        eps_j = coeffs_j[0]; sigma_j = coeffs_j[1]
                        eps_ij = math.sqrt(eps_i*eps_j)
                        sigma_ij = math.sqrt(sigma_i*sigma_j)
                        if sigma_ij == 0.0: sigma_ij = 2.5
                        if (i in ats_xtal) or (j in ats_xtal):
                            fh.write(f"pair_coeff {i} {j} "
                                 f" {eps_ij:g} {sigma_ij:g} 2\n")
                        else:
                            fh.write(f"pair_coeff {i} {j} "
                                 f" {eps_ij:g} {sigma_ij:g}\n")
        else:
            if soften=='Ligands' or soften=='Solvent':
                soft_atom_types = self.groups[soften]['atom_types']
            if soften=='both':
                soft_atom_types = list(self.groups['Ligands']['atom_types']) \
                                + list(self.groups['Solvent']['atom_types'])
            with open(fn, 'w') as fh:
                #pair_style lj/cut/coul/long/soft 2 0.5 10.0 12.0
                #pair_modify shift yes
                #Pair coeffs for self types (I == J)
                for i in range(1, self.num_atom_types+1):
                    lamda = 0.0 if i in soft_atom_types else 1.0
                    coeffs = self.pair_coeffs[i]
                    eps = coeffs[0]; sigma = coeffs[1]
                    if sigma == 0: sigma = 2.5
                    if len(coeffs) > 2 :
                        rcuts = ' '.join([f"{x:g}" for x in coeffs[2:]])
                    else:
                        rcuts = ''
                    fh.write(f"pair_coeff {i} {i} {eps:g} {sigma:g} {lamda} "
                             f"{rcuts}\n")

                #Pair coeffs for cross types (I <= J) using geometric mixing rule
                #Ignores explicit cutoff, will use the global cutoff
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
                        if sigma_ij == 0.0: sigma_ij = 2.5
                        if (i in ats_xtal) or (j in ats_xtal):
                            fh.write(f"pair_coeff {i} {j} {eps_ij:g} "
                                 f"{sigma_ij:g} {lamda} 2\n")
                        else:
                            fh.write(f"pair_coeff {i} {j} {eps_ij:g} "
                                 f"{sigma_ij:g} {lamda}\n")



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

        #Ligand density
        lx = self.simbox[0,1] - self.simbox[0,0]
        ly = self.simbox[1,1] - self.simbox[1,0]

        nlt = len(self.groups['LigandsTop']['molecules'])
        self.apl = lx*ly/nlt
        print(f"Area/ligand = {self.apl:f} A^2\n"
              f"Brush density = {1/self.apl:f}/A^2")
