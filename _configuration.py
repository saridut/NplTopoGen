#!/usr/bin/env python

import warnings
import math
import copy
import numpy as np
from _geom_utils import rotate_vectors_random, get_ransphere, get_frc_link

def coeffs_equal(coeffa, coeffb):
    if len(coeffa) != len(coeffb):
        return False
    out = True
    for a,b in zip(coeffa,coeffb):
        if a != b:
            try:
                out = math.isclose(float(a), float(b), rel_tol=1e-6)
            except ValueError:
                out = False
        if not out:
            break
    return out


class Configuration(object):

    def __init__(self):
        self.simbox = np.zeros((3,2))
        self.tilt_factors = np.zeros((3,))

        self.num_atom_types = 0
        self.num_bond_types = 0
        self.num_angle_types = 0
        self.num_dihedral_types = 0
        self.num_improper_types = 0

        self.num_atoms = 0
        self.num_bonds = 0
        self.num_angles = 0
        self.num_dihedrals = 0
        self.num_impropers = 0

        self.num_groups = 0
        self.num_molecules = 0

        self.atom_mass = {}
        self.atom_names = {}

        self.pair_coeffs = {}
        self.bond_coeffs = {}
        self.angle_coeffs = {}
        self.dihedral_coeffs = {}
        self.improper_coeffs = {}

        self.atoms = {}
        self.velocities = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}
        self.impropers = {}
        self.groups = {}
        self.molecules = {}
        

    def clear(self):
        '''
        Clears all data.
        '''
        self.simbox[:,:] = 0.0
        self.tilt_factors[:] = 0.0

        self.num_atom_types = 0
        self.num_bond_types = 0
        self.num_angle_types = 0
        self.num_dihedral_types = 0
        self.num_improper_types = 0

        self.num_atoms = 0
        self.num_bonds = 0
        self.num_angles = 0
        self.num_dihedrals = 0
        self.num_impropers = 0

        self.num_groups = 0
        self.num_molecules = 0

        self.atom_mass.clear()
        self.atom_names.clear()

        self.pair_coeffs.clear()
        self.bond_coeffs.clear()
        self.angle_coeffs.clear()
        self.dihedral_coeffs.clear()
        self.improper_coeffs.clear()

        self.atoms.clear()
        self.velocities.clear()
        self.bonds.clear()
        self.angles.clear()
        self.dihedrals.clear()
        self.impropers.clear()
        self.groups.clear()
        self.molecules.clear()


    def add_simbox(self, xlo, xhi, ylo, yhi, zlo, zhi):
        '''
        Adds a simulation box. x-dir is toward the right, y is up, and z points
        out of the screen.

        '''
        self.simbox = np.array([[xlo,xhi],[ylo,yhi],[zlo,zhi]], dtype=np.float64)


    def add_atom_type(self, mass=1.0, name=None):
        '''
        Adds a new atom type `iat` with mass `mass` and name `name`. 
        If `name` is None, a string "Xn" will be assigned, where n = atom type.
        Returns an integer identifier for the atom type added.

        '''
        self.num_atom_types += 1
        iat = self.num_atom_types
        self.set_atom_mass(iat, mass)
        if name is None:
            name = 'X%d'%iat
        self.set_atom_names([(iat,name)])
        return iat


    def set_atom_names(self, names):
        '''
        Assigns names to atom types. `names` is a list of tuples (at,name),
        where `at` is the atom type and `name` is the name to be assigned 
        to it. `name` should be <= 8 characters.

        This method cannot be called before the corresponding atom types have been
        added.

        '''
        for each in names:
            iat = each[0]; name = each[1]
            assert iat >= 1 and iat <= self.num_atom_types
            if len(name) > 16:
                print('Atom name `%s` longer than 16 characters.'%name)
                print('Truncating name to the first 16 characters.')
                self.atom_names[iat] = name[0:16]
            else:
                self.atom_names[iat] = name


    def set_atom_mass(self, iat, mass):
        assert iat >= 1 and iat <= self.num_atom_types
        assert mass > 0
        self.atom_mass[iat] = mass


    def add_bond_type(self, params=None):
        self.num_bond_types += 1
        ibt = self.num_bond_types
        if params:
            self.bond_coeffs[ibt] = list(params)
        return ibt


    def set_bond_coeff(self, ibt, params):
        assert ibt >= 1 and ibt <= self.num_bond_types
        self.bond_coeffs[ibt] = list(params)


    def add_angle_type(self, params=None):
        self.num_angle_types += 1
        iant = self.num_angle_types
        if params:
            self.angle_coeffs[iant] = list(params)
        return iant


    def set_angle_coeff(self, iant, params):
        assert iant >= 1 and iant <= self.num_angle_types
        self.angle_coeffs[iant] = list(params)


    def add_dihedral_type(self, params=None):
        self.num_dihedral_types += 1
        idt = self.num_dihedral_types
        if params:
            self.dihedral_coeffs[idt] = list(params)
        return idt


    def set_dihedral_coeff(self, idt, params):
        assert idt >= 1 and idt <= self.num_dihedral_types
        self.dihedral_coeffs[idt] = list(params)


    def add_improper_type(self, params=None):
        self.num_improper_types += 1
        iit = self.num_improper_types
        if params:
            self.improper_coeffs[iit] = list(params)
        return iit


    def set_improper_coeff(self, iit, params):
        assert iit >= 1 and iit <= self.num_improper_types
        self.improper_coeffs[iit] = list(params)


    def set_pair_coeff(self, typ, params):
        '''
        All atom types must be specified before calling this function.

        '''
        assert typ >= 1 and typ <= self.num_atom_types
        self.pair_coeffs[typ] = list(params)


    def add_atom(self, typ, charge, coords, imol=0, atm_id=None):
        assert typ >= 1 and typ <= self.num_atom_types
        if atm_id is None:
            atm_id = len(self.atoms) + 1
        self.atoms[atm_id] = {'type': typ, 'charge': charge, 'imol': imol,
                'coords': np.copy(coords[0:3])}
        self.num_atoms += 1
        self.num_molecules = max(self.num_molecules, imol)
        return atm_id


    def append_atom_bonded(self, typ, charge, len_bond, method, im1,
            im2=None, theta=None, uhat=None, sep=None):
        assert typ >= 1 and typ <= self.num_atom_types
        atm_id = len(self.atoms) + 1

        itr = 0; maxitr = 100 #For method = 'efjc'|'efrc'

        if method == 'alignx':
            rim1 = self.atoms[im1]['coords']
            ri = rim1 + np.array([len_bond, 0.0, 0.0])
        elif method == 'aligny':
            rim1 = self.atoms[im1]['coords']
            ri = rim1 + np.array([0.0, len_bond, 0.0])
        elif method == 'alignz':
            rim1 = self.atoms[im1]['coords']
            ri = rim1 + np.array([0.0, 0.0, len_bond])
        elif method == 'align':
            rim1 = self.atoms[im1]['coords']
            ri = rim1 + len_bond*uhat
        elif method == 'fjc':
            rim1 = self.atoms[im1]['coords']
            ri = rim1 + get_ransphere(len_bond)
        elif method == 'frc':
            rim2 = self.atoms[im2]['coords']
            rim1 = self.atoms[im1]['coords']
            ri = rim1 + len_bond*get_frc_link(rim1-rim2, theta)
        elif method == 'efjc':
            rim1 = self.atoms[im1]['coords']
            while itr < maxitr:
                ri = rim1 + get_ransphere(len_bond)
                if not self.has_overlap(ri.reshape((1,3)), 'all', None, sep):
                    break
                itr += 1
            else:
                warnings.warn('Maximum iteration reached for appending bonded atom')
        elif method == 'efrc':
            rim2 = self.atoms[im2]['coords']
            rim1 = self.atoms[im1]['coords']
            while itr < maxitr:
                ri = rim1 + len_bond*get_frc_link(rim1-rim2, theta)
                if not self.has_overlap(ri.reshape((1,3)), 'all', None, sep):
                    break
                itr += 1
            else:
                warnings.warn('Maximum iteration reached for appending bonded atom')
        else:
            raise ValueError('Unknown method "%s"'%method)

        imol = self.atoms[im1]['imol']
        self.atoms[atm_id] = {'type': typ, 'charge': charge, 'imol': imol, 'coords': ri}
        self.num_atoms += 1
        return atm_id


    def append_atom_unbonded(self, typ, charge, sep=None):
        assert typ >= 1 and typ <= self.num_atom_types
        atm_id = len(self.atoms) + 1
        itr = 0; maxitr = 100
        if not sep:
            ri = np.random.random_sample((3,))
            ri = self.simbox[:,0] + ri*(self.simbox[:,1]-self.simbox[:,0])
        else:
            while itr < maxitr:
                #Choose any random position within the box. 
                ri = np.random.random_sample((3,))
                ri = self.simbox[:,0] + ri*(self.simbox[:,1]-self.simbox[:,0])
                if not self.has_overlap(ri.reshape((1,3)), 'all', None, sep):
                    break
                itr += 1
            else:
                warnings.warn('Maximum iteration reached for appending unbonded atom')
        self.atoms[atm_id] = {'type': typ, 'charge': charge, 'imol': 0, 'coords': ri}
        self.num_atoms += 1
        return atm_id

    
    def set_atom_coords(self, atm_ids, coords):
        n = len(atm_ids)
        for i in range(n):
            atm_id = atm_ids[i]
            self.atoms[atm_id]['coords'] = np.copy(coords[i,:])


    def get_atom_coords(self, atm_ids=None, out=None):
        if atm_ids is None:
            n = self.num_atoms; aids = range(1, n+1)
        else:
            n = len(atm_ids); aids = atm_ids
        if out is None:
            coords = np.zeros((n,3))
        else:
            nr, nc = out.shape
            assert nr >= n
            assert nc >= 3
            coords = out 
        for i in range(n):
            aid = aids[i]
            coords[i,:] = np.copy(self.atoms[aid]['coords'])
        if out is None:
            return coords


    def add_velocity(self, atm_id, vel):
        assert atm_id in self.atoms
        self.velocities[atm_id] = vel
        return None


    def add_bond(self, typ, atom_i, atom_j, bnd_id=None):
        assert typ >= 1 and typ <= self.num_bond_types
        assert atom_i in self.atoms
        assert atom_j in self.atoms
        if bnd_id is None:
            bnd_id = len(self.bonds) + 1
        self.bonds[bnd_id] = {'type': typ, 'atm_i': atom_i, 'atm_j': atom_j}
        self.num_bonds += 1
        return bnd_id


    def add_angle(self, typ, atom_i, atom_j, atom_k, ang_id=None):
        assert typ >= 1 and typ <= self.num_angle_types
        assert atom_i in self.atoms
        assert atom_j in self.atoms
        assert atom_k in self.atoms
        if ang_id is None:
            ang_id = len(self.angles) + 1
        self.angles[ang_id] = {'type': typ, 'atm_i': atom_i, 'atm_j': atom_j,
                'atm_k': atom_k}
        self.num_angles += 1
        return ang_id


    def add_dihedral(self, typ, atom_i, atom_j, atom_k, atom_l, dhd_id=None):
        assert typ >= 1 and typ <= self.num_dihedral_types
        assert atom_i in self.atoms
        assert atom_j in self.atoms
        assert atom_k in self.atoms
        assert atom_l in self.atoms
        if dhd_id is None:
            dhd_id = len(self.dihedrals) + 1
        self.dihedrals[dhd_id] = {'type': typ, 'atm_i': atom_i, 'atm_j': atom_j,
                'atm_k': atom_k, 'atm_l': atom_l}
        self.num_dihedrals += 1
        return dhd_id


    def add_improper(self, typ, atom_i, atom_j, atom_k, atom_l, imp_id):
        assert typ >= 1 and typ <= self.num_improper_types
        assert atom_i in self.atoms
        assert atom_j in self.atoms
        assert atom_k in self.atoms
        assert atom_l in self.atoms
        if imp_id is None:
            imp_id = len(self.impropers) + 1
        self.impropers[imp_id] = {'type': typ, 'atm_i': atom_i, 'atm_j': atom_j,
                'atm_k': atom_k, 'atm_l': atom_l}
        self.num_impropers += 1
        return imp_id

    
    def set_group(self, name, atom_types=None, atoms=None, molecules=None):
        """
        Parameters
        ----------
        atom_types : range | list
        atoms : range | list
        molecules : range | list

        Returns
        -------
        None

        """
        if name in self.groups.keys():
            s = input("Group name must be unique. Enter a different name: ")
            name_ = str(s)
        else:
            name_ = str(name)
        self.groups[name_] = {}
        if atom_types is not None:
            self.groups[name_]['atom_types'] = atom_types
        if molecules is not None:
            self.groups[name_]['molecules'] = molecules
        if atoms is not None:
            self.groups[name_]['atoms'] = atoms
        self.num_groups += 1


    def remove_duplicate_types(self):
        self.remove_duplicate_atom_types()
        for x in ['bond', 'angle', 'dihedral', 'improper']:
            self.remove_duplicate_x_types(x)


    def remove_duplicate_atom_types(self):
        assert len(self.atom_names) == self.num_atom_types
        assert len(self.atom_mass) == self.num_atom_types
        assert len(self.pair_coeffs) == self.num_atom_types
        types = np.zeros((self.num_atom_types,2), dtype=np.int32)
        types[:,0] = np.arange(1,self.num_atom_types+1)
        for typ_i in range(1, self.num_atom_types):
            if types[typ_i-1,0] < typ_i:
                continue
            mass_i = self.atom_mass[typ_i]
            coeff_i = self.pair_coeffs[typ_i]
            for typ_j in range(typ_i+1, self.num_atom_types+1):
                if types[typ_j-1,0] <= typ_i:
                    continue
                if mass_i == self.atom_mass[typ_j] and \
                        coeffs_equal(coeff_i, self.pair_coeffs[typ_j]):
                    types[typ_j-1,0] = typ_i
        if np.any(types[:,0] != np.arange(1,self.num_atom_types+1)):
            uniqtyps = np.unique(types[:,0])
            for i in range(self.num_atom_types):
                m = np.abs(uniqtyps-types[i,0]).argmin()
                types[i,1] = m + 1
            for val in self.atoms.values():
                m = val['type']; val['type'] = types[m-1,1]
        
            atm_mass_tmp = copy.deepcopy(self.atom_mass)
            atm_nam_tmp = copy.deepcopy(self.atom_names)
            pair_coeff_tmp = copy.deepcopy(self.pair_coeffs)
            self.atom_mass = {}; self.atom_names = {}; self.pair_coeffs = {}
            for i,v in enumerate(uniqtyps):
                self.atom_mass[i+1] = atm_mass_tmp.pop(v)
                self.atom_names[i+1] = atm_nam_tmp.pop(v)
                self.pair_coeffs[i+1] = pair_coeff_tmp.pop(v)

        self.num_atom_types = len(self.atom_mass)
        assert len(self.atom_names) == self.num_atom_types
        assert len(self.pair_coeffs) == self.num_atom_types


    def remove_duplicate_x_types(self, x):
        if x == 'bond':
            num_x_types = self.num_bond_types
            x_coeffs = self.bond_coeffs
            x_topo = self.bonds
        elif x == 'angle':
            num_x_types = self.num_angle_types
            x_coeffs = self.angle_coeffs
            x_topo = self.angles
        elif x == 'dihedral':
            num_x_types = self.num_dihedral_types
            x_coeffs = self.dihedral_coeffs
            x_topo = self.dihedrals
        elif x == 'improper':
            num_x_types = self.num_improper_types
            x_coeffs = self.improper_coeffs
            x_topo = self.impropers
        else:
            raise ValueError('Unknown input: %s'%x)

        if num_x_types == 0:
            return #No type information, nothing to do 

        assert len(x_coeffs) == num_x_types
        types = np.zeros((num_x_types,2), dtype=np.int32)
        types[:,0] = np.arange(1,num_x_types+1)
        for typ_i in range(1, num_x_types):
            if types[typ_i-1,0] < typ_i:
                continue
            coeff_i = x_coeffs[typ_i]
            for typ_j in range(typ_i+1, num_x_types+1):
                if types[typ_j-1,0] <= typ_i:
                    continue
                if coeffs_equal(coeff_i, x_coeffs[typ_j]):
                    types[typ_j-1,0] = typ_i
        if np.any(types[:,0] != np.arange(1,num_x_types+1)):
            uniqtyps = np.unique(types[:,0])
            for i in range(num_x_types):
                m = np.abs(uniqtyps-types[i,0]).argmin()
                types[i,1] = m + 1
            if len(x_topo) > 0:
                for val in x_topo.values():
                    m = val['type']; val['type'] = types[m-1,1]
        
            x_coeff_tmp = copy.deepcopy(x_coeffs)
            x_coeffs.clear()
            for i,v in enumerate(uniqtyps):
                x_coeffs[i+1] = x_coeff_tmp.pop(v)

        num_x_types = len(x_coeffs)

        if x == 'bond':
            self.num_bond_types = num_x_types
        elif x == 'angle':
            self.num_angle_types = num_x_types
        elif x == 'dihedral':
            self.num_dihedral_types = num_x_types
        elif x == 'improper':
            self.num_improper_types = num_x_types



    def add_img_flag(self, atm_id, iflag):
        '''
        Adds an image flag to an atom. Ensure that either all atoms possess an
        image flag or no atom does.

        iflag : (3,) int, ndarray
        '''
        self.atoms[atm_id]['img_flag'] = iflag



    def apply_pbc_atoms(self, pos, directions='xyz'):
        '''
        Applies PBC along all directions to a set of atoms with coordinates
        `pos`. The input is updated with the new positions.

        '''
        flag = np.array([1,1,1], dtype=np.int32)
        if 'x' not in directions: flag[0] = 0
        if 'y' not in directions: flag[1] = 0
        if 'z' not in directions: flag[2] = 0

        boxl = self.simbox[:,1] - self.simbox[:,0]
        coords = copy.deepcopy(pos)
        for i in range(coords.shape[0]):
            dr = coords[i,:] - self.simbox[:,0] #Origin at left lower back corner
            pb = np.floor(dr/boxl) #Index of the periodic box
            pb = np.where(flag==0, flag, pb)
            dr -= boxl*pb   # PBC
            coords[i,:] = self.simbox[:,0] + dr #Back to origin
        return coords



    def apply_pbc(self, directions='xyz', add_img_flag=False):
        '''
        Applies PBC along all directions.

        '''
        flag = np.array([1,1,1], dtype=np.int32)
        if 'x' not in directions: flag[0] = 0
        if 'y' not in directions: flag[1] = 0
        if 'z' not in directions: flag[2] = 0

        boxl = self.simbox[:,1] - self.simbox[:,0]
        for i in range(1, self.num_atoms+1):
            if 'img_flag' in self.atoms[i]:
                continue
            coords = self.atoms[i]['coords']
            dr = coords - self.simbox[:,0] #Shift origin to the left lower back corner
            pb = np.floor(dr/boxl) #Index of the periodic box
            pb = np.where(flag==0, flag, pb)
            dr -= boxl*pb   # PBC
            coords[:] = self.simbox[:,0] + dr #Back to the origin
            if add_img_flag:
                self.atoms[i]['img_flag'] = pb


    def unwrap_pbc(self):
        '''
        Unwraps PBC along all directions by traversing all bonds. Note that only
        bonded atoms can be unwrapped (that too in the absence of rings).

        '''
        if len(self.bonds) == 0:
            return

        boxl = self.simbox[:,1] - self.simbox[:,0]
        for i in range(1, len(self.bonds)+1):
            atm_i = self.bonds[i]['atm_i']
            atm_j = self.bonds[i]['atm_j']
            coords_i = self.atoms[atm_i]['coords']
            coords_j = self.atoms[atm_j]['coords']
            #Atom i has higher atm_id
            dr = coords_i - coords_j
            dr -= boxl*np.rint(dr/boxl)   # PBC
            coords_i[:] = coords_j + dr


    def fit_simbox(self, sep=0.0):
        '''
        Changes the simulation box dimensions to the extent of the atom
        positions, i.e., it is a minimal bounding box for the atoms.
        Does not change boundary conditions as set before.
        `self.add_simbox` must be called and all atoms already added 
        before calling this function.
        Useful for setting the box size when it is hard to guess or calculate
        box size, e.g. for single molecules in unbounded domain.

        '''
        rlo = np.array([np.inf, np.inf, np.inf])
        rhi = np.array([np.NINF, np.NINF, np.NINF])

        for iatm in range(1, len(self.atoms)+1):
            coords = self.atoms[iatm]['coords']
            rlo = np.minimum(rlo, coords)
            rhi = np.maximum(rhi, coords)
        #Update box dimensions
        self.simbox[:,0] = rlo - sep
        self.simbox[:,1] = rhi + sep


    def translate(self, r, only_atoms=False, atm_ids=None, out_coords=None):
        '''
        Translates the whole system (atoms and box) by a vector `r`.
        '''
        dr = np.copy(r)
        if atm_ids is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atm_ids
        if out_coords is None:
            for i in aids:
                self.atoms[i]['coords'] += dr
            if not only_atoms:
                self.simbox[:,0] += dr
                self.simbox[:,1] += dr
        else:
            nr, nc = out_coords.shape
            assert nr >= len(aids)
            assert nc >= 3
            for i,k in enumerate(aids):
                out_coords[i,:] = self.atoms[k]['coords'] + dr


    def get_barycenter(self, atoms=None):
        '''
        Returns the centroid.
        '''
        if atoms is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atoms
        com = np.zeros((3,))
        for i in aids:
            com += self.atoms[i]['coords']
        com /= len(aids)
        return com


    def get_com(self, atoms=None):
        '''
        Returns the center of mass.
        '''
        if atoms is None:
            aids = range(1, self.num_atoms+1)
        else:
            aids = atoms
        com = np.zeros((3,)); totmass = 0.0
        for i in aids:
            at = self.atoms[i]['type']
            mass = self.atom_mass[at]
            coords = self.atoms[i]['coords']
            com += (mass*coords); totmass += mass
        com /= totmass
        return com


#   def has_overlap(self, ri, sep):
#       '''
#       Checks if an atom at position `ri` overlaps with existing atoms within a
#       separation distance of `sep`.
#       '''
#       out = False
#       sepsq = sep*sep
#       for j in self.atoms.keys():
#           rj = self.atoms[j]['coords']
#           rij = rj - ri
#           rijmsq = np.dot(rij, rij)
#           if rijmsq < sepsq:
#               out = True
#               break
#       return out



    def has_overlap(self, pos, key, val, sep):
        '''
        Checks atomwise overlap of a set of atoms `new` with an existing set 
        of atoms `old` within a separation distance of `sep`.

        pos : (n,3) ndarray of atom positions
        key : 'atm_id'/'atm_type'/'all'
        val : integer 1D numpy array of arguments

        '''
        out = False
        sepsq = sep*sep
        boxl = self.simbox[:,1] - self.simbox[:,0]
        rboxl = 1.0/boxl
        
        #Atom ids of the set of existing atoms against which overlap is to be
        #tested
        if key == 'atm_id':
            set_ext = val
        elif key == 'atm_type':
            set_ext = []
            for i in range(1, len(self.atoms)+1):
                if self.atoms[i]['type'] in val:
                    set_ext.append(i)
        elif key == 'all':
            set_ext = np.arange(1, len(self.atoms)+1, 1, dtype=np.int32)
        else:
            raise ValueError('Unknown key "%s"'%key)

        for j in range(pos.shape[0]):
            rj = pos[j,:]
            for i in set_ext:
                ri = self.atoms[i]['coords']
                rij = rj - ri
                k = np.trunc(rij*rboxl) + np.where(rij >= 0, 0.5, -0.5)
                rij -= k*boxl
                rijmsq = rij[0]*rij[0] + rij[1]*rij[1] + rij[2]*rij[2]
                if rijmsq < sepsq:
                    out = True
                    return out
        return out



    def get_total_charge(self):
        '''
        Returns the total charge for the configuration.
        '''
        charges = np.zeros((self.num_atoms,), dtype=np.float64)
        for iatm in range(1,self.num_atoms+1):
            charges[iatm-1] = self.atoms[iatm]['charge']
        return math.fsum(charges)


    def get_total_mass(self):
        '''
        Returns the total mass for the configuration.
        '''
        mass = np.zeros((self.num_atoms,), dtype=np.float64)
        for iatm in range(1,len(self.atoms)+1):
            at = self.atoms[iatm]['type']
            mass[iatm-1] = self.atom_mass[at]
        return math.fsum(mass)


    def tweak_cvff_impropers(self):
        """
        Ensure that CVFF improper coefficients are integers.

        """
        for val in self.improper_coeffs.values():
            val[1] = int(val[1]); val[2] = int(val[2])


    
