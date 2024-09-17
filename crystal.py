#!/usr/bin/env python

from copy import deepcopy
import math
import numpy as np


def _get_lattice_points_2d(lattice, normal_axis, normal_pos,
                           lpar, lo, hi, boundary='pp'):
    """
    Generates points on a 2D lattice embedded in 3D space.

    Parameters
    ----------
    lattice : str
        {'sc', 'bcc'}
    normal_axis : {0, 1, 2}
        Axis along the positive normal. If `normal_axis = 0`, the elements of
        `lpar`, `lo`, & `hi` contain the corresponding values along axes 1 & 2,
        respectively.  Likewise, for `normal_axis = 1` the elemental values are
        along axes 0 & 2 and for `normal_axis = 2` the elemental values are
        along axes 1 & 2. 
    normal_pos : float
        Position of the lattice plane along the normal_axis
    lpar : scalar/list/tuple/ndarray of size (2,)
        Lattice parameter
    lo : (2,) array_like
        Lower bounds along two directions
    hi : (2,) array_like
        Upper bounds along two directions
    boundary : str
        Two characters specifying the nature of the boundary along the two
        axes. Allowed values are {'p', 'n', 'c'}. 'p' denotes periodic
        boundary, 'n' denotes non-periodic boundary, and 'c' denotes an
        end-capped boundary. For 'p' & 'c', `lpar` will be modified if
        necessary.

    Returns
    -------
    coords : ndarray of size (n,3)
        array of the coordinates of the lattice points

    """
    #Check boundary flags to see if the lattice parameter needs to be modified
    if np.isscalar(lpar):
        lpar_ = np.ones((2,))*lpar
    else:
        lpar_ = np.array(lpar)

    taxes = [0,1,2]; taxes.remove(normal_axis)

    nuc = np.zeros((2,), dtype=np.int32) #Number of unit cells along each direction
    for i in range(2):
        l = hi[i] - lo[i]
        if boundary[i]=='p':
            nuc[i] = round(l/lpar_[i])
            if nuc[i] < 1: nuc[i] = 1
            lpar_[i] = l/nuc[i]
        elif boundary[i]=='c':
            nuc[i] = round(l/lpar_[i])
            if nuc[i] < 1: nuc[i] = 1
            lpar_[i] = l/nuc[i]
            nuc[i] += 1
        elif boundary[i]=='n':
            nuc[i] = math.ceil(l/lpar_[i])
        print(f"Lattice parameter along axis {taxes[i]} = {lpar_[i]}")

    coords = []
    origin = np.zeros((3,))
    origin[normal_axis] = normal_pos
    origin[taxes[0]] = lo[0]
    origin[taxes[1]] = lo[1]
    coord = np.zeros((3,))
    a = lpar_[0]; b = lpar_[1]
    for j in range(nuc[1]):
        for i in range(nuc[0]):
            if lattice == 'sc':
                coord[taxes[0]] = i*a; coord[taxes[1]] = j*b
                coords.append(origin+coord)
            elif lattice == 'bcc':
                coord[taxes[0]] = i*a;
                coord[taxes[1]] = j*b
                coords.append(origin+coord)

                coord[taxes[0]] = (i+0.5)*a
                coord[taxes[1]] = (j+0.5)*b
                coords.append(origin+coord)

    #Check if boundary conditions are satisfied
    coords_final = []
    for each in coords:
        add = True
        for i in range(2):
            if boundary[i] == 'c' and each[taxes[i]] > hi[i]:
                add = False; break
            if boundary[i] == 'n' and each[taxes[i]] >= hi[i]:
                add = False; break
        if add: 
            coords_final.append(each)
    return np.array(coords_final)



def get_lattice_points(lattice, lpar, lo, hi, boundary='ppp'):
    """
    Generates points on a lattice.

    Parameters
    ----------
    lattice : str
        {'sc', 'bcc', 'fcc'}
    lpar : scalar/list/tuple/ndarray of size (3,)
        Lattice parameter
    lo : (3,) array_like
        Lower bounds along x, y, & z directions
    hi : (3,) array_like
        Upper bounds along x, y, & z directions
    boundary : str
        Three characters specifying the nature of the boundary along the x, y,
        & z axes. Allowed values are {'p', 'n', 'c'}. 'p' denotes periodic
        boundary, 'n' denotes non-periodic boundary, and 'c' denotes an
        end-capped boundary. For 'p' & 'c', `lpar` will be modified if
        necessary.

    Returns
    -------
    coords : ndarray of size (n,3)
        array of the coordinates of the lattice points

    Notes
    -----
    It is possible to set the lower and upper bounds equal to each other along
    only one of the three dimensions -- resulting in a 2D lattice. In this case
    `lattice` must be either `sc` or `bcc`.

    """
    #Check if lattice is 2D. If so, set the number of unit cells along the
    #third dimension equal to one.
    if np.any(np.array(lo) == np.array(hi)):
        assert np.count_nonzero(np.array(lo) == np.array(hi)) <= 1
        assert lattice in ['sc', 'bcc']
        normal_axis = np.nonzero(np.array(lo) == np.array(hi))[0][0]
        taxes = [0,1,2]; taxes.remove(normal_axis)
        normal_pos = lo[normal_axis]
        if np.isscalar(lpar):
            lpar_ = lpar
        else:
            lpar_ = [ lpar[taxes[0]], lpar[taxes[1]] ]
        lo_ = [ lo[taxes[0]], lo[taxes[1]] ]
        hi_ = [ hi[taxes[0]], hi[taxes[1]] ]
        bndry = boundary[taxes[0]] + boundary[taxes[1]]
        coords = _get_lattice_points_2d(lattice, normal_axis, normal_pos,
                                        lpar_, lo_, hi_, bndry)
        return coords

    #Check boundary flags to see if the lattice parameter needs to be modified
    if np.isscalar(lpar):
        lpar_ = np.ones((3,))*lpar
    else:
        lpar_ = np.array(lpar)
    nuc = np.zeros((3,), dtype=np.int32) #Number of unit cells along each direction
    for i in range(3):
        l = hi[i] - lo[i]
        if boundary[i]=='p':
            nuc[i] = round(l/lpar_[i])
            if nuc[i] < 1: nuc[i] = 1
            lpar_[i] = l/nuc[i]
        elif boundary[i]=='c':
            nuc[i] = round(l/lpar_[i])
            if nuc[i] < 1: nuc[i] = 1
            lpar_[i] = l/nuc[i]
            nuc[i] += 1
        elif boundary[i]=='n':
            nuc[i] = math.ceil(l/lpar_[i])
        print(f"Lattice parameter along axis {i} = {lpar_[i]}")

    coords = []
    origin = np.array(lo)
    coord = np.zeros((3,))
    a = lpar_[0]; b = lpar_[1]; c = lpar_[2]
    for k in range(nuc[2]):
        for j in range(nuc[1]):
            for i in range(nuc[0]):
                if lattice == 'sc':
                    coord[0] = i*a; coord[1] = j*b; coord[2] = k*c
                    coords.append(origin+coord)
                elif lattice == 'bcc':
                    coord[0] = i*a;
                    coord[1] = j*b
                    coord[2] = k*c
                    coords.append(origin+coord)

                    coord[0] = (i+0.5)*a
                    coord[1] = (j+0.5)*b
                    coord[2] = (k+0.5)*c
                    coords.append(origin+coord)
                elif lattice == 'fcc':
                    coord[0] = i*a;
                    coord[1] = j*b
                    coord[2] = k*c
                    coords.append(origin+coord)

                    coord[0] = (i+0.5)*a
                    coord[1] = (j+0.5)*b
                    coord[2] = k*c
                    coords.append(origin+coord)

                    coord[0] = (i+0.5)*a
                    coord[1] = j*b
                    coord[2] = (k+0.5)*c
                    coords.append(origin+coord)

                    coord[0] = i*a
                    coord[1] = (j+0.5)*b
                    coord[2] = (k+0.5)*c
                    coords.append(origin+coord)

    #Check if boundary conditions are satisfied
    coords_final = []
    for each in coords:
        add = True
        for i in range(3):
            if boundary[i] == 'c' and each[i] > hi[i]:
                add = False; break
            if boundary[i] == 'n' and each[i] >= hi[i]:
                add = False; break
        if add: 
            coords_final.append(each)
    return np.array(coords_final)



class Crystal(object):
    """
    Specifies a crystal lattice.

    """

    def __init__(self, a, b=None, c=None, caxes=np.eye(3), fcoords=None, 
                 atom_types={}, atom_order=[]):
        """
        Parameters
        ----------
        a : float
            Lattice parameter along crystallographic axis `a` in angstrom.
        b : float
            Lattice parameter along crystallographic axis `b` in angstrom. If
            `None`, is set equal to `a`.
        c : float
            Lattice parameter along crystallographic axis `c` in angstrom. If
            `None`, is set equal to `a`.
        caxes : (3,3) ndarray of floats
            Each row is a unit vector along the crystallographic axes `a`, `b`,
            and `c`, respectively.
        fcoords : (N,3) ndarray of floats
            Fractional coordinates of the atoms in a unit cell. Row specifies
            the fractional coordinates of atom `i` of the unit cell with respect
            to `caxes`. 
        atom_types : dict of dict
            Specification for each atomic species. Type numbering begins at `1`
            with the following pattern: atom_types = {1: {'name': <str>, 
            'mass': <float>}, ...}.
        atom_order : (N,) ndarray of ints
            The type of each atom in the order specified by the rows in `fcoords`.

        """
        self.a = a
        self.b = self.a if b is None else b
        self.c = self.a if c is None else c
        self.caxes = deepcopy(caxes)
        self.fcoords = deepcopy(fcoords)
        self.atom_types = deepcopy(atom_types)
        self.atom_order = np.array(atom_order, dtype=np.int32)
        scale = np.diag([self.a, self.b, self.c])
        self.ucell = np.matmul(self.fcoords, (scale@self.caxes).T)


    def get_num_atom_types(self):
        """
        Returns the number of atom types in the unit cell

        """
        return len(self.atom_types)


    def get_atom_name(self, ityp):
        """
        Returns the name of an atom of type `ityp`.

        Parameters
        ----------
        ityp : int
            Type of an atom

        Returns
        -------
        str
            Name of an atom

        """
        assert ityp in self.atom_types
        return self.atom_types[ityp]['name']


    def get_atom_mass(self, ityp):
        """
        Returns the mass of an atom of type `ityp`.

        Parameters
        ----------
        ityp : int
            Type of an atom

        Returns
        -------
        float
            Mass of an atom

        """
        assert ityp in self.atom_types
        return self.atom_types[ityp]['mass']


    def get_atom_type(self, iatm=0):
        """
        Returns the type of the *i*th atom in the unit cell.

        Parameters
        ----------
        iatm : int
            Id of an atom. If 0, all atom types are retuned.

        Returns
        -------
        int | sequence of ints
            Type of an atom or types of all atoms

        """
        if iatm == 0:
            return self.atom_order
        else:
            return self.atom_order[iatm-1]


    def get_num_atoms(self, ityp=0):
        """
        Returns the number of atoms of type `ityp` in the unit cell.

        Parameters
        ----------
        ityp : int
            Atom type. If 0, the number of atoms of all types are retuned.

        Returns
        -------
        int or sequence of ints
            Number of atoms of type `ityp`. In case `ityp = 0`, the number of
            atoms for all atom types are returned.

        """
        if ityp == 0:
            return self.ucell.shape[0]
        else:
            assert ityp in self.atom_types
            return np.count_nonzero(self.atom_order==ityp)


    def get_atoms_in_layer(self, ilayer):
        """
        Returns the atoms in layer `ilayer` of the crystal.

        Parameters
        ----------
        ilayer : int
            Layer number. Must be >= 1.

        Returns
        -------
        tuple of lists
            The first element of the tuple is a list of atoms positions and the
            second elememt a list of corresponding atom types.

        """
        assert ilayer >= 1
        j = (ilayer-1)//4; k = (ilayer-1)%4
        origin = np.array([0, 0, j*self.c])
        pos = [ self.ucell[2*k,:]+origin, self.ucell[2*k+1,:]+origin ]
        types = [self.atom_order[2*k], self.atom_order[2*k+1]]
        return pos, types


fcoords = np.array([ 
   [0.00, 0.00, 0.00],
   [0.50, 0.50, 0.00],
   [0.25, 0.25, 0.25],
   [0.75, 0.75, 0.25],
   [0.50, 0.00, 0.50],
   [0.00, 0.50, 0.50],
   [0.75, 0.25, 0.75],
   [0.25, 0.75, 0.75] 
   ])

#1atm Rabani a=6.1379
CdSe_zb = Crystal(6.1379, fcoords=fcoords, 
                  atom_types={1: {'name': 'Cd', 'mass': 112.411}, 
                              2: {'name': 'Se', 'mass': 78.96}},
                  atom_order=[1, 1, 2, 2, 1, 1, 2, 2]
                  )

CdTe_zb = Crystal(6.56, fcoords=fcoords, 
                  atom_types={1: {'name': 'Cd', 'mass': 112.411}, 
                              2: {'name': 'Te', 'mass': 127.60}},
                  atom_order=[1, 1, 2, 2, 1, 1, 2, 2]
                  )

CdS_zb = Crystal(5.89, fcoords=fcoords, 
                  atom_types={1: {'name': 'Cd', 'mass': 112.411}, 
                              2: {'name': 'S', 'mass': 32.065}},
                  atom_order=[1, 1, 2, 2, 1, 1, 2, 2]
                  )
