#!/usr/bin/env python

"""
Rotates a molecule to a desired configuration.

"""
import math
import numpy as np
from _config_io import write_ldf
from molecule import Molecule

def get_rotmat_axis_angle(axis, angle):
    R = np.zeros((3,3))
    sin = np.sin(angle)
    cos = np.cos(angle)
    icos = 1.0 - cos
    R[0,0] = axis[0]*axis[0]*icos + cos
    R[0,1] = axis[0]*axis[1]*icos - axis[2]*sin
    R[0,2] = axis[0]*axis[2]*icos + axis[1]*sin
    R[1,0] = axis[0]*axis[1]*icos + axis[2]*sin
    R[1,1] = axis[1]*axis[1]*icos + cos
    R[1,2] = axis[1]*axis[2]*icos - axis[0]*sin
    R[2,0] = axis[2]*axis[0]*icos - axis[1]*sin
    R[2,1] = axis[1]*axis[2]*icos + axis[0]*sin
    R[2,2] = axis[2]*axis[2]*icos + cos
    return R

toluene = Molecule(name='Toluene', 
             fn='solvents/toluene/slvnt_toluene.lmp')

#Bring atom 2 to origin
r = toluene.atoms[2]['coords']
toluene.translate(-r, only_atoms=True)

#Align bond 2->1 along x-axis
u = toluene.atoms[1]['coords'] - toluene.atoms[2]['coords']
uhat = u/np.linalg.norm(u)
theta = math.acos(uhat[0])
axis = np.cross(np.array([1,0,0]), uhat)
axis /= np.linalg.norm(axis)
rotmat = get_rotmat_axis_angle(axis, -theta)
for i in range(1, toluene.num_atoms+1):
    v = toluene.atoms[i]['coords']
    toluene.atoms[i]['coords'] = np.dot(rotmat, v)

#Rotate the ring about x-axis to align the plane along z-axis
u = toluene.atoms[3]['coords'] - toluene.atoms[2]['coords']
v = toluene.atoms[4]['coords'] - toluene.atoms[3]['coords']
axis = np.cross(u, v)
axis /= np.linalg.norm(axis)
theta = math.acos(axis[2])
rotmat = get_rotmat_axis_angle(np.array([1,0,0]), theta)
for i in range(1, toluene.num_atoms+1):
    v = toluene.atoms[i]['coords']
    toluene.atoms[i]['coords'] = np.dot(rotmat, v)


toluene.fit_simbox()

toluene.write(fn='inpos.lmp', title='Toluene')
