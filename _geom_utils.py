#!/usr/bin/env python

'''
Contains routines for rigid body rotation and other geometry specific
utilities.

'''

import math
import numpy as np

#-------------------------------------------------------------------------------

def get_ransphere(rad):
    #Generates a random vector distributed uniformly on the surface of a sphere
    #of radius r
    ransphere = np.zeros((3,))
    while True:
        zeta1 = -1.0 + 2*np.random.random()
        zeta2 = -1.0 + 2*np.random.random()
        zetasq = zeta1*zeta1 + zeta2*zeta2
        if zetasq <= 1.0:
            rt = math.sqrt(1.0 - zetasq)
            ransphere[0] = 2*zeta1*rt
            ransphere[1] = 2*zeta2*rt
            ransphere[2] = 1.0 - 2*zetasq
            break
    ransphere = ransphere/np.linalg.norm(ransphere)
    return rad*ransphere

#-------------------------------------------------------------------------------

def fix_axis_angle(axis, angle, normalize=True):
    if normalize:
        norm = np.linalg.norm(axis)
        if not math.isclose(norm, 1.0, abs_tol=1e-14, rel_tol=1e-14):
            axis /= norm
    angle = math.fmod(angle, 2*math.pi)
    if angle < 0.0:
        angle = -angle
        axis = -axis
    if angle > math.pi:
        angle = 2*math.pi - angle
        axis = -axis
    return (axis, angle)

#-------------------------------------------------------------------------------

def rotate_vector_axis_angle(v, axis, angle):
    '''
    Rotates vectors about axis by angle.

    '''
    rotmat = get_rotmat_axis_angle(axis, angle)
    return np.dot(v, rotmat.T)

#-------------------------------------------------------------------------------

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

#-------------------------------------------------------------------------------

def rotate_vectors_random(v):
    '''
    Rotates vectors about a random axis by random angle.
    v: (N,3) numpy array
    
    '''
    axis = get_ransphere(1.0)
    angle = 2*math.pi*np.random.random()
    axis, angle = fix_axis_angle(axis, angle)
    rotmat = get_rotmat_axis_angle(axis, angle)
    return np.dot(v, rotmat.T)

#-------------------------------------------------------------------------------

def get_frc_link(ri, theta):
    '''
    Given a segment r of a freely rotating chain, return the next link
    r such that rip1 = ri + l*r, where l is the length of the link

    '''
    phi = 2*math.pi*np.random.random()
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    
    rihat = ri/np.linalg.norm(ri)
    
    yhat = np.array([0, 1, 0])
    axis = np.cross(yhat, rihat)
    axis = axis/np.linalg.norm(axis)
    cos_angle = np.dot(rihat, yhat)
    angle = math.acos(cos_angle)
    axis, angle = fix_axis_angle(axis, angle)
    
    r = np.array([stheta*cphi, ctheta, stheta*sphi])
    r = rotate_vector_axis_angle(r, axis, angle)
    
    #b = r
    #b = b/np.linalg.norm(b)
    #cos_theta = np.dot(a, b)
    #print(math.acos(cos_theta)*180.0/math.pi)
    return r

#-------------------------------------------------------------------------------

