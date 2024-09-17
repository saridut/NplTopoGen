#!/usr/bin/env python
import math
import numpy as np
import linalg as la

'''
Euler angle ranges:

XYZ, ZYX: phi in [-pi, pi],     theta in [-pi/2, pi/2], psi in [-pi, pi]
XZY, YZX: phi in [-pi, pi],     theta in [-pi, pi],     psi in [-pi/2, pi/2]
ZXY, YXZ: phi in [-pi/2, pi/2], theta in [-pi, pi],     psi in [-pi, pi]

Euler angle sequence: XYZ (world). First rotation about X, second rotation
about Y, and the third rotation about Z axis of the world(i.e. fixed) frame.
This is the same as the sequence used in Blender.
In contrast, the XYZ sequence is understood in the Aerospace community as:
First rotation about Z-axis, second rotation about Y-axis, and the third
rotation about X-axis of the body frame.

Ref: http://www.geometrictools.com/Documentation/EulerAngles.pdf
'''

def rotmat_euler(euler, seq='XYZ', world=True):
    rotmat_funcs = {'XYZ': rotmat_XYZ, 'XZY': rotmat_XZY,
                    'YXZ': rotmat_YXZ, 'YZX': rotmat_YZX,
                    'ZXY': rotmat_ZXY, 'ZYX': rotmat_ZYX
                    }
    if not world:
        euler = -euler
    phi, theta, psi = tuple(euler)
    rotmat = rotmat_funcs[seq](phi, theta, psi)
    if not world:
        rotmat = rotmat.T
    return rotmat


def factor_rotmat(rotmat, seq='XYZ', world=True):
    factor_rotmat_funcs = {
            'XYZ': factor_rotmat_XYZ, 'XZY': factor_rotmat_XZY,
            'YXZ': factor_rotmat_YXZ, 'YZX': factor_rotmat_YZX,
            'ZXY': factor_rotmat_ZXY, 'ZYX': factor_rotmat_ZYX
            }
    if not world:
        rotmat = rotmat.T
    factors = factor_rotmat_funcs[seq](rotmat) 
    if not world:
        factors = -factors
    return factors


def rotmat_XYZ(phi, theta, psi):
    rotmat = np.zeros((3,3)) 
    sin_phi = math.sin(phi)
    sin_theta = math.sin(theta)
    sin_psi = math.sin(psi)
    cos_phi = math.cos(phi)
    cos_theta = math.cos(theta)
    cos_psi = math.cos(psi)
    rotmat[0,0] = cos_theta*cos_psi
    rotmat[0,1] = sin_phi*sin_theta*cos_psi - cos_phi*sin_psi
    rotmat[0,2] = cos_phi*sin_theta*cos_psi + sin_phi*sin_psi
    rotmat[1,0] = cos_theta*sin_psi
    rotmat[1,1] = sin_psi*sin_theta*sin_phi + cos_phi*cos_psi
    rotmat[1,2] = cos_phi*sin_theta*sin_psi - sin_phi*cos_psi
    rotmat[2,0] = -sin_theta
    rotmat[2,1] = sin_phi*cos_theta
    rotmat[2,2] = cos_phi*cos_theta
    return rotmat


def factor_rotmat_XYZ(rotmat):
    if rotmat[2,0] < 1.0:
        if rotmat[2,0] > -1.0:
            theta = math.asin(-rotmat[2,0])
            psi = math.atan2(rotmat[1,0], rotmat[0,0])
            phi = math.atan2(rotmat[2,1], rotmat[2,2])
        else:
            #Not unique: phi - psi = atan2(-rotmat[1,2], rotmat[1,1])
            theta = math.pi/2
            psi = -math.atan2(-rotmat[1,2], rotmat[1,1])
            phi = 0.0
    else:
        #Not unique: phi + psi = atan2(-rotmat[1,2], rotmat[1,1])
        phi = 0.0
        theta = -math.pi/2
        psi = math.atan2(-rotmat[1,2], rotmat[1,1])
    return np.array([phi, theta, psi])


def rotmat_XZY(phi, theta, psi):
    rotmat = np.zeros((3,3)) 
    sin_phi = math.sin(phi)
    sin_theta = math.sin(theta)
    sin_psi = math.sin(psi)
    cos_phi = math.cos(phi)
    cos_theta = math.cos(theta)
    cos_psi = math.cos(psi)
    rotmat[0,0] = cos_theta*cos_psi
    rotmat[0,1] = sin_phi*sin_theta - cos_phi*cos_theta*sin_psi
    rotmat[0,2] = cos_phi*sin_theta + sin_phi*cos_theta*sin_psi
    rotmat[1,0] = sin_psi
    rotmat[1,1] = cos_phi*cos_psi
    rotmat[1,2] = -sin_phi*cos_psi
    rotmat[2,0] = -sin_theta*cos_psi
    rotmat[2,1] = sin_phi*cos_theta + cos_phi*sin_theta*sin_psi
    rotmat[2,2] = cos_phi*cos_theta - sin_phi*sin_theta*sin_psi
    return rotmat


def factor_rotmat_XZY(rotmat):
    if rotmat[1,0] < 1.0:
        if rotmat[1,0] > -1.0:
            phi = math.atan2(-rotmat[1,2], rotmat[1,1])
            theta = math.atan2(-rotmat[2,0], rotmat[0,0])
            psi = math.asin(rotmat[1,0])
        else:
            #Not unique: phi - theta = atan2(rotmat[2,1], rotmat[2,2])
            phi = 0.0
            theta = -math.atan2(rotmat[2,1], rotmat[2,2])
            psi = -math.pi/2
    else:
        #Not unique: phi + theta = atan2(rotmat[2,1], rotmat[2,2])
        phi = 0.0
        theta = math.atan2(rotmat[2,1], rotmat[2,2])
        psi = math.pi/2
    return np.array([phi, theta, psi])


def rotmat_YXZ(phi, theta, psi):
    rotmat = np.zeros((3,3)) 
    sin_phi = math.sin(phi)
    sin_theta = math.sin(theta)
    sin_psi = math.sin(psi)
    cos_phi = math.cos(phi)
    cos_theta = math.cos(theta)
    cos_psi = math.cos(psi)
    rotmat[0,0] = cos_theta*cos_psi - sin_phi*sin_theta*sin_psi
    rotmat[0,1] = -cos_phi*sin_psi
    rotmat[0,2] = sin_theta*cos_psi + sin_phi*cos_theta*sin_psi
    rotmat[1,0] = sin_phi*sin_theta*cos_psi + cos_theta*sin_psi
    rotmat[1,1] = cos_phi*cos_psi
    rotmat[1,2] = sin_theta*sin_psi - sin_phi*cos_theta*cos_psi
    rotmat[2,0] = -cos_phi*sin_theta
    rotmat[2,1] = sin_phi
    rotmat[2,2] = cos_phi*cos_theta
    return rotmat


def factor_rotmat_YXZ(rotmat):
    if rotmat[2,1] < 1.0:
        if rotmat[2,1] > -1.0:
            phi = math.asin(rotmat[2,1])
            theta = math.atan2(-rotmat[2,0], rotmat[2,2])
            psi = math.atan2(-rotmat[0,1], rotmat[1,1])
        else:
            #Not unique: theta - psi = atan2(rotmat[0,2], rotmat[0,0])
            phi = -math.pi/2
            theta = 0.0
            psi = -math.atan2(rotmat[0,2], rotmat[0,0])
    else:
        #Not unique: theta + psi = atan2(rotmat[0,2], rotmat[0,0])
        phi = math.pi/2
        theta = 0.0
        psi = math.atan2(rotmat[0,2], rotmat[0,0])
    return np.array([phi, theta, psi])


def rotmat_YZX(phi, theta, psi):
    rotmat = np.zeros((3,3)) 
    sin_phi = math.sin(phi)
    sin_theta = math.sin(theta)
    sin_psi = math.sin(psi)
    cos_phi = math.cos(phi)
    cos_theta = math.cos(theta)
    cos_psi = math.cos(psi)
    rotmat[0,0] = cos_theta*cos_psi
    rotmat[0,1] = -sin_psi
    rotmat[0,2] = sin_theta*cos_psi
    rotmat[1,0] = sin_phi*sin_theta + cos_phi*cos_theta*sin_psi
    rotmat[1,1] = cos_phi*cos_psi
    rotmat[1,2] = cos_phi*sin_theta*sin_psi - sin_phi*cos_theta
    rotmat[2,0] = sin_phi*cos_theta*sin_psi - cos_phi*sin_theta
    rotmat[2,1] = sin_phi*cos_psi
    rotmat[2,2] = sin_phi*sin_theta*sin_psi + cos_phi*cos_theta
    return rotmat


def factor_rotmat_YZX(rotmat):
    if rotmat[0,1] < 1.0:
        if rotmat[0,1] > -1.0:
            phi = math.atan2(rotmat[2,1], rotmat[1,1])
            theta = math.atan2(rotmat[0,2], rotmat[0,0])
            psi = math.asin(-rotmat[0,1])
        else:
            #Not unique: theta - phi = atan2(-rotmat[2,0], rotmat[2,2])
            phi = -math.atan2(-rotmat[2,0], rotmat[2,2])
            theta = 0.0
            psi = math.pi/2
    else:
        #Not unique: theta + phi = atan2(-rotmat[2,0], rotmat[2,2])
        phi = math.atan2(-rotmat[2,0], rotmat[2,2])
        theta = 0.0
        psi = -math.pi/2
    return np.array([phi, theta, psi])


def rotmat_ZXY(phi, theta, psi):
    rotmat = np.zeros((3,3)) 
    sin_phi = math.sin(phi)
    sin_theta = math.sin(theta)
    sin_psi = math.sin(psi)
    cos_phi = math.cos(phi)
    cos_theta = math.cos(theta)
    cos_psi = math.cos(psi)
    rotmat[0,0] = cos_theta*cos_psi + sin_phi*sin_theta*sin_psi
    rotmat[0,1] = sin_phi*sin_theta*cos_psi - cos_theta*sin_psi
    rotmat[0,2] = cos_phi*sin_theta
    rotmat[1,0] = cos_phi*sin_psi
    rotmat[1,1] = cos_phi*cos_psi
    rotmat[1,2] = -sin_phi
    rotmat[2,0] = sin_phi*cos_theta*sin_psi - sin_theta*cos_psi
    rotmat[2,1] = sin_phi*cos_theta*cos_psi + sin_theta*sin_psi
    rotmat[2,2] = cos_phi*cos_theta
    return rotmat


def factor_rotmat_ZXY(rotmat):
    if rotmat[1,2] < 1.0:
        if rotmat[1,2] > -1.0:
            phi = math.asin(-rotmat[1,2])
            theta = math.atan2(rotmat[0,2], rotmat[2,2])
            psi = math.atan2(rotmat[1,0], rotmat[1,1])
        else:
            #Not unique: psi - theta = atan2(-rotmat[0,1], rotmat[0,0])
            phi = math.pi/2
            theta = -math.atan2(-rotmat[0,1], rotmat[0,0])
            psi = 0.0
    else:
        #Not unique: psi + theta = atan2(-rotmat[0,1], rotmat[0,0])
        phi = -math.pi/2
        theta = math.atan2(-rotmat[0,1], rotmat[0,0])
        psi = 0.0
    return np.array([phi, theta, psi])


def rotmat_ZYX(phi, theta, psi):
    rotmat = np.zeros((3,3)) 
    sin_phi = math.sin(phi)
    sin_theta = math.sin(theta)
    sin_psi = math.sin(psi)
    cos_phi = math.cos(phi)
    cos_theta = math.cos(theta)
    cos_psi = math.cos(psi)
    rotmat[0,0] = cos_theta*cos_psi
    rotmat[0,1] = -cos_theta*sin_psi
    rotmat[0,2] = sin_theta
    rotmat[1,0] = sin_phi*sin_theta*cos_psi + cos_phi*sin_psi
    rotmat[1,1] = cos_phi*cos_psi - sin_phi*sin_theta*sin_psi
    rotmat[1,2] = -sin_phi*cos_theta
    rotmat[2,0] = sin_phi*sin_psi - cos_phi*sin_theta*cos_psi
    rotmat[2,1] = sin_phi*cos_psi + cos_phi*sin_theta*sin_psi
    rotmat[2,2] = cos_phi*cos_theta
    return rotmat


def factor_rotmat_ZYX(rotmat):
    if rotmat[0,2] < 1.0:
        if rotmat[0,2] > -1.0:
            phi = math.atan2(-rotmat[1,2], rotmat[2,2])
            theta = math.asin(rotmat[0,2])
            psi = math.atan2(-rotmat[0,1], rotmat[0,0])
        else:
            #Not unique: psi - phi = atan2(rotmat[1,0], rotmat[1,1])
            phi = -math.atan2(rotmat[1,0], rotmat[1,1])
            theta = -math.pi/2
            psi = 0.0
    else:
        #Not unique: psi + phi = atan2(rotmat[1,0], rotmat[1,1])
        phi = math.atan2(rotmat[1,0], rotmat[1,1])
        theta = math.pi/2
        psi = 0.0
    return np.array([phi, theta, psi])
    


