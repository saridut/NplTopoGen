#!/usr/bin/env python

import math
import numpy as np


def unitized(r):
    v = np.copy(r)
    nrm = np.linalg.norm(v)
    if math.isclose(nrm, 0.0, rel_tol=1e-14):
        return v
    else:
        return v/np.linalg.norm(v)


def unitize(r):
    r /= np.linalg.norm(r)
    return r


def is_unitized(r):
    return math.isclose(np.linalg.norm(r), 1.0, rel_tol=1e-12)


def perm_tensor():
    '''
    This function returns the permutation tensor.
    '''
    epsilon = np.zeros((3,3,3))
    epsilon[1,2,0] =  1.0
    epsilon[2,1,0] = -1.0
    epsilon[0,2,1] = -1.0
    epsilon[2,0,1] =  1.0
    epsilon[0,1,2] =  1.0
    epsilon[1,0,2] = -1.0
    return epsilon


def cross_mat(r):
    mat = np.zeros((3,3))
    mat[0,1] = -r[2]
    mat[0,2] = r[1]
    mat[1,0] = r[2]
    mat[1,2] = -r[0]
    mat[2,0] = -r[1]
    mat[2,1] = r[0]
    return mat


def decomp_LTL(A, overwrite=False, zero_upper=True, out=None):
    m, n = A.shape
    assert m == n
    if overwrite:
        L = A
    else:
        if out is not None:
            assert n,n == out.shape
            out[:,:] = 0.0
            L = out
        else:
            L = np.zeros((n,n))

    for i in range(n-1,-1,-1):
        L[i,i] = math.sqrt(A[i,i] - np.dot(L[i+1:,i],L[i+1:,i]))
        for j in range(i):
            L[i,j] = (A[i,j] - np.dot(L[i+1:,i], L[i+1:,j]))/L[i,i]

    if overwrite and zero_upper:
        for i in range(n):
            for j in range(i+1,n):
                L[i,j] = 0.0
    return L


