#!/usr/bin/env python
import math
import numpy as np

import linalg as la
import eulang

#Euler angle sequence: XYZ (world). First rotation about X, second rotation
#about Y, and the third rotation about Z axis of the world(i.e. fixed) frame.
#This is the same as the sequence used in Blender.
#In contrast, the XYZ sequence is understood in the Aerospace community as:
#First rotation about Z-axis, second rotation about Y-axis, and the third
#rotation about X-axis of the body frame.


#Axis_angle------------------------------------------------------------
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


def get_rand_axis_angle():
    '''
    Generates a random pair of axis-angle. The axis is a random vector from
    the surface of a unit sphere. Algorithm from Allen & Tildesley p. 349.
    '''
    axis = np.zeros((3,))
    #Generate angle: A uniform random number from [0.0, 2*pi)
    angle = 2.0*math.pi*np.random.random()
    while True:
        #Generate two uniform random numbers from [-1, 1)
        zeta1 = 2.0*np.random.random() - 1.0
        zeta2 = 2.0*np.random.random() - 1.0
        zetasq = zeta1**2 + zeta2**2
        if zetasq <= 1.0:
            break
    rt = np.sqrt(1.0-zetasq)
    axis[0] = 2.0*zeta1*rt
    axis[1] = 2.0*zeta2*rt
    axis[2] = 1.0 - 2.0*zetasq
    return fix_axis_angle(axis, angle)


def axis_angle_to_quat(axis, angle):
    w = math.cos(angle/2)
    v = math.sin(angle/2)*axis
    q = np.array([w, v[0], v[1], v[2]])
    return normalize_quat(q)


def axis_angle_to_euler(axis, angle, seq='XYZ', world=True):
    rotmat = get_rotmat_axis_angle(axis, angle)
    euler = factorize_rotmat(rotmat, seq=seq, world=world)
    return euler


def axis_angle_to_dcm(axis, angle):
    dcm = get_shiftmat_axis_angle(axis, angle, forward=True)
    return dcm


def any_to_axis_angle(orientation):
    ori_repr = orientation['repr']
    if ori_repr == 'quat':
        quat = np.array(orientation['quat'])
        axis, angle = quat_to_axis_angle(quat)
    elif ori_repr == 'euler':
        euler = np.array(orientation['euler'])
        seq = orientation['seq']
        world = orientation['world']
        axis, angle = euler_to_axis_angle(euler, seq=seq, world=world)
    elif ori_repr == 'axis_angle':
        axis = np.array(orientation['axis'])
        angle = orientation['angle']
    elif ori_repr == 'dcm':
        axis, angle = dcm_to_axis_angle(orientation['dcm'])
    else:
        raise ValueError(
            'Unrecognized orientation repr {0}'.format(ori_repr))
    return axis, angle


def rotate_vector_axis_angle(v, axis, angle):
    '''
    Rotates vectors about axis by angle.

    '''
    rotmat = get_rotmat_axis_angle(axis, angle)
    return np.dot(v, rotmat.T)


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


def extract_axis_angle_from_rotmat(rotmat):
    trace = np.trace(rotmat)
    angle = math.acos((trace-1)/2)
    if angle > 0:
        if angle < math.pi:
            u0 = rotmat[2,1] - rotmat[1,2]
            u1 = rotmat[0,2] - rotmat[2,0]
            u2 = rotmat[1,0] - rotmat[0,1]
        else:
            #Find the largest entry in the diagonal of rotmat
            k = np.argmax(np.diag(rotmat))
            if k == 0:
                u0 = math.sqrt(rotmat[0,0]-rotmat[1,1]-rotmat[2,2]+1)/2
                s = 1.0/(2*u0)
                u1 = s*rotmat[0,1]
                u2 = s*rotmat[0,2]
            elif k == 1:
                u1 = math.sqrt(rotmat[1,1]-rotmat[0,0]-rotmat[2,2]+1)/2
                s = 1.0/(2*u1)
                u0 = s*rotmat[0,1]
                u2 = s*rotmat[1,2]
            elif k == 2:
                u2 = math.sqrt(rotmat[2,2]-rotmat[0,0]-rotmat[1,1]+1)/2
                s = 1.0/(2*u2)
                u0 = s*rotmat[0,2]
                u1 = s*rotmat[1,2]
    else:
        u0 = 1.0
        u1 = 0.0
        u2 = 0.0
    return fix_axis_angle(np.array([u0, u1, u2]), angle, normalize=True)


def shift_vector_axis_angle(v, axis, angle, forward=False):
    shiftmat = get_shiftmat_axis_angle(axis, angle, forward=forward)
    return np.dot(v, shiftmat.T)


def shift_tensor2_axis_angle(a, axis, angle, forward=False):
    shiftmat = get_shiftmat_axis_angle(axis, angle, forward=forward)
    return np.einsum('ip,jq,pq', shiftmat, shiftmat, a)


def shift_tensor3_axis_angle(a, axis, angle, forward=False):
    shiftmat = get_shiftmat_axis_angle(axis, angle, forward=forward)
    return np.einsum('ip,jq,kr,pqr', shiftmat, shiftmat, shiftmat, a)


def get_shiftmat_axis_angle(axis, angle, forward=False):
    shiftmat = get_rotmat_axis_angle(-axis, angle) 
    if not forward:
        shiftmat = shiftmat.T
    return shiftmat


#Direction cosine matrix-----------------------------------------------
def dcm_from_axes(A, B):
    '''
    Returns the direction cosine matrix of axes(i.e. frame) B w.r.t.
    axes(i.e. frame) A.

    Parameters
    ----------
    A : (3,3) ndarray
        The rows of A represent the orthonormal basis vectors of frame A.

    B : (3,3) ndarray
        The rows of B represent the orthonormal basis vectors of frame B.

    Returns
    -------
    (3,3) ndarray
        The dcm of frame B w.r.t. frame A.

    '''
    return np.dot(B, A.T)


def dcm_to_quat(dcm):
    mat = get_rotmat_dcm(dcm)
    axis, angle = extract_axis_angle_from_rotmat(mat)
    return axis_angle_to_quat(axis, angle)


def dcm_to_euler(dcm, seq='XYZ', world=True):
    mat = get_rotmat_dcm(dcm)
    euler = factorize_rotmat(mat, seq=seq, world=world)
    return euler


def dcm_to_axis_angle(dcm):
    mat = get_rotmat_dcm(dcm)
    axis, angle = extract_axis_angle_from_rotmat(mat)
    return (axis, angle)


def any_to_dcm(orientation):
    ori_repr = orientation['repr']
    if ori_repr == 'quat':
        quat = np.array(orientation['quat'])
        dcm = quat_to_dcm(quat)
    elif ori_repr == 'euler':
        euler = np.array(orientation['euler'])
        seq = orientation['seq']
        world = orientation['world']
        dcm = euler_to_dcm(euler, seq=seq, world=world)
    elif ori_repr == 'axis_angle':
        axis = np.array(orientation['axis'])
        angle = orientation['angle']
        dcm = axis_angle_to_dcm(axis, angle)
    elif ori_repr == 'dcm':
        dcm = dcm_to_quat(orientation['dcm'])
    else:
        raise ValueError(
            'Unrecognized orientation repr {0}'.format(ori_repr))
    return dcm


def rotate_vector_dcm(v, dcm):
    rotmat = get_rotmat_dcm(dcm)
    return np.dot(v, rotmat.T)


def get_rotmat_dcm(dcm):
    return dcm.T


def shift_vector_dcm(v, dcm, forward=False):
    shiftmat = get_shiftmat_dcm(dcm, forward=forward)
    return np.dot(v, shiftmat.T)


def shift_tensor2_dcm(a, dcm, forward=False):
    shiftmat = get_shiftmat_dcm(dcm, forward=forward)
    return np.einsum('ip,jq,pq', shiftmat, shiftmat, a)


def shift_tensor3_dcm(a, dcm, forward=False):
    shiftmat = get_shiftmat_dcm(dcm, forward=forward)
    return np.einsum('ip,jq,kr,pqr', shiftmat, shiftmat, shiftmat, a)


def get_shiftmat_dcm(dcm, forward=False):
    shiftmat = dcm
    if not forward:
        shiftmat = shiftmat.T
    return shiftmat


#Euler angle-----------------------------------------------------------
def factorize_rotmat(rotmat, seq='XYZ', world=True):
    return eulang.factor_rotmat(rotmat, seq=seq, world=world)


def euler_to_euler(euler, seq, world, to_seq, to_world):
    rotmat = get_rotmat_euler(euler, seq=seq, world=world)
    return factorize_rotmat(rotmat, seq=to_seq, world=to_world)


def euler_to_quat(euler, seq='XYZ', world=True):
    axis, angle = euler_to_axis_angle(euler, seq=seq, world=world)
    return axis_angle_to_quat(axis, angle)


def euler_to_dcm(euler, seq='XYZ', world=True):
    dcm = get_shiftmat_euler(euler, seq=seq, world=world, forward=True)
    return dcm


def euler_to_axis_angle(euler, seq='XYZ', world=True):
    rotmat = get_rotmat_euler(euler, seq=seq, world=world)
    axis, angle = extract_axis_angle_from_rotmat(rotmat)
    return (axis, angle)


def any_to_euler(orientation, to_seq, to_world):
    ori_repr = orientation['repr']
    if ori_repr == 'quat':
        quat = np.array(orientation['quat'])
        euler = quat_to_euler(quat, seq=to_seq, world=to_world)
    elif ori_repr == 'euler':
        euler = np.array(orientation['euler'])
        seq = orientation['seq']
        world = orientation['world']
        euler = euler_to_euler(euler, seq, world, to_seq, to_world)
    elif ori_repr == 'axis_angle':
        axis = np.array(orientation['axis'])
        angle = orientation['angle']
        euler = axis_angle_to_euler(axis, angle, seq=to_seq, world=to_world)
    elif ori_repr == 'dcm':
        euler = dcm_to_euler(orientation['dcm'], seq=to_seq, world=to_world)
    else:
        raise ValueError(
            'Unrecognized orientation repr {0}'.format(ori_repr))
    return euler


def rotate_vector_euler(v, euler, seq='XYZ', world=True):
    '''
    Rotates vectors about axis by angle.

    '''
    rotmat = get_rotmat_euler(euler, seq=seq, world=world)
    return np.dot(v, rotmat.T)


def get_rotmat_euler(euler, seq='XYZ', world=True):
    return eulang.rotmat_euler(euler, seq=seq, world=world)


def shift_vector_euler(v, euler, seq='XYZ', world=True, forward=False):
    shiftmat = get_shiftmat_euler(euler, seq=seq, world=world, forward=forward)
    return np.dot(v, shiftmat.T)


def shift_tensor2_euler(a, euler, forward=False):
    shiftmat = get_shiftmat_euler(euler, forward=forward)
    return np.einsum('ip,jq,pq', shiftmat, shiftmat, a)


def shift_tensor3_euler(a, euler, forward=False):
    shiftmat = get_shiftmat_euler(euler, forward=forward)
    return np.einsum('ip,jq,kr,pqr', shiftmat, shiftmat, shiftmat, a)


def get_shiftmat_euler(euler, seq='XYZ', world=True, forward=False):
    rotmat = get_rotmat_euler(euler, seq=seq, world=world)
    if forward:
        shiftmat = rotmat.T
    else:
        shiftmat = rotmat
    return shiftmat


#Quaternion-----------------------------------------------------------

def get_rand_quat():
    q = np.random.random((4,))
    return normalize_quat(q)


def get_identity_quat():
    return np.array([1.0, 0.0, 0.0, 0.0])


def get_rand_quat():
    axis, angle = get_rand_axis_angle()
    return axis_angle_to_quat(axis, angle)


def get_perturbed_quat(q):
    raise NotImplementedError


def quat_to_axis_angle(q):
    angle = 2*math.acos(q[0])
    sin = math.sqrt(1.0-q[0]**2)
    if angle > 0.0:
        if angle < math.pi:
            axis = q[1:4]/sin
        else:
            rotmat = get_rotmat_quat(q)
            axis, angle = extract_axis_angle_from_rotmat(rotmat)
    else:
        axis = np.array([1.0, 0.0, 0.0])
    return fix_axis_angle(axis, angle, normalize=True)


def quat_to_euler(q, seq='XYZ', world=True):
    rotmat = get_rotmat_quat(q)
    return factorize_rotmat(rotmat, seq=seq, world=world)


def quat_to_dcm(q):
    return get_shiftmat_quat(q, forward=True)


def any_to_quat(orientation):
    ori_repr = orientation['repr']
    if ori_repr == 'quat':
        quat = np.array(orientation['quat'])
    elif ori_repr == 'euler':
        euler = np.array(orientation['euler'])
        seq = orientation['seq']
        world = orientation['world']
        quat = euler_to_quat(euler, seq=seq, world=world)
    elif ori_repr == 'axis_angle':
        axis = np.array(orientation['axis'])
        angle = orientation['angle']
        quat = axis_angle_to_quat(axis, angle)
    elif ori_repr == 'dcm':
        quat = dcm_to_quat(orientation['dcm'])
    else:
        raise ValueError(
            'Unrecognized orientation repr {0}'.format(ori_repr))
    return quat


def rotate_vector_quat(v, q):
    rotmat = get_rotmat_quat(q)
    return np.dot(v, rotmat.T)


def get_rotmat_quat(q):
    rotmat = np.empty((3,3))

    q0sq = q[0]**2
    q1sq = q[1]**2
    q2sq = q[2]**2
    q3sq = q[3]**2
    q0q1 = q[0]*q[1]
    q0q2 = q[0]*q[2]
    q0q3 = q[0]*q[3]
    q1q2 = q[1]*q[2]
    q1q3 = q[1]*q[3]
    q2q3 = q[2]*q[3]

    rotmat[0,0] = 2*(q0sq + q1sq) - 1.0
    rotmat[0,1] = 2*(q1q2 - q0q3)
    rotmat[0,2] = 2*(q1q3 + q0q2)
    rotmat[1,0] = 2*(q1q2 + q0q3)
    rotmat[1,1] = 2*(q0sq + q2sq) - 1.0
    rotmat[1,2] = 2*(q2q3 - q0q1)
    rotmat[2,0] = 2*(q1q3 - q0q2)
    rotmat[2,1] = 2*(q2q3 + q0q1)
    rotmat[2,2] = 2*(q0sq + q3sq) - 1.0
    return rotmat


def shift_vector_quat(v, q, forward=False):
    shiftmat = get_shiftmat_quat(q, forward=forward)
    return np.dot(v, shiftmat.T)


def shift_tensor2_quat(a, quat, forward=False):
    shiftmat = get_shiftmat_quat(quat, forward=forward)
    return np.einsum('ip,jq,pq', shiftmat, shiftmat, a)


def shift_tensor3_quat(a, quat, forward=False):
    shiftmat = get_shiftmat_quat(quat, forward=forward)
    return np.einsum('ip,jq,kr,pqr', shiftmat, shiftmat, shiftmat, a)


def get_shiftmat_quat(q, forward=False):
    if forward:
        shiftmat = get_rotmat_quat(get_conjugated_quat(q))
    else:
        shiftmat = get_rotmat_quat(q)
    return shiftmat


def conjugate_quat(q):
    '''
    Conjugates a quaternion in-place.

    '''
    q[1:4] = -q[1:4]
    return q


def get_conjugated_quat(q):
    '''
    Conjugates a quaternion and returns a copy.
    '''
    p = np.copy(q)
    p[1:4] = -p[1:4]
    return p


def invert_quat(q):
    '''
    Inverts a quaternion in-place.

    '''
    return conjugate_quat(q)


def get_inverted_quat(q):
    '''
    Inverts a quaternion and returns it as a new instance.

    '''
    p = np.copy(q)
    return conjugate_quat(p)


def normalize_quat(q):
    '''
    Normalizes a quaternion in-place.

    '''
    q /= np.linalg.norm(q)
    return q


def get_normalized_quat(q):
    '''
    Normalizes a quaternion and returns it as a copy.

    '''
    p = np.copy(q)
    return normalize_quat(p)


def quat_is_normalized(q):
    norm = np.linalg.norm(q)
    if math.isclose(norm, 1.0, rel_tol=1e-14):
        return True
    else:
        return False


def get_quat_prod(p, q):
    p0, p1, p2, p3 = tuple(p)
    prod_mat = np.array([[p0, -p1, -p2, -p3],
                         [p1,  p0, -p3,  p2],
                         [p2,  p3,  p0, -p1],
                         [p3, -p2,  p1,  p0]])
    pq = normalize_quat(np.dot(prod_mat, q))
    return pq


def interpolate_quat(q1, q2, t):
    theta = get_angle_between_quat(q1, q2)
    q = (q1*math.sin((1.0-t)*theta)
            + q2*math.sin(t*theta))/math.sin(theta)
    return normalize_quat(q)


def get_angle_between_quat(p, q):
    '''
    Returns the angle between two quaternions p and q.

    '''
    return math.acos(np.dot(p,q))


def quat_deriv_to_ang_vel(q, qdot):
    mat = quat_deriv_to_ang_vel_mat(q)
    return np.dot(mat, qdot)


def quat_deriv_to_ang_vel_mat(q):
    q0, q1, q2, q3 = tuple(q)
    return 2*np.array([[-q1,  q0, -q3,  q2],
                       [-q2,  q3,  q0, -q1],
                       [-q3, -q2,  q1,  q0]])


def ang_vel_to_quat_deriv(q, ang_vel):
    mat = ang_vel_to_quat_deriv_mat(q)
    qdot = np.dot(mat, ang_vel)
    return qdot


def ang_vel_to_quat_deriv_mat(q):
    q0, q1, q2, q3 = tuple(q)
    return 0.5*np.array([[-q1, -q2, -q3],
                         [ q0,  q3, -q2],
                         [-q3,  q0,  q1],
                         [ q2, -q1,  q0]])


#Other functions------------------------------------------------------
def translate(v, delta):
    '''
    Translates vectors inplace by delta.

    '''
    n = v.shape[0]
    for i in range(n):
        v[i,:] += delta
    return v


def align(v, old, new):
    '''
    old and new represent coordinate axes. They must be unit vectors.

    '''
    assert old.shape[0] == new.shape[0]
    n = old.shape[0]
    if n == 1:
        angle = math.acos(np.dot(old, new))
        axis = la.unitized(np.cross(old, new))
        return rotate_vector_axis_angle(v, axis, angle)
    elif n == 2:
        z_old = la.unitized(np.cross(old[0,:], old[1,:]))
        z_new = la.unitized(np.cross(new[0,:], new[1,:]))
        axes_old = np.vstack((old, z_old))
        axes_new = np.vstack((new, z_new))
        dcm = dcm_from_axes(axes_old, axes_new)
        return rotate_vector_dcm(v, dcm)
    elif n == 3:
        dcm = dcm_from_axes(old, new)
        return rotate_vector_dcm(v, dcm)


def mat_is_dcm(mat):
    return mat_is_rotmat(mat)


def mat_is_rotmat(mat):
    det_is_one = math.isclose(np.linalg.det(mat), 1.0, abs_tol=1e-12, rel_tol=1e-12)
    is_orthogonal = np.allclose(np.dot(mat, mat.T), np.identity(3))
    return is_orthogonal and det_is_one
