"""
The "calculus_utils" module
---------------------------
contains math functions that can be repeatedly used.
"""

from math import *
import numpy as np
from dolfin import *

worldcomm = MPI.comm_world
selfcomm = MPI.comm_self

mpisize = MPI.size(worldcomm)
mpirank = MPI.rank(worldcomm)

def compute_rate(x,y):
    """
    Return the slope of two data points.

    Parameters
    ----------
    x : ndarray
    y : ndarray

    Returns
    -------
    res : float
    """
    return (y[1]-y[0])/(x[1]-x[0])

def vec_angle(vec1, vec2, degree=True):
    """
    Return the angle between two vectors.
    
    Parameters
    ----------
    vec1 : ndarray
    vec2 : ndarray
    degree : bool, optional, if True, return the angle in degree

    Returns
    -------
    theta : float
    """
    theta = np.arccos(np.dot(vec1, vec2)\
        /(np.linalg.norm(vec1)*np.linalg.norm(vec2)))
    if degree:
        theta = theta*180./np.pi
    return theta

def linear_extrapolation(x_extra, x_pts, y_pts):
    """
    Return the linear extrapolation for two points.

    Parameters
    ----------
    x_extra : float, extrapolated point
    x_pts : ndarray
    y_pts : ndarray

    Returns
    -------
    y_extra : float
    """
    y_extra = y_pts[0] + (x_extra - x_pts[0])/(x_pts[1] - x_pts[0])\
        *(y_pts[1] - y_pts[0])
    return y_extra

def relative_error(x, x_ref):
    """
    Return the relative error.

    Parameters
    ----------
    x : float
    x_ref : float

    Returns
    -------
    res : float
    """
    return abs(x-x_ref)/x_ref

def compute_element_length(data):
    """
    Return the distance between two adjacent points of ``data``.

    Parameters
    ----------
    data : ndarray

    Returns
    -------
    el_length : float
    """
    el_length = np.linalg.norm(np.diff(data, axis=0), axis=1)
    return el_length

def array_middle_points(data):
    """
    Return the middle points of an array.

    Parameters
    ----------
    data : ndarray

    Returns
    -------
    res : ndarray
    """
    res = np.zeros((data.shape[0]-1, data.shape[1]))
    for i in range(data.shape[0]-1):
        res[i, :] = (data[i, :] + data[i+1, :])/2
    return res

def extrapolate_array(vec, direction="both"):
    """
    Extrapolate a vector along the specified direction.

    Parameters
    ----------
    vec : ndarray
    direction : str, {"both", "right", "left"}, optional

    Returns
    -------
    extrapo_vec : ndarray
    """
    num_el = vec.shape[0]
    x_seq = np.linspace(0,num_el, num_el+1)
    if direction == "left":
        el_left = linear_extrapolation(x_seq[0]-1, x_seq[:2], vec[:2])
        extrapo_vec = np.concatenate((np.array([el_left]), vec))
    elif direction == "right":
        el_right = linear_extrapolation(x_seq[-1]+1, x_seq[-2:], vec[-2:])
        extrapo_vec = np.concatenate((vec, np.array([el_right])))
    elif direction == "both":
        el_left = linear_extrapolation(x_seq[0]-1, x_seq[:2], vec[:2])
        el_right = linear_extrapolation(x_seq[-1]+1, x_seq[-2:], vec[-2:])
        extrapo_vec = np.concatenate((np.array([el_left]), vec, 
                                      np.array([el_right])))
    else:
        if mpirank == 0:
            raise TypeError("Direction type {} is "
                            "not defined.".format(direction))
    return extrapo_vec

def remove_elements_by_value(data, value, axis=0, side=0):
    """
    Remove elements in the ``data`` when they are smaller or greater 
    than ``value`` in the specified axis.

    Parameters
    ----------
    data : ndarray
    value : float
    axis : int, 0, 1 or 2, optional, the column index 
    side : int, 0 or 1, optional
        If side=0, remove the elements that smaller than the value.
        if side=1, remove the elements that greater than the value.

    Returns
    -------
    res : ndarray
    """
    num_rows, num_cols = data.shape
    if axis > num_cols:
        if mpirank == 0:
            raise IndexError("axis index {} out of range.".format(axis))
    res_array = []

    if side == 0:
        for i in range(num_rows):
                if data[i,axis] > value:
                    res_array += [data[i],]
    elif side == 1:
        for i in range(num_rows):
                if data[i,axis] < value:
                    res_array += [data[i],]
    else:
        if mpirank == 0:
            raise IndexError("side index {} is not valid.".format(side))
    return np.array(res_array)

def sort_coord(coord, axis=0):
    """
    Sort the coordinates from min to max.

    Parameters
    ----------
    coord : ndarray
    axis : int, {0, 1, 2}, optional, default is 0

    Returns
    -------
    sorted_coord : ndarray
    """
    sort_ind = np.argsort(coord[:,axis])
    sorted_coord = coord[sort_ind]
    return sorted_coord

def normalize_diff(vec, axis=0):
    """
    Return normalized differentiation of a vector.
    
    Parameters
    ----------
    vec : ndarray
    axis : int, {0, 1, 2}, optional, default is 0

    Returns
    -------
    vec_res : ndarray
    """
    vec_diff = np.diff(vec, axis=axis)
    vec_res = np.zeros((vec.shape[0], vec.shape[1]))
    for i in range(vec_diff.shape[0]):
        vec_res[i,:] = vec_diff[i,:]/np.linalg.norm(vec_diff[i,:])
    return vec_res

if __name__ == '__main__':
    pass