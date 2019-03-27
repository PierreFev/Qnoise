# coding=utf8

""""
Some usefull functions
"""

import numpy as np


def num_diff(x, y):
    """
    Numerical derivation dy/dx.
    """
    return np.gradient(y)/np.gradient(x)


def WtodBm(w):
    return 10*np.log10(w*1000.)


def dBmtoW(dbm):
    return 10**(dbm*0.1)/1000


def get_angle(x, y):
    """
    Return the unwrapped angle in radian
    """
    return np.unwrap(np.arctan2(x, y))


def get_mag(x, y):
    """
    Return the magnitude
    """
    return np.sqrt(np.array(x)**2+np.array(y)**2)


def rot(x, y, phi):
    """
    Applies a rotation matrix to x and y
    """
    return x*np.cos(phi)-y*np.sin(phi), x*np.sin(phi)+y*np.cos(phi)
