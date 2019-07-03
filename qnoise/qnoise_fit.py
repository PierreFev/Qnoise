# coding=utf8

""""
Fit functions for quantum noise
"""

import numpy as np
import scipy as sp
import scipy.constants as C
import scipy.special





def xcothx(x):
    """
    This functions returns x/tanh(x) and deals with singularity at x=0.
    Useful for finite temperature shot-noise formulas
    """
    x = np.asanyarray(x)
    # remove 0 from x
    nx = np.where(x == 0, 1e-16, x)
    return nx/np.tanh(nx)


def dxcothx(x):
    """
    Returns the derivative of x/tanh(x) with x, and deals with singularity at x=0.
    Useful for finite temperature shot-noise formulas
    """
    x = np.asanyarray(x)
    # remove 0 from x
    nx = np.where(abs(x) < 1e-16, 1e-16, x)
    return 1./np.tanh(nx)+nx*(1-1./np.tanh(nx)**2)


def Sii(I,R,T):
    """
    non-symmetrized noise spectral density at zero frequency for a tunnel junction
    """
    return 2*C.k*T/R*xcothx(C.e*R*I/(2*C.k*T))


def dSiidV(I,R,T):
    """
    derivative with voltage of the non-symmetrized noise spectral density at zero frequency for a tunnel junction
    """
    x = C.e*I*R/(2*C.k*T)
    return C.e/R*dxcothx(x)


def SiiHarm(I,R,T,dI,n):
    """
    Fourier coefficients of Sii(V(t)) when V(t) is a sinewave.
    """
    t = np.linspace(0, 1, 500, endpoint=False)
    It = I[:, None]+dI*np.cos(2*np.pi*t[None, :])
    St = Sii(It, R, T)
    return Fcoef(St, t, n)


def dSiiHarm(I,R,T,dI,n):
    """
    Fourier coefficients of dSii/dV (V(t)) when V(t) is a sinewave.
    """
    t = np.linspace(0, 1, 500, endpoint=False)
    It = I[:, None]+dI*np.cos(2*np.pi*t[None, :])
    Dt = dSiidV(It, R, T)
    return Fcoef(Dt, t, n)


def Fcoef(Dt, t, n):
    """
    Fourier coefficients of a time serie (for equally spaced samples).
    """
    if n == 0:
        return np.sum(Dt, axis=-1)/len(t)
    else:
        A = 2*np.sum(Dt*np.cos(n*2*np.pi*t), axis=-1)/len(t)
        B = 2*np.sum(Dt*np.sin(n*2*np.pi*t), axis=-1)/len(t)
        return [A, B]


def QnoisefitI(I, f, R, T, *args, **kwargs):
    A = xcothx((C.e*I*R-C.h*f)/(2*C.k*T))+ xcothx((C.e*I*R+C.h*f)/(2*C.k*T))
    return A*C.k*T/R

def qNoise_phAss(I,f,R,T,Iac,f0,N=5):
    """
    Return photoassisted shot-noise for a tunnel junction in the tunneling regime
    """
    hf = C.h*f0
    kbt = C.k*T
    ev = C.e*I*R
    vac = C.e*Iac*R/hf
    n = np.arange(-N, N+1)[:, None]
    x = (ev+C.h*f-n*hf)/(2.*kbt)
    x2 = (ev-C.h*f-n*hf)/(2.*kbt)
    tmp = sp.special.jv(n, vac)**2 * (xcothx(x)+xcothx(x2))*(C.k*T)/R
    return tmp.sum(axis=0)

def dummy_func():
    """
    Dummy function description
    """
    pass
dummy_func.display_str = 'dummy'


