# coding=utf8

""""
Functions for the cyclostationary S3 project
"""

import utils
import qnoise_fit
from pyHegel import fitting as fit
import numpy as np

def getOIP2fromMoments(s2, s3, **kwargs):
    """"
    Return an effective OIP2 in W, (or dBm if 'dBm'=True) from the moments S2 and S3
    """

    def funclin(x,a,b):
        return a*x+b

    data_fit = fit.fitcurve(funclin, s2**2, s3, [1,0], noadjust=[])
    oip2 = 81./2/(data_fit[0][0]**2)
    if kwargs.get('dBm', False):
        oip2 = np.log10(oip2*1000)*10
    return oip2, data_fit

def calcS3env(Idc,I3, R, Tj, Genv):
    """"
    Computes effect on S3 due to finite temperature of the environment when noise is adiabaticaly modulated
    """
    t = np.linspace(0, 1, 1000, endpoint=False)
    It = Idc[:, None]+I3*np.cos(2*np.pi*t[None, :])
    Dt = qnoise_fit.dSiidV(It, R, Tj)
    D3 = qnoise_fit.Fcoef(Dt, t, 1)[0]
    return Genv*D3


def calcSfb(Idc,I3, R, Tj,g1,g2,g3):
    """"
    Computes effect on S3 due to finite impedance of measurement circuit when noise is adiabaticaly modulated
    """
    t = np.linspace(0, 1, 1000, endpoint=False)
    It = Idc[:, None]+I3*np.cos(2*np.pi*t[None, :])
    St = qnoise_fit.Sii(It, R, Tj)
    Dt = qnoise_fit.dSiidV(It, R, Tj)
    S0 = qnoise_fit.Fcoef(St, t, 0)
    D3 = qnoise_fit.Fcoef(Dt, t, 1)[0]
    S3 = qnoise_fit.Fcoef(St, t, 1)[0]
    D0 = qnoise_fit.Fcoef(Dt, t, 0)
    D6 = qnoise_fit.Fcoef(Dt, t, 2)[0]
    return g1*S0*D3+g2*S3*D0+g3*D6*S3


def calcS3eq(I3, R, T, G_env, G_lin):
    """"
    Computes expected signal on S3 with no dc bias with environment
    """
    t = np.linspace(0, 1, 1000, endpoint=False)
    It = I3[:, None]*np.cos(2*np.pi*t[None, :])
    Dt = qnoise_fit.dSiidV(It, R, T)
    D3 = qnoise_fit.Fcoef(Dt, t, 1)[0]
    return G_env*D3+G_lin*I3