# coding=utf8

""""
Functions for the cyclostationary S3 project
"""

import utils
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

