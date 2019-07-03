# coding=utf8
""""
Useful functions to process 2D histograms obtained from homodyne measurements
"""

import numpy as np
import scipy.ndimage
from scipy.ndimage.interpolation import geometric_transform


def getDiffProb(hON, hOFF, center=(0, 0), dx=(1, 1), smoothing=None, **kwargs):
    def translate(img, x, y, order=1):
        def transform(coords):
            return coords[0] - x, coords[1] - y

        return geometric_transform(img, transform, order=order)

    center = np.array(center)
    dx = np.array(dx)

    x0 = center[0]
    y0 = center[1]
    n = 1
    imgON = (hON/np.sum(hON, axis=(0, 1)))[x0 - dx[0]:x0 + dx[0], y0 - dx[1]:y0 + dx[1]]
    imgOFF = (hOFF/np.sum(hON, axis=(0, 1)))[x0 - dx[0]:x0 + dx[0], y0 - dx[1]:y0 + dx[1]]
    vx = np.linspace(x0 - dx[0], x0 + dx[0] - 1, 2 * dx[0])
    vy = np.linspace(y0 - dx[1], y0 + dx[1] - 1, 2 * dx[1])

    mx = getMoment([imgON],vx,vy,1,0,centered=False)
    my = getMoment([imgON],vx,vy,0,1,centered=False)
    mx2 = getMoment([imgOFF],vx,vy,1,0,centered=False)
    my2 = getMoment([imgOFF],vx,vy,0,1,centered=False)

    imgON_corr = translate(imgON, 511-mx, 511-my)
    imgOFF_corr = translate(imgOFF, 511-mx2, 511-my2)
    if smoothing is None:
        return imgON_corr - imgOFF_corr
    else:
        return scipy.ndimage.filters.gaussian_filter(imgON_corr - imgOFF_corr, smoothing, mode='constant')



def getWan(histo_in, vx, vy, a, n, **kwargs):
    """"
    Gives the contribution of the n order rotational symmetry to the a order moment of 2D histograms
    """

    histo = np.transpose(np.array(histo_in), axes=(0,-1,-2))
    if kwargs.get('renorm', 'True'):
        w = np.sum(histo, axis=(-1, -2))
    else:
        w = np.ones(np.shape(histo)[0])
    xx, yy = np.meshgrid(vx, vy)

    mx = np.sum(histo / w[:, None, None] * xx[None, :, :], axis=(-1, -2))
    my = np.sum(histo / w[:, None, None] * yy[None, :, :], axis=(-1, -2))

    r = np.sqrt((xx[None, :, :] - mx[:, None, None]) ** 2 + (yy[None, :, :] - my[:, None, None]) ** 2)**a
    theta = np.arctan2((yy[None, :, :] - my[:, None, None]), (xx[None, :, :] - mx[:, None, None]))

    wa = np.sum(histo / w[:, None, None] * r * np.cos(n * theta), axis=(-1, -2)) / 2 / np.pi
    wb = np.sum(histo / w[:, None, None] * r * np.sin(n * theta), axis=(-1, -2)) / 2 / np.pi

    return np.array(wa), np.array(wb)


def getMoment(histo_in, vx, vy, n, m, *args, **kwargs):
    """"
    Computes moments <X^nY^m> of the probability distribution of a 2D histograms array
    """

    histo = np.transpose(np.array(histo_in), axes=(0,-1,-2))

    if kwargs.get('renorm', 'True'):
        w = np.sum(histo, axis=(-1, -2))
    else:
        w = np.ones(np.shape(histo)[0])

    xx, yy = np.meshgrid(vx, vy)

    if kwargs.get('centered', 'True'):
        mx = np.sum(histo / w[:, None, None] * xx[None, :, :], axis=(-1, -2))
        my = np.sum(histo / w[:, None, None] * yy[None, :, :], axis=(-1, -2))
    else:
        mx, my = np.zeros(np.shape(histo)[0]), np.zeros(np.shape(histo)[0])

    out = np.sum(histo / w[:, None, None] * (xx[None, :, :] - mx[:, None, None]) ** n
                 * (yy[None, :, :] - my[:, None, None]) ** m, axis=(-1, -2))

    return out


def importHisto(prefix, n0, n):
    """"
    Averages 2D histograms arrays from file
    prefix_str(n0)Fwd.npz to prefix_str(n0+n)Fwd.npz
    and
    prefix_str(n0)Rev.npz to prefix_str(n0+n)Rev.npz
    """
    print 'importing file:' + prefix + str(n0) + 'Fwd.npz'
    data_up = np.load(prefix + str(n0) + 'Fwd.npz')
    print 'importing file:' + prefix + str(n0) + 'Rev.npz'
    data_dn = np.load(prefix + str(n0) + 'Rev.npz')

    print 'Decompressing data'
    hON = data_up['syncON']
    hOFF = data_up['syncOFF']
    hON += data_dn['syncON'][::-1, :, :]
    hOFF += data_dn['syncOFF'][::-1, :, :]

    data_up.close()
    data_dn.close()
    for i in np.arange(n0 + 1, n0 + n):
        print 'importing file:' + prefix + str(i) + 'Fwd.npz'
        data_up = np.load(prefix + str(i) + 'Fwd.npz')
        print 'importing file:' + prefix + str(i) + 'Fwd.npz'
        data_dn = np.load(prefix + str(i) + 'Rev.npz')
        print 'Decompressing data'
        hON += data_up['syncON']
        hOFF += data_up['syncOFF']
        hON += data_dn['syncON'][::-1, :, :]
        hOFF += data_dn['syncOFF'][::-1, :, :]
        data_up.close()
        data_dn.close()
    print 'done'
    return hON / n / 2, hOFF / n / 2