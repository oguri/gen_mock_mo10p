#!/usr/bin/env python
import numpy as np 
import sys
import scipy.special as sc

import gen_mock_om10p
import source_tab

#
# number counts of lens galaxies
#
def dndzdv_lens(v, z, cosmo):
    vol = source_tab.calc_vol(z, cosmo)
 
    return vf(v, z, cosmo) * vol

def vf(v, z, cosmo):
    #return vf_sh(v, z, cosmo)
    #return vf_cp(v, z, cosmo)
    return vf_be(v, z, cosmo) * vf_to(v, z, cosmo) / vf_to(v, np.zeros_like(z), cosmo)

def vf_sh(v, z, cosmo):
    hh = (cosmo.H0 / 70.0) ** 3

    return schechter(v, (1.4e-3) * hh, 88.8, 6.5, 1.93)

def vf_be(v, z, cosmo):
    hh = (cosmo.H0 / 70.0) ** 3

    return schechter(v, (2.099e-2) * hh, 113.78, 0.94, 1.85)

def vf_cp(v, z, cosmo):
    hh = (cosmo.H0 / 70.0) ** 3

    return schechter(v, (2.74e-3) * hh, 161.0, 2.32, 2.67)

def vf_to(v, z, cosmo):
    a  = 10 ** (-1.753749 + 0.204934 * z -0.057387 * z * z)
    al = -1.793255 -0.337430 * z - z * z * 0.023378
    be = 0.443074 + 1.219927 * z -0.280891 * z * z
    ss = 10 ** (2.022011 -0.044166 * z + 0.007290 * z * z)

    vv = v / ss
    lv = np.log10(vv)
    
    return (a / ss) * (vv ** (al + be * lv - 1.0)) * np.exp((-1.0) * vv) * (vv - 2.0 * be * lv - al)

def schechter(v, ps, vs, a, b):
    ga = np.exp(sc.gammaln(a / b))

    return ps * ((v / vs) ** a) * np.exp((-1.0) * ((v / vs) ** b)) * (b / ga) / v

#
# for checks
#
if __name__ == '__main__':
    cosmo = gen_mock_om10p.init_cosmo()

    print(dndzdv_lens(125.0, 0.5, cosmo))
    
    
