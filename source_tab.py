#!/usr/bin/env python
import numpy as np 
import sys
from colossus.cosmology import cosmology
from scipy.stats import poisson

import gen_mock_om10p
import source_qso
import source_sn

#
# make source table
#
def make_srctab(mmax, fov, flag_type_min, flag_type_max, cosmo):
    dm = 0.02
    dz = 0.01

    m = np.arange(14.0 + 0.5 * dm, mmax, dm)
    z = np.arange(0.1, 5.499, dz)
    mm, zz = np.meshgrid(m, z)
    fmm = mm.flatten()
    fzz = zz.flatten()

    m_all = []
    z_all = []
    f_all = []
    for i in range(flag_type_min, flag_type_max + 1):
        nn = fov * dndzdmobs(fmm, fzz, i, cosmo) * dm * dz
        n = np.random.poisson(nn)
        
        cut_n = n[n > 0]
        cut_m = fmm[n > 0]
        cut_z = fzz[n > 0]
        
        for j in range(len(cut_n)):
            m_all.append([cut_m[j]] * cut_n[j])
            z_all.append([cut_z[j]] * cut_n[j])
            f_all.append([i] * cut_n[j])

    m_tab = np.array([x for xx in m_all for x in xx])
    z_tab = np.array([x for xx in z_all for x in xx])
    f_tab = np.array([x for xx in f_all for x in xx])

    return m_tab, z_tab, f_tab
        
#
# number count of source
#
def dndzdmobs(m, z, flag_type, cosmo):
    if flag_type == 0:
        ma = source_qso.mtoma_qso(m, z, cosmo)
        return source_qso.lf_func_qso(ma, z, cosmo) * calc_vol(z, cosmo)
    else:
        dm = source_sn.mtodm_sn(m, z, flag_type, cosmo)
        return source_sn.lf_func_sn(dm, z, flag_type, cosmo) * calc_vol(z, cosmo) / (1.0 + z)
#
# calculate comoving colume element
#
def calc_vol(z, cosmo):
    dis = cosmo.angularDiameterDistance(z) / (cosmo.H0 / 100.0)
    drdz = (2997.92458 / ((1.0 + z) * cosmo.Ez(z))) / (cosmo.H0 / 100.0)
  
    return (dis * dis / 3283.0) * drdz * (1.0 + z) * (1.0 + z) * (1.0 + z)

#
# checking total number fo sources
# 
def n_src_tot(mlim, fov, flag_type_min, flag_type_max, cosmo):
    dm = 0.02
    dz = 0.01

    m = np.arange(14.0 + 0.5 * dm, mlim, dm)
    z = np.arange(0.1, 5.499, dz)
    mm, zz = np.meshgrid(m, z)
    fmm = mm.flatten()
    fzz = zz.flatten()

    ntot = 0.0
    for i in range(flag_type_min, flag_type_max + 1):
        nn = fov * dndzdmobs(fmm, fzz, i, cosmo) * dm * dz
        ntot = ntot + np.sum(nn)

    return ntot
#
# for checks
#
if __name__ == '__main__':
    cosmo = gen_mock_om10p.init_cosmo()

    source_qso.set_spline_kcor_qso()
    source_sn.set_spline_kcor_sn(1)
    
    #print(dndzdmobs(np.array([22.0, 23.0]), np.array([1.4, 2.7]), 2, cosmo))
    #make_srctab(27.0, 20000.0, 0, 0, cosmo)

    # number of unlensed sources for full LSST
    # QSOs (cf. QSO (measured) Nnon-lens in Table 2 of OM10)
    print('%e ' % n_src_tot(23.3, 20000.0, 0, 0, cosmo))
    # SNe Ia (cf. SN (Ia) Nnon-lens in Table 3 of OM10)
    print('%e ' % n_src_tot(22.6, 50000.0, 1, 1, cosmo))
    # SNe cc (cf. SN (cc) Nnon-lens in Table 3 of OM10)
    print('%e ' % n_src_tot(22.6, 50000.0, 2, 5, cosmo))
    
