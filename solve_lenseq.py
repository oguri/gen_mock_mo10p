#!/usr/bin/env python
import numpy as np 
import sys
import scipy.stats as st
import scipy.special as sc

import gen_mock_om10p
import solve_lenseq_glafic
import solve_lenseq_lenstronomy

#
# solve lens equation
#
def calc_image(lens_par, tm, rt_range, flag_mag, flag_solver, cosmo):
    out_img, kapgam = solve_lenseq(lens_par, tm, rt_range, flag_solver, cosmo)
    
    nim = len(out_img)

    if nim <= 1:
        n = 0.0
        a = []
        return out_img, nim, n, n, n, a

    #mag = list(zip(*out_img))[2]
    #mag_abs = list(map(abs, mag))
    #mag_abs.sort(reverse = True)
    #mag_tot = sum(mag_abs)

    nim = len(out_img)
    xi = []
    yi = []
    mi = []
    for i in range(nim):
        xi.append(out_img[i][0])
        yi.append(out_img[i][1])
        mi.append(abs(out_img[i][2]))

    mi.sort(reverse = True)
    mag_tot = sum(mi)

    si = []
    for i in range(nim - 1):
        for j in range(i + 1, nim):
            si.append(((xi[i] - xi[j]) ** 2) + ((yi[i] - yi[j]) ** 2))

    sep = np.sqrt(max(si))

    if flag_mag > 0:
        ii = min(flag_mag, nim)
        mag = mi[ii - 1]
    else:
        mag = mag_tot
        
    if nim == 2:
        fr = mi[1] / mi[0]
    else:
        fr = 1.0

    return out_img, nim, sep, mag, fr, kapgam
        
def solve_lenseq(lens_par, tm, rt_range, flag_solver, cosmo,):
    if flag_solver == 0:
        return solve_lenseq_glafic.solve_lenseq_glafic(lens_par, tm, rt_range, cosmo)
    else: 
        return solve_lenseq_lenstronomy.solve_lenseq_lenstronomy(lens_par, tm, rt_range, cosmo)

#
# Einstein radius
#
def vtoein(v, zl, zs, cosmo):
    # valid only for flat universe
    rl = cosmo.comovingDistance(0.0, zl) / (cosmo.H0 / 100.0)
    rs = cosmo.comovingDistance(0.0, zs) / (cosmo.H0 / 100.0)
    
    dol = (1.0 / (1.0 + zl)) * rl 
    dos = (1.0 / (1.0 + zs)) * rs 
    dls = (1.0 / (1.0 + zs)) * (rs - rl)

    return vtoein_func(v, dos, dls)

def vtoein_func(v, dos, dls):
    vv = v / 2.9979e5
    rthe = 4.0 * np.pi * vv * vv * dls / dos

    # 206265.0 for radian -> arcsec conversion
    return rthe * 206265.0

#
# generate lens parameters
#
def set_lenspar(n):
    e_tab  = gene_e(n)
    te_tab = gene_ang(n) 

    return e_tab, te_tab
    
def set_shear(z):
    g  = gene_gam(z)
    tg = gene_ang(1)[0]

    return g, tg
    
def gene_e(n):
    em = 0.3
    se = 0.16

    e = st.truncnorm.rvs((0.0 - em) / se, (0.9 - em) / se, loc = em, scale = se, size = n)

    return e

def gene_gam(z):
    # old version
    #lgm = np.log10(0.05)
    #slg = 0.2
    #
    #lg = st.truncnorm.rvs(-1.0e99, (np.log10(0.9) - lgm) / slg, loc = lgm, scale = slg, size = n)
    #
    #return 10 ** lg
    #
    if z < 1.0:
        sig = 0.023 * z
    else:
        sig = 0.023 + 0.032 * np.log(z)

    g1 = st.truncnorm.rvs(-0.5 / sig, 0.5 / sig, loc = 0.0, scale = sig, size = 1)[0]
    g2 = st.truncnorm.rvs(-0.5 / sig, 0.5 / sig, loc = 0.0, scale = sig, size = 1)[0]

    return np.sqrt(g1 * g1 + g2 * g2)

def gene_ang(n):
    return (np.random.rand(n) - 0.5) * 360.0 

#
# dynamical normalization
# 
def bb_fac(e):
    return np.sqrt(0.5 * bb_obl(e) + 0.5 * bb_pro(e))

def bb_obl(e):
    a1 = 0.115
    a2 = 0.217
    a3 = 0.696

    return np.exp(a1 * np.sqrt(e) + a2 * e * e + a3 * (e ** 5))

def bb_pro(e):
    a1 = 0.258
    a2 = 0.827
    a3 = 6.0

    return 1.0 - a1 * e + a2 * (e ** a3)

#
# for checks
#
if __name__ == '__main__':
    cosmo = gen_mock_om10p.init_cosmo()

    #print(bb_fac(0.283))

    lens_par = [0.5, 300.0, 2.0, 0.3, 15.0, 0.1, -50.0, 0.1, -0.05]
    tm = vtoein(lens_par[1], lens_par[0], lens_par[2], cosmo)
    #print(tm)
    rt_range = 4.0
    flag_solver = 0
    out_img, kapgam = solve_lenseq(lens_par, tm, rt_range, flag_solver, cosmo)
    print(out_img)
    #print(kapgam)
    #print(len(out_img))
    #print(out_img[1][0], out_img[1][1], out_img[1][2])

    #out, n, sep, mag, fr, kapgam = calc_image(lens_par, tm, 4.0, 3, 1, cosmo)
    #print(n, sep, mag, fr)
    
    
