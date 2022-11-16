#!/usr/bin/env python
import numpy as np 

import solve_lenseq
import glafic

def solve_lenseq_glafic(lens_par, tm, rt_range, cosmo):
    glafic.init(cosmo.Om0, 1.0 - cosmo.Om0, -1.0, cosmo.H0 / 100.0, 'out', (-1.0) * tm * (rt_range + 3.0), (-1.0) * tm * (rt_range + 3.0), tm * (rt_range + 3.0), tm * (rt_range + 3.0), tm / 5.0, tm / 5.0, 5, verb = 0)

    glafic.startup_setnum(2, 0, 0)
    glafic.set_lens(1, 'sie',  lens_par[0], lens_par[1] * solve_lenseq.bb_fac(lens_par[3]), 0.0, 0.0, lens_par[3], lens_par[4], 0.0, 0.0)
    #glafic.set_lens(1, 'sie',  lens_par[0], lens_par[1], 0.0, 0.0, lens_par[3], lens_par[4], 0.0, 0.0)
    glafic.set_lens(2, 'pert', lens_par[0], lens_par[2], 0.0, 0.0, lens_par[5], lens_par[6], 0.0, 0.0)
    #glafic.set_point(1, lens_par[2], lens_par[7], lens_par[8])

    glafic.model_init(verb = 0)

    img_out = glafic.point_solve(lens_par[2], lens_par[7], lens_par[8], verb = 0)

    kapgam = []
    for i in range(len(img_out)):
        a = glafic.calcimage(lens_par[2], img_out[i][0], img_out[i][1])
        kapgam.append([a[3], a[4], a[5]])

    glafic.quit()

    return img_out, kapgam

