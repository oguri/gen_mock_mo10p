#!/usr/bin/env python
import numpy as np 

import solve_lenseq

from astropy.cosmology import FlatLambdaCDM
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

def solve_lenseq_lenstronomy(lens_par, tm, rt_range, cosmo):
    cosmo_astropy = FlatLambdaCDM(H0 = cosmo.H0, Om0 = cosmo.Om0, Ob0 = 0.)

    ein = tm * (solve_lenseq.bb_fac(lens_par[3]) ** 2)
    qq = (1.0 - (1.0 - lens_par[3])) / (1.0 + (1.0 - lens_par[3]))
    te = 2.0 * np.pi * (lens_par[4] + 90.0) / 180.0
    e1 = qq * np.cos(te)
    e2 = qq * np.sin(te)
    tg = 2.0 * np.pi * lens_par[6] / 180.0
    g1 = lens_par[5] * np.cos(tg)
    g2 = lens_par[5] * np.sin(tg)
    
    lens_model_list = ['SIE', 'SHEAR']

    lensModel = LensModel(lens_model_list = lens_model_list, z_lens = lens_par[0], z_source = lens_par[2], cosmo = cosmo_astropy)

    kwargs_sie = {'theta_E': ein, 'e1': e1, 'e2': e2, 'center_x': 0.0, 'center_y': 0.0}
    kwargs_shear = {'gamma1': g1, 'gamma2': g2, 'ra_0': 0.0, 'dec_0': 0.0}

    kwargs_lens = [kwargs_sie, kwargs_shear]
    solver = LensEquationSolver(lensModel)

    x, y = solver.image_position_from_source(kwargs_lens = kwargs_lens, sourcePos_x = lens_par[7], sourcePos_y = lens_par[8], min_distance = tm / 80.0, search_window = tm * (rt_range + 3.0))

    if(len(x) > 0):
        mag = lensModel.magnification(x, y, kwargs_lens)
        kappa = lensModel.kappa(x, y, kwargs_lens)
        gamma1, gamma2 = lensModel.gamma(x, y, kwargs_lens)

        arrival_time = lensModel.arrival_time(x, y, kwargs_lens)
        relative_delay = arrival_time - arrival_time.min()

        img_out = (np.vstack([x, y, mag, relative_delay]).T).tolist()
        kapgam = (np.vstack([kappa, gamma1, gamma2]).T).tolist()

    else:
        img_out = []
        kapgam = []
        
    return img_out, kapgam

