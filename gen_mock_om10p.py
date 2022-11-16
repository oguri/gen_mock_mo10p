#!/usr/bin/env python
import numpy as np
import sys
import getopt
import scipy
from astropy.io import fits

from colossus.cosmology import cosmology

import source_qso
import source_sn
import source_tab
import solve_lenseq
import lens_vf

#
# generate mocks
#
def run_command_line(argv):
    try:
        opts, args = getopt.getopt(argv, 'h', ['help', 'area=', 'ilim=', 'zlmax=', 'source=', 'prefix=', 'solver='])
    except getopt.GetoptError as err:
        print('Error: wrong option')
        print(str(err))
        sys.exit(0)

    # default values
    area = 2000.0
    ilim = 23.3
    zlmax = 3.0
    source = 'qso'
    prefix = 'test_qso'
    solver = 'glafic'
    sepmin = 0.0
    sepmax = 100.0
    frlim = 0.1
    flag_mag = 3

    for o, a in opts:
        if o in ('-h', '--help'):
            print('Example: \n ./gen_mock_om10p.py --area=5000.0 --ilim=22.6 --zlmax=2.0 --source=sn --prefix=test_sn --solver=glafic')
            sys.exit()
        elif o in ('--area'):
            area = float(a)
        elif o in ('--ilim'):
            ilim = float(a)
        elif o in ('--zlmax'):
            zlmax = float(a)
        elif o in ('--source'):
            source = a
        elif o in ('--prefix'):
            prefix = a
        elif o in ('--solver'):
            solver = a

    print('# area  : %f' % area)
    print('# ilim  : %f' % ilim)
    print('# zlmax : %f' % zlmax)
    print('# source: %s' % source)
    print('# prefix: %s' % prefix)
    print('# solver: %s' % solver)
            
    # preparation
    cosmo = init_cosmo()
      
    source_qso.set_spline_kcor_qso()
    source_sn.set_spline_kcor_sn(1)
  
    if solver =='glafic':
        flag_solver = 0
    elif solver == 'lenstronomy':
        flag_solver = 1
    else:
        print('Error: solver should be either glafic or lenstronomy')
        sys.exit(0)

    if source == 'qso':
        flag_type_min = 0
        flag_type_max = 0
    elif source == 'sn':
        flag_type_min = 1
        flag_type_max = 5
    else:
        print('Error: source should be either qso or sn')
        sys.exit(0)

    imax = ilim + 3.5

    # run mock 
    do_mock_wfits(imax, area, ilim, sepmin, sepmax, flag_type_min, flag_type_max, flag_mag, frlim, zlmax, flag_solver, prefix, cosmo)
    

def do_mock_wfits(mmax, fov, mlim, sepmin, sepmax, flag_type_min, flag_type_max, flag_mag, frlim, zmax, flag_solver, prefix, cosmo):
    do_mock(mmax, fov, mlim, sepmin, sepmax, flag_type_min, flag_type_max, flag_mag, frlim, zmax, flag_solver, prefix, cosmo)
    conv_out_fits(prefix, cosmo)

    return
    
def do_mock(mmax, fov, mlim, sepmin, sepmax, flag_type_min, flag_type_max, flag_mag, frlim, zmax, flag_solver, prefix, cosmo):
    m_tab, z_tab, f_tab = source_tab.make_srctab(mmax, fov, flag_type_min, flag_type_max, cosmo)
    d = float(len(m_tab)) / (fov * 3600.0 * 3600.0)
    n_srctab = float(len(m_tab))

    ofile1 = open(prefix + '_result.dat', 'x')
    ofile2 = open(prefix + '_log.dat', 'x')
    
    dz = 0.001
    vmin = 10.0
    vmax = 450.01
    dlv = 0.001
    rt_range = 4.0

    zz = np.arange(dz, zmax + dz, dz)
    vv = 10 ** np.arange(np.log10(vmin), np.log10(vmax), dlv)
    dv = vv * (10 ** (0.5 * dlv) - 10 ** ((-0.5) * dlv))

    lid = 1
    for i in range(len(zz)):
        zz2 = np.array([zz[i]] * len(vv))
        nn = fov * lens_vf.dndzdv_lens(vv, zz2, cosmo) * dv * dz
        n = np.random.poisson(nn)

        cut_n = n[n > 0]
        cut_z = zz2[n > 0]
        cut_v = vv[n > 0]

        # one by one to save memory
        zl_all = []
        for j in range(len(cut_n)):
            zl_all.append([cut_z[j]] * cut_n[j])
            
        zl_tab = np.array([x for xx in zl_all for x in xx])
        zl_all = []

        vl_all = []
        for j in range(len(cut_n)):
            vl_all.append([cut_v[j]] * cut_n[j])
            
        vl_tab = np.array([x for xx in vl_all for x in xx])
        vl_all = []

        tl_all = []
        for j in range(len(cut_n)):
            tl_all.append([solve_lenseq.vtoein_func(cut_v[j], 1.0, 1.0)] * cut_n[j])
            
        tl_tab = np.array([x for xx in tl_all for x in xx])
        tl_all = []

        id_tab = np.arange(lid, lid + len(zl_tab), dtype = 'uint64')
        lid += len(zl_tab)
        
        nn2 = 4.0 * rt_range * rt_range * tl_tab * tl_tab * d
        n2 = np.random.poisson(nn2)

        cut_n2 = n2[n2 > 0]
        cut_zl = zl_tab[n2 > 0]
        cut_vl = vl_tab[n2 > 0]
        cut_tl = tl_tab[n2 > 0]
        cut_id = id_tab[n2 > 0]

        cut_e, cut_te = solve_lenseq.set_lenspar(len(cut_n2))
        
        for j in range(len(cut_n2)):
            for k in range(cut_n2[j]):
                kk = int(n_srctab * np.random.rand())
                zs = z_tab[kk]
                if zs > (zz[i] + 0.5 * dz):
                    ms = m_tab[kk]
                    fs = f_tab[kk]
                    tm = solve_lenseq.vtoein(cut_vl[j], zz[i], zs, cosmo)
                    if tm > (sepmin * 0.2):
                        x = (np.random.rand() - 0.5) * cut_tl[j] * rt_range * 2.0
                        y = (np.random.rand() - 0.5) * cut_tl[j] * rt_range * 2.0
                        if np.abs(x) < (tm * (rt_range + 1.0)) and np.abs(y) < (tm * (rt_range + 1.0)):
                            g, tg = solve_lenseq.set_shear(zs)
                            lens_par = [zz[i], cut_vl[j], zs, cut_e[j], cut_te[j], g, tg, x, y]
                            out_img, nim, sep, mag, fr, kapgam = solve_lenseq.calc_image(lens_par, tm, rt_range, flag_mag, flag_solver, cosmo)
                            if nim > 1:
                                mobs = ms - 2.5 * np.log10(mag)
                                if mobs <= mlim and sep >= sepmin and sep <= sepmax and fr >= frlim:
                                    dump_result(lens_par, out_img, sep, ms, mobs, fs, kapgam, cut_id[j], ofile1, ofile2)

    ofile1.close()
    ofile2.close()
    
    return

def dump_result(lens_par, out_img, sep, mori, mobs, flag_type, kapgam, lid, ofile1, ofile2):
    nim = len(out_img)
    
    out = '%d %e %e %e %e %e %e %13e %13e %13e %13e %13e %13e %d %d\n' % (nim, lens_par[0], lens_par[1], lens_par[2], mori, mobs, sep, lens_par[3], lens_par[4], lens_par[5], lens_par[6], lens_par[7], lens_par[8], flag_type, lid)
    ofile1.write(out)

    out = '%d %13e %13e %d %d\n' % (nim, lens_par[7], lens_par[8], flag_type, lid)
    ofile2.write(out)
    for i in range(nim):
        out = '%13e %13e %13e %13e  %13e %13e %13e\n' % (out_img[i][0], out_img[i][1], out_img[i][2], out_img[i][3], kapgam[i][0], kapgam[i][1], kapgam[i][2])
        ofile2.write(out)

    ofile1.flush()
    ofile2.flush()

    return

def conv_out_fits(prefix, cosmo):
    result = np.loadtxt(prefix + '_result.dat')

    nimg     = result[:,  0].astype(np.int16)
    zlens    = result[:,  1].astype(np.float64)
    veldisp  = result[:,  2].astype(np.float64)
    zsrc     = result[:,  3].astype(np.float64)
    magi_in  = result[:,  4].astype(np.float64)
    magi     = result[:,  5].astype(np.float64)
    imsep    = result[:,  6].astype(np.float64)
    ellip    = result[:,  7].astype(np.float64)
    phie     = result[:,  8].astype(np.float64)
    gamma    = result[:,  9].astype(np.float64)
    phig     = result[:, 10].astype(np.float64)
    xsrc     = result[:, 11].astype(np.float64)
    ysrc     = result[:, 12].astype(np.float64)
    flagtype = result[:, 13].astype(np.int16)
    lensid   = result[:, 14].astype(np.int32)
    
    data_log = open(prefix + '_log.dat', 'r')

    nimg_max  = 4
    tmp_ximg  = []
    tmp_yimg  = []
    tmp_mag   = []
    tmp_delay = []
    tmp_kappa = []
    tmp_gam1  = []
    tmp_gam2  = []
    for i in range(len(nimg)):
        source_data = data_log.readline().split()
        if np.int16(source_data[0]) != nimg[i]:
            sys.exit('Error: data fortmat wrong')

        tmp_tmp_ximg  = []
        tmp_tmp_yimg  = []
        tmp_tmp_mag   = []
        tmp_tmp_delay = []
        tmp_tmp_kappa = []
        tmp_tmp_gam1  = []
        tmp_tmp_gam2  = []
        for j in range(nimg_max):
            if j < nimg[i]:
                line_img = data_log.readline().split()
                tmp_tmp_ximg.append(float(line_img[0]))
                tmp_tmp_yimg.append(float(line_img[1]))
                tmp_tmp_mag.append(float(line_img[2]))
                tmp_tmp_delay.append(float(line_img[3]))
                tmp_tmp_kappa.append(float(line_img[4]))
                tmp_tmp_gam1.append(float(line_img[5]))
                tmp_tmp_gam2.append(float(line_img[6]))
            else:
                tmp_tmp_ximg.append(0.0)
                tmp_tmp_yimg.append(0.0)
                tmp_tmp_mag.append(0.0)
                tmp_tmp_delay.append(0.0)
                tmp_tmp_kappa.append(0.0)
                tmp_tmp_gam1.append(0.0)
                tmp_tmp_gam2.append(0.0)

        tmp_ximg.append(tmp_tmp_ximg)
        tmp_yimg.append(tmp_tmp_yimg)
        tmp_mag.append(tmp_tmp_mag)
        tmp_delay.append(tmp_tmp_delay)
        tmp_kappa.append(tmp_tmp_kappa)
        tmp_gam1.append(tmp_tmp_gam1)
        tmp_gam2.append(tmp_tmp_gam2)

    ximg  = np.matrix(tmp_ximg)
    yimg  = np.matrix(tmp_yimg)
    mag   = np.matrix(tmp_mag)
    delay = np.matrix(tmp_delay)
    kappa =  np.matrix(tmp_kappa)
    gam1  =  np.matrix(tmp_gam1)
    gam2  =  np.matrix(tmp_gam2)
    
    name = ['LENSID', 'FLAGTYPE', 'NIMG', 'ZLENS', 'VELDISP', 'ELLIP', 'PHIE', 'GAMMA', 'PHIG', 'ZSRC', 'XSRC', 'YSRC', 'MAGI_IN', 'MAGI', 'IMSEP', 'XIMG', 'YIMG', 'MAG', 'DELAY', 'KAPPA', 'GAM1', 'GAM2']
    format = ['J', 'I', 'I', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', '4D', '4D', '4D', '4D', '4D', '4D', '4D']
    arrays = [lensid, flagtype, nimg, zlens, veldisp, ellip, phie, gamma, phig, zsrc, xsrc, ysrc, magi_in, magi, imsep, ximg, yimg, mag, delay, kappa, gam1, gam2]
    columns = []
    for i in range(len(name)):
        columns.append(fits.Column(name = name[i], format = format[i], array = arrays[i]))
    coldefs = fits.ColDefs(columns)
    tbhdu = fits.BinTableHDU.from_columns(coldefs)
    tbhdu.writeto(prefix + '.fits')

#
# set cosmological parameters
#
def init_cosmo():
    #my_cosmo = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.81, 'ns': 0.96}
    my_cosmo = {'flat': True, 'H0': 67.0, 'Om0': 0.315, 'Ob0': 0.05, 'sigma8': 0.81, 'ns': 0.96}
    cosmo = cosmology.setCosmology('my_cosmo', my_cosmo)

    return cosmo

#
# main function
#
if __name__ == '__main__':
    run_command_line(sys.argv[1:])

    #cosmo = init_cosmo()

    #source_qso.set_spline_kcor_qso()
    #source_sn.set_spline_kcor_sn(1)

    # 0: glafic, 1: lenstronomy
    #flag_solver = 0

    # full LSST mock (1 realization)
    #do_mock(27.0, 20000.0, 23.3, 0.0, 100.0, 0, 0, 3, 0.1, 3.0, flag_solver, 'qso_mock', cosmo)
    #do_mock(27.0, 50000.0, 22.6, 0.0, 100.0, 1, 5, 3, 0.1, 2.0, flag_solver, 'sne_mock', cosmo)

    # subsamples for checking/testing
    #do_mock_wfits(27.0, 2000.0, 23.3, 0.0, 100.0, 0, 0, 3, 0.1, 3.0, 0, 'test_qso_mock_glafic', cosmo)
    #./gen_mock_om10p.py --area=2000.0 --ilim=23.3 --zlmax=3.0 --source=qso --prefix=test2_qso_mock_glafic --solver=glafic
    #do_mock_wfits(27.0, 5000.0, 22.6, 0.0, 100.0, 1, 5, 3, 0.1, 2.0, 0, 'test_sne_mock_glafic', cosmo)
    #./gen_mock_om10p.py --area=5000.0 --ilim=22.6 --zlmax=2.0 --source=sn --prefix=test2_sne_mock_glafic --solver=glafic
    #do_mock_wfits(27.0, 2000.0, 23.3, 0.0, 100.0, 0, 0, 3, 0.1, 3.0, 1, 'test_qso_mock_lenstronomy', cosmo)
    #./gen_mock_om10p.py --area=2000.0 --ilim=23.3 --zlmax=3.0 --source=qso --prefix=test2_qso_mock_lenstronomy --solver=lenstronomy
    #do_mock_wfits(27.0, 5000.0, 22.6, 0.0, 100.0, 1, 5, 3, 0.1, 2.0, 1, 'test_sne_mock_lenstronomy', cosmo)
    #./gen_mock_om10p.py --area=5000.0 --ilim=22.6 --zlmax=2.0 --source=sn --prefix=test2_sne_mock_lenstronomy --solver=lenstronomy

    
