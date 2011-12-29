#!/usr/bin/env python
# encoding: utf-8
"""Burn-in tests for the fsps f2py wrapper"""

import numpy as np
import fsps

def main():
    use_vega_mags, apply_redshift = 0, 0

    imf, imf1, imf2, imf3, vdmc, mdave, dell, delt, sbss, fbhb, pagb = \
        0,1.3,2.3,2.3,0.08,0.5,0.,0.,0.,0.,1.
            
    model_name = "test_fsps"
    dust_type = 0
    zmet = 0
    sfh = 1
    tau = 0.5
    const = 0.1
    fburst = 0.
    tburst = 0.
    dust_tesc = 7.
    dust1 = 0.
    dust2 = 0.
    dust_clumps = -99.
    frac_no_dust = 0.1
    dust_index = -0.7,
    mwr = 3.1
    wgp1 = 1
    wgp2 = 1
    wgp3 = 1
    duste_gamma = 0.01
    duste_umin = 1.0
    duste_qpah = 3.5
    tage = 0.
            
    fsps.fsps.setup(use_vega_mags, apply_redshift)
    nBands = fsps.fsps.get_n_bands()
    nAges = fsps.fsps.get_n_ages()
    nMasses = fsps.fsps.get_n_masses()
    print "There are", nBands, "bands"
    print "There are", nAges, "ages"
    print "There are", nMasses, "masses"
    #fsps.fsps.setup_all_ssp(imf, imf1, imf2, imf3, vdmc, mdave, dell, delt,
    #        sbss, fbhb, pagb)
    #fsps.fsps.comp_sp(dust_type, zmet, sfh, tau, const, 
    #        fburst, tburst, dust_tesc, dust1, dust2, dust_clumps, 
    #        frac_no_dust, dust_index, mwr, wgp1, wgp2, wgp3, 
    #        duste_gamma, duste_umin, duste_qpah, tage)
    #mags = fsps.fsps.get_mags_at_age(120, nBands) # index 120 in age array
    #print "Mags:", mags
    #print fsps.fsps.get_stats_at_age(120)
    #allMags = fsps.fsps.get_mags(nBands, nAges)
    #print "All mags:", allMags
    #print "All mags shape", allMags.shape
    #allAges, allMass, allLbol, allSFR, allMDust = fsps.fsps.get_stats(nAges)
    #print allAges


    zz = 8
    tt = 10
    nMass = fsps.fsps.get_n_masses_isochrone(zz, tt) # n masses for this isoc
    print "nMasses for isochrone", nMass
    time, Z, massInit, logL, logT, logg, ffco, phase, wght, isocMags = \
            fsps.fsps.get_isochrone(zz, tt, nMass, nBands)
    print time, Z
    print isocMags.shape

    print "All tests complete!"

if __name__ == '__main__':
    main()


