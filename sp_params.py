#!/usr/bin/env python
# encoding: utf-8
"""
General parameters for FSPS, including classes for ParameterSets, filter
lists, and metallicity look-up tables.

History
-------
2011-12-30 - Created by Jonathan Sick

"""
import bson

class ParameterSet(object):
    """An input parameter set for a FSPS model run."""
    def __init__(self, name, **kwargs):
        if name is None:
            self.name = str(bson.objectid.ObjectId())
        else:
            self.name = str(name) # name of this model
        # Default values
        self.p = {"compute_vega_mags":0, "dust_type":0, "imf_type":0,
                "isoc_type":'pdva', "redshift_colors":0, "time_res_incr":2,
                "zred":0., "zmet":1, "sfh":0, "tau":1., "const":0.,
                "sf_start":0.,"tage":0., "fburst":0., "tburst":0., "imf1":1.3,
                "imf2":2.3, "imf3":2.3, "vdmc":0.08, "mdave":0.5,
                "dust_tesc":7., "dust1":0., "dust2":0., "dust_clumps":-99.,
                "frac_nodust":0., "dust_index":-0.7, "mwr":3.1,
                "uvb":1., "wgp1":1, "wgp2":1, "wgp3":0, "dell":0.,
                "delt":0., "sbss":0., "fbhb":0, "pagb":1.}
        self.knownParameters = self.p.keys()
        # Update values with user's arguments
        for k, v in kwargs.iteritems():
            if k in self.knownParameters:
                self.p[k] = v
    
    def __getitem__(self, key):
        """Return the parameter value given the key."""
        return self.p[key]
    
    def command(self):
        """Write the string for this parameter set.
        
        .. note:: Deprecated with the f2py wrapper.
        """
        # These are pset variables, (aside from sfh)
        dt = [("zred","%.2f"),("zmet","%02i"),("tau","%.10f"),("const","%.4f"),
                ("sf_start","%.2f"),("tage","%.4f"),("fburst","%.4f"),
                ("tburst","%.4f"),("imf1","%.2f"),("imf2","%.2f"),
                ("imf3","%.2f"),("vdmc","%.2f"),("mdave","%.1f"),
                ("dust_tesc","%.2f"),("dust1","%.6f"),("dust2","%.6f"),
                ("dust_clumps","%.1f"),("frac_nodust","%.2f"),
                ("dust_index","%.2f"),("mwr","%.2f"),("uvb","%.2f"),
                ("wgp1","%i"),("wgp2","%i"),("wgp3","%i"),("dell","%.2f"),
                ("delt","%.2f"),("sbss","%.2f"),("fbhb","%.2f"),
                ("pagb","%.2f")]
        cmd = str(self.name) + " " + " ".join([s % self.p[k] for (k,s) in dt])
        return cmd
    
    def get_doc(self):
        """Returns the document dictionary to insert in MongoDB."""
        return self.p

FILTER_LIST = [(1,'V','Johnson V (from Bessell 1990 via M. Blanton)  - this defines V=0 for the Vega system'),
        (2,"U","Johnson U (from Bessell 1990 via M. Blanton)"),
        (3,"CFHT_B","CFHT B-band (from Blanton's kcorrect)"),
        (4,"CFHT_R","CFHT R-band (from Blanton's kcorrect)"),
        (5,"CFHT_I","CFHT I-band (from Blanton's kcorrect)"),
        (6,"TMASS_J","2MASS J filter (total response w/atm)"),
        (7,"TMASS_H","2MASS H filter (total response w/atm))"),
        (8,"TMASS_Ks","2MASS Ks filter (total response w/atm)"),
        (9,"SDSS_u","SDSS Camera u Response Function, airmass = 1.3 (June 2001)"),
        (10,"SDSS_g","SDSS Camera g Response Function, airmass = 1.3 (June 2001)"),
        (11,"SDSS_r","SDSS Camera r Response Function, airmass = 1.3 (June 2001)"),
        (12,"SDSS_i","SDSS Camera i Response Function, airmass = 1.3 (June 2001)"),
        (13,"SDSS_z","SDSS Camera z Response Function, airmass = 1.3 (June 2001)"),
        (14,"WFC_ACS_F435W","WFC ACS F435W  (http://acs.pha.jhu.edu/instrument/photometry/)"),
        (15,"WFC_ACS_F606W","WFC ACS F606W  (http://acs.pha.jhu.edu/instrument/photometry/)"),
        (16,"WFC_ACS_F775W","WFC ACS F775W (http://acs.pha.jhu.edu/instrument/photometry/)"),
        (17,"WFC_ACS_F814W","WFC ACS F814W  (http://acs.pha.jhu.edu/instrument/photometry/)"),
        (18,"WFC_ACS_F850LP","WFC ACS F850LP  (http://acs.pha.jhu.edu/instrument/photometry/)"),
        (19,"IRAC_1","IRAC Channel 1"),
        (20,"IRAC_2","IRAC Channel 2"),
        (21,"IRAC_3","IRAC Channel 3"),
        (22,"ISAAC_Js","ISAAC Js"),
        (23,"ISAAC_Ks","ISAAC Ks"),
        (24,"FORS_V","FORS V"),
        (25,"FORS_R","FORS R"),
        (26,"NICMOS_F110W","NICMOS F110W"),
        (27,"NICMOS_F160W","NICMOS F160W"),
        (28,"GALEX_NUV","GALEX NUV"),
        (29,"GALEX_FUV","GALEX FUV"),
        (30,"DES_g","DES g  (from Huan Lin, for DES camera)"),
        (31,"DES_r","DES r  (from Huan Lin, for DES camera)"),
        (32,"DES_i","DES i  (from Huan Lin, for DES camera)"),
        (33,"DES_z","DES z  (from Huan Lin, for DES camera)"),
        (34,"DES_Y","DES Y  (from Huan Lin, for DES camera)"),
        (35,"WFCAM_Z","WFCAM Z  (from Hewett et al. 2006, via A. Smith)"),
        (36,"WFCAM_Y","WFCAM Y  (from Hewett et al. 2006, via A. Smith)"),
        (37,"WFCAM_J","WFCAM J  (from Hewett et al. 2006, via A. Smith)"),
        (38,"WFCAM_H","WFCAM H  (from Hewett et al. 2006, via A. Smith)"),
        (39,"WFCAM_K","WFCAM K  (from Hewett et al. 2006, via A. Smith)"),
        (40,"BC03_B","Johnson B (from BC03.  This is the B2 filter from Buser)"),
        (41,"Cousins_R","Cousins R (from Bessell 1990 via M. Blanton)"),
        (42,"Cousins_I","Cousins I (from Bessell 1990 via M. Blanton)"),
        (43,"B","Johnson B (from Bessell 1990 via M. Blanton)"),
        (44,"WFPC2_F555W","WFPC2 F555W (http://acs.pha.jhu.edu/instrument/photometry/WFPC2/)"),
        (45,"WFPC2_F814W","WFPC2 F814W (http://acs.pha.jhu.edu/instrument/photometry/WFPC2/)"),
        (46,"Cousins_I_2","Cousins I (http://acs.pha.jhu.edu/instrument/photometry/GROUND/)"),
        (47,"WFC3_F275W","WFC3 F275W (ftp://ftp.stsci.edu/cdbs/comp/wfc3/)"),
        (48,"Steidel_Un","Steidel Un (via A. Shapley; see Steidel et al. 2003)"),
        (49,"Steidel_G","Steidel G  (via A. Shapley; see Steidel et al. 2003)"),
        (50,"Steidel_Rs","Steidel Rs (via A. Shapley; see Steidel et al. 2003)"),
        (51,"Steidel_I","Steidel I  (via A. Shapley; see Steidel et al. 2003)"),
        (52,"MegaCam_u","CFHT MegaCam u* (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html, Dec 2010)"),
        (53,"MegaCam_g","CFHT MegaCam g' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)"),
        (54,"MegaCam_r","CFHT MegaCam r' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)"),
        (55,"MegaCam_i","CFHT MegaCam i' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)"),
        (56,"MegaCam_z","CFHT MegaCam z' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)"),
        (57,"WISE_W1","3.4um WISE W1 (http://www.astro.ucla.edu/~wright/WISE/passbands.html)"),
        (58,"WISE_W2","4.6um WISE W2 (http://www.astro.ucla.edu/~wright/WISE/passbands.html)"),
        (59,"WISE_W3","12um WISE W3 (http://www.astro.ucla.edu/~wright/WISE/passbands.html)"),
        (60,"WISE_W4","22um WISE W4 22um (http://www.astro.ucla.edu/~wright/WISE/passbands.html)"),
        (61,"WFC3_F125W","WFC3 F125W (ftp://ftp.stsci.edu/cdbs/comp/wfc3/)"),
        (62,"WFC3_F160W","WFC3 F160W (ftp://ftp.stsci.edu/cdbs/comp/wfc3/)"),
        (63,"UVOT_W2","UVOT W2 (from Erik Hoversten, 2011)"),
        (64,"UVOT_M2","UVOT M2 (from Erik Hoversten, 2011)"),
        (65,"UVOT_W1","UVOT W1 (from Erik Hoversten, 2011)"),
        (66,"MIPS_24","Spitzer MIPS 24um"),
        (67,"MIPS_70","Spitzer MIPS 70um"),
        (68,"MIPS_160","Spitzer MIPS 160um"),
        (69,"SCUBA_450WB","JCMT SCUBA 450WB (www.jach.hawaii.edu/JCMT/continuum/background/background.html)"),
        (70,"SCUBA_850WB","JCMT SCUBA 850WB"),
        (71,"PACS_70","Herschel PACS 70um"),
        (72,"PACS_100","Herschel PACS 100um"),
        (73,"PACS_160","Herschel PACS 160um"),
        (74,"SPIRE_250","Herschel SPIRE 250um"),
        (75,"SPIRE_350","Herschel SPIRE 350um"),
        (76,"SPIRE_500","Herschel SPIRE 500um"),
        (77,"IRAS_12","IRAS 12um"),
        (78,"IRAS_25","IRAS 25um"),
        (79,"IRAS_60","IRAS 60um"),
        (80,"Bessell_L","Bessell & Brett (1988) L band"),
        (81,"Bessell_LP","Bessell & Brett (1988) L' band"),
        (82,"Bessell_M","Bessell & Brett (1988) M band")]

def get_metallicity_LUT(isocType, specType):
    """docstring for as_metallicity"""
    if isocType=="pdva" and specType=="basel":
        return (0.0002, 0.0003, 0.0004, 0.0005,0.0006,0.0008,0.0010,
            0.0012,0.0016,0.0020,0.0025,0.0031,0.0039,0.0049,0.0061,
            0.0077,0.0096,0.0120,0.0150,0.0190,0.0240,0.0300)
