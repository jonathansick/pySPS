The Filter List
===============

FSPS has a large number of built-in filters (see `$SPS_HOME/data/FILTER_LIST` and `$SPS_HOME/data/allfilters.dat`). Magnitudes in the output data structures--*e.g.*, :class:`fsps.MagParser` or the HDF5 model table--can be addressed using the string-based keys in the table below.

Table of Filters
----------------

This filter table is current as of FSPS v2.3. If you add custom filters to `$SPS_HOME/data/allfilters.dat`, append a similar entry to `fsps.FILTER_LIST`.

=== ============== =======================================================================================
ID  Key            Comment
=== ============== =======================================================================================
1   V              Johnson V (from Bessell 1990 via M. Blanton). Defines V=0 for the Vega system
2   U              Johnson U (from Bessell 1990 via M. Blanton)
3   CFHT_B         CFHT B-band (from Blanton's kcorrect)
4   CFHT_R         CFHT R-band (from Blanton's kcorrect)
5   CFHT_I         CFHT I-band (from Blanton's kcorrect)
6   TMASS_J        2MASS J filter (total response w/atm)
7   TMASS_H        2MASS H filter (total response w/atm))
8   TMASS_Ks       2MASS Ks filter (total response w/atm)
9   SDSS_u         SDSS Camera u Response Function, airmass = 1.3 (June 2001)
10  SDSS_g         SDSS Camera g Response Function, airmass = 1.3 (June 2001)
11  SDSS_r         SDSS Camera r Response Function, airmass = 1.3 (June 2001)
12  SDSS_i         SDSS Camera i Response Function, airmass = 1.3 (June 2001)
13  SDSS_z         SDSS Camera z Response Function, airmass = 1.3 (June 2001)
14  WFC_ACS_F435W  WFC ACS F435W  (http://acs.pha.jhu.edu/instrument/photometry/)
15  WFC_ACS_F606W  WFC ACS F606W  (http://acs.pha.jhu.edu/instrument/photometry/)
16  WFC_ACS_F775W  WFC ACS F775W (http://acs.pha.jhu.edu/instrument/photometry/)
17  WFC_ACS_F814W  WFC ACS F814W  (http://acs.pha.jhu.edu/instrument/photometry/)
18  WFC_ACS_F850LP WFC ACS F850LP  (http://acs.pha.jhu.edu/instrument/photometry/)
19  IRAC_1         IRAC Channel 1
20  IRAC_2         IRAC Channel 2
21  IRAC_3         IRAC Channel 3
22  ISAAC_Js       ISAAC Js
23  ISAAC_Ks       ISAAC Ks
24  FORS_V         FORS V
25  FORS_R         FORS R
26  NICMOS_F110W   NICMOS F110W
27  NICMOS_F160W   NICMOS F160W
28  GALEX_NUV      GALEX NUV
29  GALEX_FUV      GALEX FUV
30  DES_g          DES g  (from Huan Lin, for DES camera)
31  DES_r          DES r  (from Huan Lin, for DES camera)
32  DES_i          DES i  (from Huan Lin, for DES camera)
33  DES_z          DES z  (from Huan Lin, for DES camera)
34  DES_Y          DES Y  (from Huan Lin, for DES camera)
35  WFCAM_Z        WFCAM Z  (from Hewett et al. 2006, via A. Smith)
36  WFCAM_Y        WFCAM Y  (from Hewett et al. 2006, via A. Smith)
37  WFCAM_J        WFCAM J  (from Hewett et al. 2006, via A. Smith)
38  WFCAM_H        WFCAM H  (from Hewett et al. 2006, via A. Smith)
39  WFCAM_K        WFCAM K  (from Hewett et al. 2006, via A. Smith)
40  BC03_B         Johnson B (from BC03.  This is the B2 filter from Buser)
41  Cousins_R      Cousins R (from Bessell 1990 via M. Blanton)
42  Cousins_I      Cousins I (from Bessell 1990 via M. Blanton)
43  B              Johnson B (from Bessell 1990 via M. Blanton)
44  WFPC2_F555W    WFPC2 F555W (http://acs.pha.jhu.edu/instrument/photometry/WFPC2/)
45  WFPC2_F814W    WFPC2 F814W (http://acs.pha.jhu.edu/instrument/photometry/WFPC2/)
46  Cousins_I_2    Cousins I (http://acs.pha.jhu.edu/instrument/photometry/GROUND/)
47  WFC3_F275W     WFC3 F275W (ftp://ftp.stsci.edu/cdbs/comp/wfc3/)
48  Steidel_Un     Steidel Un (via A. Shapley; see Steidel et al. 2003)
49  Steidel_G      Steidel G  (via A. Shapley; see Steidel et al. 2003)
50  Steidel_Rs     Steidel Rs (via A. Shapley; see Steidel et al. 2003)
51  Steidel_I      Steidel I  (via A. Shapley; see Steidel et al. 2003)
52  MegaCam_u      CFHT MegaCam u* (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html, Dec 2010)
53  MegaCam_g      CFHT MegaCam g' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)
54  MegaCam_r      CFHT MegaCam r' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)
55  MegaCam_i      CFHT MegaCam i' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)
56  MegaCam_z      CFHT MegaCam z' (http://cadcwww.dao.nrc.ca/megapipe/docs/filters.html)
57  WISE_W1        3.4um WISE W1 (http://www.astro.ucla.edu/~wright/WISE/passbands.html)
58  WISE_W2        4.6um WISE W2 (http://www.astro.ucla.edu/~wright/WISE/passbands.html)
59  WISE_W3        12um WISE W3 (http://www.astro.ucla.edu/~wright/WISE/passbands.html)
60  WISE_W4        22um WISE W4 22um (http://www.astro.ucla.edu/~wright/WISE/passbands.html)
61  WFC3_F125W     WFC3 F125W (ftp://ftp.stsci.edu/cdbs/comp/wfc3/)
62  WFC3_F160W     WFC3 F160W (ftp://ftp.stsci.edu/cdbs/comp/wfc3/)
63  UVOT_W2        UVOT W2 (from Erik Hoversten, 2011)
64  UVOT_M2        UVOT M2 (from Erik Hoversten, 2011)
65  UVOT_W1        UVOT W1 (from Erik Hoversten, 2011)
66  MIPS_24        Spitzer MIPS 24um
67  MIPS_70        Spitzer MIPS 70um
68  MIPS_160       Spitzer MIPS 160um
69  SCUBA_450WB    JCMT SCUBA 450WB (http://www.jach.hawaii.edu/JCMT/continuum/background/background.html)
70  SCUBA_850WB    JCMT SCUBA 850WB
71  PACS_70        Herschel PACS 70um
72  PACS_100       Herschel PACS 100um
73  PACS_160       Herschel PACS 160um
74  SPIRE_250      Herschel SPIRE 250um
75  SPIRE_350      Herschel SPIRE 350um
76  SPIRE_500      Herschel SPIRE 500um
77  IRAS_12        IRAS 12um
78  IRAS_25        IRAS 25um
79  IRAS_60        IRAS 60um
80  Bessell_L      Bessell & Brett (1988) L band
81  Bessell_LP     Bessell & Brett (1988) L' band
82  Bessell_M      Bessell & Brett (1988) M band
=== ============== =======================================================================================