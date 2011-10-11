"""Generates input files for fspsq."""
import socket
import os
import subprocess
import multiprocessing
import datetime
import cPickle as pickle

import pytz
import pymongo
import numpy as np
#import numpy.lib.recfunctions as recfunctions
import tables # HDF5
import matplotlib.mlab as mlab

import bson
from bson.binary import Binary
from pymongo.son_manipulator import SONManipulator

class FSPSLibrary(object):
    """For building/accessing a library of FSPS stellar population models.
    
    FSPSLibrary is itself an interface for the PyMongo database that stores
    model parameters and realizations from the FSPS code.
    """
    def __init__(self, libname, dbname="fsps"):
        super(FSPSLibrary, self).__init__()
        self.libname = libname
        self.dbname = dbname
        
        connection = pymongo.Connection()
        self.host = connection.host
        self.port = connection.port
        self.db = connection[self.dbname]
        self.db.add_son_manipulator(NumpySONManipulator())
        self.collection = self.db[self.libname]
    
    def register_pset(self, pset):
        """Adds a ParameterSet to the collection of model definitions."""
        self.remove_existing_model(pset.name)
        doc = {"_id": pset.name, "pset": pset.get_doc(),
            "compute_started": False, "compute_complete": False}
        self.collection.insert(doc)
    
    def compute_models(self, nThreads=1, maxN=10):
        """Fire up FSPS to generate SP models.
        
        Parameters
        ----------
        
        nThreads : int, optional
            Number of processes to be spawned, for multiprocessing. If set
            to `1`, multiprocessing is not used; useful for debugging.
        maxN : int, optional
            Maximum number of jobs that can be given to a single
            `fspsq` process.
        """
        fspsqPath = os.path.join(os.getcwd(), "fspsq") # echo for debug
        commandPaths = ["processor-%i" % i for i in xrange(1, nThreads+1)]
        queue_runner = QueueRunner(self.libname, self.dbname, self.host,
                self.port, maxN, fspsqPath)
        # print args
        if nThreads > 1:
            pool = multiprocessing.Pool(processes=nThreads)
            pool.map(queue_runner, commandPaths)
        else:
            map(queue_runner, commandPaths)
    
    def reset(self):
        """Drop (delete) the library.
        
        The collection member is reinstantiated. This allows the collection
        to be freshly re-populated with models.
        """
        self.db.drop_collection(self.libname)
        self.collection = self.db[self.libname]
    
    def remove_existing_model(self, modelName):
        """Delete a model, by name, if it exists."""
        if self.collection.find_one({"_id": modelName}) is not None:
            self.collection.remove({"_id": modelName})
    
    def count_models(self):
        return self.collection.find({}).count()

    def create_table(self, outputPath, query={}, tage=None,
            isocType="pdva", specType="basel", clobber=True):
        """Create an HDF5 table that combines outputs from models
        in the library.
        """
        query.update({"compute_complete":True,
            "np_data": {"$exists": 1}})
        docs = self.collection.find(query) # , limit=2
        print "working on %i docs to read" % docs.count()
        lut = get_metallicity_LUT(isocType, specType)
        # TODO need to generalize definition of columns. A user ought to
        # be able to use any pset columns, any set of mags, and the spectra
        #magNames = ['TMASS_J','TMASS_H','TMASS_Ks','MegaCam_u','MegaCam_g',
        #        'MegaCam_r','MegaCam_i','MegaCam_z','GALEX_NUV','GALEX_FUV']
        magCols = [(s,np.float,) for (i,s,c) in FILTER_LIST]
        #magCols = [(s,np.float) for s in magNames]
        psetCols = [('dust_type',np.int),('imf_type',np.int),('sfh',np.int),
                ('tau',np.float),('const',np.float),('sf_start',np.float),
                ('fburst',np.float),('tburst',np.float),('dust_tesc',np.float),
                ('dust1',np.float),('dust2',np.float),('frac_nodust',np.float)]
        sfhCols = [('age',np.float),('mass',np.float),('lbol',np.float),
                ('sfr',np.float)]
        miscCols = [('Z',np.float)] # metallicity, taken from zmet look-up-table
        specCols = [('spec',np.float,SpecParser.nlambda(specType))]
        allCols = psetCols+sfhCols+miscCols+magCols+specCols
        tableDtype = np.dtype(allCols)

        if os.path.exists(outputPath) and clobber:
            os.remove(outputPath)
        title = os.path.splitext(os.path.basename(outputPath))[0]
        h5file = tables.openFile(outputPath, mode="w", title=title)
        table = h5file.createTable("/", 'models', tableDtype,
                "Model Output Table")
        print h5file
        for doc in docs:
            print "reading", doc['_id']
            npData = doc['np_data']
            nRows = len(npData)
            
            # Appent pset cols and misc cols
            extraNames = []
            extraArrays = []

            zmet = doc['pset']['zmet']
            Z = lut[zmet-1]
            Z = np.ones(nRows, dtype=np.float) * Z
            extraNames.append('Z')
            extraArrays.append(Z)
            for cName, cType in psetCols:
                p = doc['pset'][cName]
                pArray = np.ones(nRows, dtype=cType) * p
                extraNames.append(cName)
                extraArrays.append(pArray)
            npDataAll = mlab.rec_append_fields(npData, extraNames, extraArrays)

            # Trim the recarray to just the desired fields
            #npDataTrim = mlab.rec_keep_fields(npDataAll,
            #    ['Z','tau','age','mass','lbol','sfr','TMASS_J','TMASS_H',
            #    'TMASS_Ks','MegaCam_u','MegaCam_g','MegaCam_r','MegaCam_i',
            #    'MegaCam_z','GALEX_NUV','GALEX_FUV'])
            # select row closest to the target age
            if tage is not None:
                ageGyr = 10.**npDataAll['age'] / 10.**9
                i = np.argmin((ageGyr - tage)**2)
                row = np.atleast_1d(np.array(npDataAll[i], copy=True))
                table.append(row)
            else:
                #table.append(npDataAll) # should work but corrupts data
                row = table.row
                for i in xrange(nRows):
                    print "row", i
                    for x in allCols:
                        name = x[0]
                        print name, npDataAll[i][name]
                        row[name] = npDataAll[i][name]
                    row.append()
        table.flush()
        h5file.flush()
        h5file.close()

    @property
    def _get_table_def(self):
        ccDtype = np.dtype([('c1',np.int),('c2',np.int),('xi',np.int),
            ('yi',np.int),('ml',np.float)])
        return ccDtype

class QueueRunner(object):
    """Executes a queue of FSPS models.
    
    This is typically called via :meth:`FSPSLibrary.compute_models()`.
    """
    def __init__(self, libname, dbname, dbhost, dbport, maxN, fspsqPath,
            tage=None):
        #super(QueueRunner, self).__init__()
        self.libname = libname
        self.dbname = dbname
        self.dbhost = dbhost
        self.dbport = dbport
        self.maxN = maxN
        self.fspsqPath = fspsqPath

    def __call__(self, commandPath):
        """Executed in the pool mapping; looks for and computes models."""
        # print "hello"
        thisHost = socket.gethostname() # hostname of current process
        
        # Connect to the library in MongoDB
        connection = pymongo.Connection(host=self.dbhost, port=self.dbport)
        db = connection[self.dbname]
        db.add_son_manipulator(NumpySONManipulator())
        self.collection = db[self.libname]
        
        commonVarSets = self._make_common_var_sets()
        # print "commonVarSets:", commonVarSets
        for varSet in commonVarSets:
            while True:
                psets = []
                modelNames = []
                now = datetime.datetime.utcnow()
                now = now.replace(tzinfo=pytz.utc)
                while len(psets) <= self.maxN:
                    q = {"compute_complete": False, "compute_started": False}
                    q.update(varSet)
                    # print "q", q
                    doc = self.collection.find_and_modify(query=q,
                        update={"$set": {"compute_started": True,
                                        "queue_date": now,
                                        "compute_host": thisHost}},)
                    # print "doc", doc
                    if doc is None: break # no available models
                    modelName = str(doc['_id'])
                    pset = ParameterSet(modelName, **doc['pset'])
                    psets.append(pset)
                    modelNames.append(pset.name)
                if len(psets) == 0: break # empty job queue
                # Startup a computation: write command file and start fspsq
                cmdTxt = "\n".join([p.command() for p in psets])+"\n"
                if os.path.exists(commandPath): os.remove(commandPath)
                f = open(commandPath, 'w')
                f.write(cmdTxt)
                f.close()
                nModels = len(psets)
                shellCmd = self._make_shell_command(self.fspsqPath,
                        commandPath, nModels, varSet)
                print "cmd::", shellCmd
                subprocess.call(shellCmd, shell=True)
                # Get output data from FSPS
                self._gather_fsps_outputs(modelNames)

        # Delete command file when done
        os.remove(commandPath)

    def _make_common_var_sets(self):
        """Make a list of common variable setups.
        
        .. note:: This merely lists all *possible* combinations of common variable
        configurations. There is not guarantee that there are models needing
        computation for each configuration set.
        
        Parameters
        ----------
        
        c : pymongo.Collection instance
        
        Returns
        -------
        
        List of dictionaries. Each dictionary has keys representing common
        variables: sfh, zmet, dust_type, imf_type, compute_vega_mags, redshift_colors
        """
        params = ['sfh', 'zmet', 'dust_type', 'imf_type',
            'compute_vega_mags', 'redshift_colors']
        possibleValues = {}
        for param in params:
            possibleValues[param] = self.collection.distinct("pset."+param)
        groups = []
        for isfh in possibleValues['sfh']:
            for izmet in possibleValues['zmet']:
                for idust_type in possibleValues['dust_type']:
                    for iimf_type in possibleValues['imf_type']:
                        for ivega in possibleValues['compute_vega_mags']:
                            for iredshift in possibleValues['redshift_colors']:
                                groups.append({"pset.sfh":isfh,"pset.zmet":izmet,
                                    "pset.dust_type":idust_type,"pset.imf_type":iimf_type,
                                    "pset.compute_vega_mags":ivega,
                                    "pset.redshift_colors":iredshift})
        return groups

    def _make_shell_command(self, fspsqPath, commandFilePath, nModels, varSet):
        """Crafts the command for running `fspsq`, returning a string."""
        cmd = "%s %s %i %i %02i %i %i %i %i" % (fspsqPath, commandFilePath, nModels,
            varSet['pset.sfh'], varSet['pset.zmet'], varSet['pset.dust_type'],
            varSet['pset.imf_type'], varSet['pset.compute_vega_mags'],
            varSet['pset.redshift_colors'])
        return cmd

    def _gather_fsps_outputs(self, modelNames, getMags=True, getSpec=True):
        """Adds the .mag and .spec outputs for each model to MongoDB.
        
        .. note:: Need to propagate the spec type through here...
        """
        fspsDir = os.environ['SPS_HOME'] # environment variable for FSPS
        outputDir = os.path.join(fspsDir, "OUTPUTS")
        # print modelNames
        for modelName in modelNames:
            magPath = os.path.join(outputDir, modelName+".out.mags")
            specPath = os.path.join(outputDir, modelName+".out.spec")
            print "Ingesting", modelName
            #try:
            magParser = MagParser(magPath)
            magData = magParser.data
            specParser = SpecParser(specPath)
            specData = specParser.data
            nLambda = specParser.nlambda('basel')
            if getMags and getSpec:
                allData = self._splice_mag_spec_arrays(magData, specData, nLambda)
            #except:
            #    # Some exception in parsing the text files; skip the output
            #    self.collection.update({"_id": modelName},
            #            {"$set": {"compute_complete": True,
            #            "ingest_fail": True}})
            #    print "\t", modelName, "ingest fail"
            #    continue
            self.collection.update({"_id": modelName}, {"$set": {"compute_complete": True}})
            binData = Binary(pickle.dumps(allData,-1))
            self.collection.update({"_id": modelName},
                {"$set": {"np_data": {'_type': 'np.ndarray', 'data':binData}}})
            # Remove data files
            os.remove(magPath)
            os.remove(specPath)

            # load data with pickle.load(doc['np_data']['data])
            
            # Using SON Manipulator:
            # print c.find_one({"_id": modelName})
            # print "type:", type(npDataSmall)
            # c.update({"_id": modelName}, {"$set": {"np_data": npDataSmall}})
            # print c.find_one({"_id": modelName})['np_data']
            # print "update complete!"

    def _splice_mag_spec_arrays(self, magData, specData, nLambda):
        """Add spectrum to teh magData structured array.
        
        .. note:: This is done by heavy-duty copying. It'd probably be
           much more efficient to figure out if
           :func:`numpy.recfunctions.append_fields` can be applied here. But
           I can't get it to work.
        """
        magDtype = [('age',np.float),('mass',np.float),('lbol',np.float),('sfr',np.float)]
        magDtype += [(name,np.float) for (idx,name,comment) in FILTER_LIST]
        dt = magDtype + [('spec',np.float,nLambda)]
        nRows = len(magData)
        allData = np.empty(nRows, dtype=dt)
        for t in magDtype:
            allData[t[0]] = magData[t[0]]
        allData['spec'] = specData['spec']
        return allData


class SpecParser(object):
    """Parses spectral tables generated by FSPS
    
    Parameters
    ----------

    specPath : str
        Path to the .spec file.
    specType : str, 'basel' or 'miles'
        The spectral catalog used by FSPS to generate this spectrum.

    Attributes
    ----------

    data : numpy record array
        Record arrays with columns `age`, `mass`, `lbol`, `sfr`, and `spec`.
        Each item in teh `spec` column is the length of `wavelenghs`.
    wavelengths : ndarray (1D)
        Wavelength axis of the spectra.
    """
    def __init__(self, specPath, specType='basel'):
        #super(SpecParser, self).__init__()
        self.specPath = specPath
        self.specType = specType
        nLambda = SpecParser.nlambda(specType)
        
        f = open(specPath, 'r')
        allLines = f.readlines()
        nRows, nLambda = [int(a) for a in allLines[8].split()]

        dtype = [('age',np.float),('mass',np.float),('lbol',np.float),
                ('sfr',np.float),('spec',np.float,nLambda)]
        labelCols = ('age','mass','lbol','sfr')
        self.data = np.zeros(nRows, dtype=dtype)
        for i in xrange(nRows):
            iLabel = 10 + i*2
            iSpec = iLabel + 1
            for c,x in zip(labelCols, allLines[iLabel].split()):
                self.data[c][i] = float(x)
            spec = np.array([float(x) for x in allLines[iSpec].split()])
            self.data['spec'][i] = spec
        f.close()
    
    @classmethod
    def nlambda(cls, specType):
        """Get the number of wavelengths expected in this spectrum."""
        if specType == 'basel':
            return 1963 # as of v2.3
        elif specType == 'miles':
            return 4222 # FIXME for v2.3
        else:
            assert "spec type invalid:", specType

    @property
    def wavelengths(self):
        """Get the numpy wavelength vector."""
        fspsDir = os.environ['SPS_HOME'] # environment variable for FSPS
        # print fspsDir
        if self.specType == "basel":
            lambdaPath = os.path.join(fspsDir,"SPECTRA","BaSeL3.1","basel.lambda")
        elif self.specType == "miles":
            lambdaPath = os.path.join(fspsDir,"SPECTRA","MILES","miles.lambda")
        else:
            assert "spec type invalid:", self.specType
        lam = np.loadtxt(lambdaPath)
        return lam

class MagParser(object):
    """Parses magnitude tables generated by FSPS.
    
    Parameters
    ----------

    magPath : str
       Path to the .mag file

    Attributes
    ----------

    data : record array
       Has columns `age`, `mass`, `lbol`, `sfr`, and magnitudes listed in
       the `FILTER_NAMES` module attribute.
    """
    def __init__(self, magPath):
        #super(MagParser, self).__init__()
        self.magPath = magPath
        
        # Define data type in same order as .mag columns so we can
        # generate a .mag record array in one line
        dt = [('age',np.float),('mass',np.float),('lbol',np.float),('sfr',np.float)]
        dt += [(name,np.float) for (idx,name,comment) in FILTER_LIST]
        self.data = np.loadtxt(magPath, comments="#", dtype=dt)
        self.data = np.atleast_1d(self.data) # to protect against 1 age results
        # NOTE be careful here as older outputs may not have the complete
        # set of outputs; perhaps allow for *fewer* filters
        # NOTE that loadtxt has no compatibility with bad data; use genfromtxt
        # http://docs.scipy.org/doc/numpy/user/basics.io.genfromtxt.html
        # genfromtxt is slower; perhaps use it as a backup?


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
    
    def command(self):
        """Write the string for this paramter set."""
        # These are pset variables, (aside from sfh and zmet)
        dt = [("zred","%.2f"),("tau","%.10f"),("const","%.4f"),
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

class NumpySONManipulator(SONManipulator):
    """Taken from Dan-FM's gist: https://gist.github.com/1143729"""
    def transform_incoming(self, value, collection):
        if isinstance(value, (list,tuple,set)):
            return [self.transform_incoming(item,collection) for item in value]
        if isinstance(value,dict):
            return dict((key,self.transform_incoming(item,collection))
                                         for key,item in value.iteritems())
        if isinstance(value,np.ndarray):
            return {'_type': 'np.ndarray', 
                    'data': Binary(pickle.dumps(value,-1))}
        return value

    def transform_outgoing(self, son, collection):
        if isinstance(son,(list,tuple,set)):
            return [self.transform_outgoing(value,collection) for value in son]
        if isinstance(son,dict):
            if son.get('_type') == 'np.ndarray':
                return pickle.loads(son.get('data'))
            return dict((key,self.transform_outgoing(value,collection))
                                         for key,value in son.iteritems())
        return son

class SPTable(object):
    """A reduction of stellar populations into a HDF5 table."""
    def __init__(self, arg):
        super(SPTable, self).__init__()
        self.arg = arg
    


if __name__ == '__main__':
    pass
