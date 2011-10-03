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
        # fspsqPath = "echo"
        args = []
        for i in range(nThreads):
            processorName = "processor-%i" % i
            commandPath = processorName+".txt"
            args.append((self.libname, self.dbname, self.host, self.port,
                maxN, fspsqPath, commandPath))
        # print args
        if nThreads > 1:
            pool = multiprocessing.Pool(processes=nThreads)
            pool.map(run_fspsq, args)
        else:
            map(run_fspsq, args)
    
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

def run_fspsq(args):
    """A process for running fsps jobs."""
    # print "hello"
    libname, dbname, host, port, maxN, fspsqPath, commandPath = args
    thisHost = socket.gethostname() # hostname of current process
    
    # Connect to the library in MongoDB
    connection = pymongo.Connection(host=host, port=port)
    db = connection[dbname]
    db.add_son_manipulator(NumpySONManipulator())
    collection = db[libname]
    
    commonVarSets = _make_common_var_sets(collection)
    # print "commonVarSets:", commonVarSets
    for varSet in commonVarSets:
        while True:
            psets = []
            modelNames = []
            now = datetime.datetime.utcnow()
            now = now.replace(tzinfo=pytz.utc)
            while len(psets) <= maxN:
                q = {"compute_complete": False, "compute_started": False}
                q.update(varSet)
                # print "q", q
                doc = collection.find_and_modify(query=q,
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
            shellCmd = _make_shell_command(fspsqPath, commandPath, nModels, varSet)
            print "cmd::", shellCmd
            subprocess.call(shellCmd, shell=True)
            # Get output data from FSPS
            _gather_fsps_outputs(collection, modelNames)

def _make_common_var_sets(c):
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
        possibleValues[param] = c.distinct("pset."+param)
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

def _make_shell_command(fspsqPath, commandFilePath, nModels, varSet):
    """Crafts the command for running `fspsq`, returning a string."""
    params = ['sfh', 'zmet', 'dust_type', 'imf_type',
        'compute_vega_mags', 'redshift_colors']
    cmd = "%s %s %i %i %02i %i %i %i %i" % (fspsqPath, commandFilePath, nModels,
        varSet['pset.sfh'], varSet['pset.zmet'], varSet['pset.dust_type'],
        varSet['pset.imf_type'], varSet['pset.compute_vega_mags'],
        varSet['pset.redshift_colors'])
    return cmd

def _gather_fsps_outputs(c, modelNames):
    """Adds the .mag and .spec outputs for each model to MongoDB.
    
    .. note:: Need to propagate the spec type through here...
    """
    fspsDir = os.environ['SPS_HOME'] # environment variable for FSPS
    outputDir = os.path.join(fspsDir, "OUTPUTS")
    # print modelNames
    for modelName in modelNames:
        magPath = os.path.join(outputDir, modelName+".out.mags")
        magSpec = os.path.join(outputDir, modelName+".out.spec")
        print "ingesting", modelName
        try:
            npdata = ingest_output(magPath, magSpec)
        except:
            # Some exception in parsing the text files; skip the output
            c.update({"_id": modelName}, {"$set": {"compute_complete": True,
                "ingest_fail": True}})
            print "\t", modelName, "ingest fail"
            continue
        npDataSmall = npdata[:1]
        c.update({"_id": modelName}, {"$set": {"compute_complete": True}})
        binData = Binary(pickle.dumps(npdata,-1))
        c.update({"_id": modelName},
            {"$set": {"np_data": {'_type': 'np.ndarray', 'data':binData}}})
        # Remove data files
        os.remove(magPath)
        os.remove(magSpec)
        # load data with pickle.load(doc['np_data']['data])
        
        # Using SON Manipulator:
        # print c.find_one({"_id": modelName})
        # print "type:", type(npDataSmall)
        # c.update({"_id": modelName}, {"$set": {"np_data": npDataSmall}})
        # print c.find_one({"_id": modelName})['np_data']
        # print "update complete!"

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


class FSPSParser(object):
    """Reads the mag and spectral output files."""
    def __init__(self, arg):
        super(FSPSParser, self).__init__()
        self.arg = arg

class SpecParser(object):
    """Parses spectral tables generated by FSPS
    
    Parameters
    ----------

    specPath : str
        Path to the .spec file.
    specType : str, 'basel' or 'miles'
        The spectral catalog used by FSPS to generate this spectrum.
    """
    def __init__(self, specPath, specType='basel'):
        #super(SpecParser, self).__init__()
        self.specPath = specPath
        self.specType = specType
        nLambda = self._get_nlambda()
        
        f = open(specPath, 'r')
        allLines = f.readlines()
        nRows = int(allLines[8])

        dtype = [('age',np.float),('mass',np.float),('lbol',np.float),
                ('sfr',np.float),('spec',np.float,nLambda)]
        labelCols = ('age','mass','lbol','sfr')
        self.data = np.zeros(nRows, dtype=dtype)
        for i in xrange(nRows):
            iLabel = 9 + i*2
            iSpec = iLabel + 1
            for c,x in zip(labelCols, allLines[iLabel].split()):
                self.data[c][i] = float(x)
            spec = [float(x) for x in allLines[iSpec].split()]
            self.data['spec'][i] = spec
        f.close()

    def _get_nlambda(self):
        """Get the number of wavelengths expected in this spectrum."""
        if self.specType == 'basel':
            return 1221
        elif self.specType == 'miles':
            return 4222
        else:
            assert "spec type invalid:", self.specType

    @property
    def wavelengths(self):
        """Return the numpy wavelength vector."""
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


def ingest_output(magPath, specPath, specType='basel'):
    """Read FSPS mag and spec output files into a numpy recarray.
    
    The magnitudes and spectral data are merged into a single numpy structure.
    
    Parameters
    ----------
    
    magPath: str
        Path to the *.mag output file.
    specPath: str
        Path to the *.spec output file
    specType: str
        Spectral library. `basti` or `miles`.
    
    Returns
    -------
    
    A NumPy record array instance.
    """
    lamb = _read_wavelengths(specType)
    nLambda = len(lamb) # length of wavelength vector
    dt = [('age',np.float),('mass',np.float),('lbol',np.float),('sfr',np.float),
        ('spec',np.float,(nLambda,))]
    dt += [(name,np.float) for (idx,name,comment) in FILTER_LIST]
    
    # Load magnitude data
    magData = np.loadtxt(magPath, comments="#")
    nRows = magData.shape[0]
    data = np.empty(nRows, dtype=dt)
    data['age'] = magData[:,0]
    data['mass'] = magData[:,1]
    data['lbol'] = magData[:,2]
    data['sfr'] = magData[:,3]
    for (filterN,name,comment) in FILTER_LIST:
        col = filterN + 4 -1
        # print col, name, magData.shape
        data[name] = magData[:,col]
        # NOTE be careful here are older outputs may not have the complete
        # set of outputs; perhaps allow for *fewer* filters
    # Load Spectra
    specData = read_spec_table(specPath, nLambda, nRows)
    for i in xrange(nRows):
        data['spec'][i] = specData[i]
    return data

def read_spec_table(specPath, nLambda, nRows):
    """Makes a numpy array with each spectrum in a row. Meant to be used in
    ingest_output; this won't give age,mass,luminosity context."""
    f = open(specPath, 'r')
    i = 0
    data = np.zeros([nRows,nLambda], dtype=np.float)
    for line in f:
        if line[0] is not " ": continue
        lineParts = line.split()
        if len(lineParts) == 4: continue
        data[i,:] = np.array(lineParts)
        i += 1
    f.close()
    return data

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
        (57,"WISE_W1","WISE W1 (http://www.astro.ucla.edu/~wright/WISE/passbands.html)"),
        (58,"WISE_W2","WISE W2 (http://www.astro.ucla.edu/~wright/WISE/passbands.html)"),
        (59,"WISE_W3","WISE W3 (http://www.astro.ucla.edu/~wright/WISE/passbands.html)"),
        (60,"WFC3_F125W","WFC3 F125W (ftp://ftp.stsci.edu/cdbs/comp/wfc3/)"),
        (61,"WFC3_F160W","WFC3 F160W (ftp://ftp.stsci.edu/cdbs/comp/wfc3/)"),
        (62,"UVOT_W2","UVOT W2 (from Erik Hoversten, 2011)"),
        (63,"UVOT_M2","UVOT M2 (from Erik Hoversten, 2011)"),
        (64,"UVOT_W1","UVOT W1 (from Erik Hoversten, 2011)")]

def _read_wavelengths(specType):
    """Read the wavelength vector given the spectral library."""
    fspsDir = os.environ['SPS_HOME'] # environment variable for FSPS
    # print fspsDir
    if specType == "basel":
        lambdaPath = os.path.join(fspsDir,"SPECTRA","BaSeL3.1","basel.lambda")
    elif specType == "miles":
        lambdaPath = os.path.join(fspsDir,"SPECTRA","MILES","miles.lambda")
    else:
        print "spec type invalid:", specType
    lam = np.loadtxt(lambdaPath)
    return lam

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

if __name__ == '__main__':
    pass
