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

from bson.binary import Binary
from pymongo.son_manipulator import SONManipulator

from sp_params import ParameterSet
from sp_params import get_metallicity_LUT
from sp_params import FILTER_LIST

from parsers import MagParser
from parsers import SpecParser

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

        The idea is that some of the fortran COMMON BLOCK parameters cannot
        be changed after `sps_setup` is called. Thus each run of fspsq is
        with models that share common block paramters.
        
        .. note:: This merely lists all *possible* combinations of common variable
        configurations. There is not guarantee that there are models needing
        computation for each configuration set.
        
        Parameters
        ----------
        c : pymongo.Collection instance
        
        Returns
        -------
        List of dictionaries. Each dictionary has keys representing common
        variables: sfh, dust_type, imf_type, compute_vega_mags, redshift_colors
        """
        params = ['sfh', 'dust_type', 'imf_type',
            'compute_vega_mags', 'redshift_colors']
        possibleValues = {}
        for param in params:
            possibleValues[param] = self.collection.distinct("pset."+param)
        groups = []
        for isfh in possibleValues['sfh']:
            for idust_type in possibleValues['dust_type']:
                for iimf_type in possibleValues['imf_type']:
                    for ivega in possibleValues['compute_vega_mags']:
                        for iredshift in possibleValues['redshift_colors']:
                            groups.append({"pset.sfh":isfh,
                                "pset.dust_type":idust_type,
                                "pset.imf_type":iimf_type,
                                "pset.compute_vega_mags":ivega,
                                "pset.redshift_colors":iredshift})
        return groups

    def _make_shell_command(self, fspsqPath, commandFilePath, nModels, varSet):
        """Crafts the command for running `fspsq`, returning a string."""
        cmd = "%s %s %i %i %i %i %i %i" % (fspsqPath, commandFilePath,
                nModels, varSet['pset.sfh'], varSet['pset.dust_type'],
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
