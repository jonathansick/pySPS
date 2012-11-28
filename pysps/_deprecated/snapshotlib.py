#!/usr/bin/env python
# encoding: utf-8
"""
For building stellar population libraries at a single observed age.

History
-------
2011-10-02 - Created by Jonathan Sick
"""

import os
import socket
import multiprocessing
import datetime
import pytz
import subprocess


import pymongo
import numpy as np

import splib
from fspsq import FSPSLibrary
from fspsq import ParameterSet
from fspsq import _make_shell_command
from fspsq import _make_common_var_sets
from fspsq import NumpySONManipulator


def main():
    pass

class SnapshotLibrary(FSPSLibrary):
    """Baseclass for a single-observed-age stellar population library."""
    def __init__(self, libname, dbname="fsps", age=13.7):
        super(SnapshotLibrary, self).__init__(libname, dbname=dbname)
        self.age = age
    
    def regster_pset(self, pset):
        """Adds a ParameterSet to the collection of model definitions.
        
        .. note:: the `tage` parameter is forced to be equal to `self.age`.
        """
        print "Special init"
        self.remove_existing_model(pset.name)
        doc = {"_id": pset.name, "pset": pset.get_doc(),
            "compute_started": False, "compute_complete": False}
        doc['pset']['tage'] = float(self.age)
        print doc
        self.collection.insert(doc)

    def compute_models(self, nThreads=1, maxN=10, clean=True):
        """Fire up FSPS to generate SP models.
        
        Parameters
        ----------
        
        nThreads : int, optional
            Number of processes to be spawned, for multiprocessing. If set
            to `1`, multiprocessing is not used; useful for debugging.
        maxN : int, optional
            Maximum number of jobs that can be given to a single
            `fspsq` process.
        clean : bool, default `True`
            When `True`, fspsq output file will be cleaned up.
        """
        fspsqPath = os.path.join(os.getcwd(), "fspsq") # echo for debug
        # fspsqPath = "echo"
        args = []
        cmdPaths = []
        for i in range(nThreads):
            processorName = "processor-%i" % i
            commandPath = processorName+".txt"
            cmdPaths.append(commandPath)
            args.append((self.libname, self.dbname, self.host, self.port,
                maxN, fspsqPath, commandPath))
        # print args
        if nThreads > 1:
            pool = multiprocessing.Pool(processes=nThreads)
            pool.map(run_fspsq, args)
        else:
            map(run_fspsq, args)

        if clean:
            for cmdPath in cmdPaths:
                os.remove(cmdPath)

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


def _gather_fsps_outputs(c, modelNames):
    """Adds the .mag and .spec outputs for each model to MongoDB.
    
    .. note:: Need to propagate the spec type through here...
    """
    fspsDir = os.environ['SPS_HOME'] # environment variable for FSPS
    outputDir = os.path.join(fspsDir, "OUTPUTS")
    # print modelNames
    failed = False
    for modelName in modelNames:
        magPath = os.path.join(outputDir, modelName+".out.mags")
        magSpec = os.path.join(outputDir, modelName+".out.spec")
        print "ingesting", modelName
        npData = ingest_output(magPath, magSpec)
        #try:
        #    npdata = fspsq.ingest_output(magPath, magSpec)
        #    failed = False
        #except:
        #    # Some exception in parsing the text files; skip the output
        #    c.update({"_id": modelName}, {"$set": {"compute_complete": True,
        #        "ingest_fail": True}})
        #    print "\t", modelName, "ingest fail"
        #    failed = True
        if failed == True: continue
        #print npDataSmall.dtype
        spCols = ['age','mass','lbol','sfr']
        obsDoc = {}
        for col in spCols:
            obsDoc[col] = float(npData[col])
        for band in fspsq.FILTER_LIST:
            col = band[1]
            #print col
            obsDoc[col] = float(npData[col])
        c.update({"_id": modelName}, {"$set": {"compute_complete": True,
            "obs": obsDoc}})
        # Remove data files
        #os.remove(magPath)
        os.remove(magSpec)

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
    dt = [('age',np.float),('mass',np.float),('lbol',np.float),('sfr',np.float)]
    dt += [(name,np.float) for (idx,name,comment) in fspsq.FILTER_LIST]
    
    # Load magnitude data
    magData = np.loadtxt(magPath, comments="#")
    data = np.empty(1, dtype=dt)
    print data['age'][0]
    print magData[0]
    data['age'][0] = magData[0]
    data['mass'][0] = magData[1]
    data['lbol'][0] = magData[2]
    data['sfr'][0] = magData[3]
    for (filterN,name,comment) in fspsq.FILTER_LIST:
        col = filterN + 4 -1
        # print col, name, magData.shape
        data[name] = magData[col]
        # NOTE be careful here are older outputs may not have the complete
        # set of outputs; perhaps allow for *fewer* filters
    return data

if __name__ == '__main__':
    main()


