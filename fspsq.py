"""Generates input files for fspsq."""
import socket
import os
import subprocess
import multiprocessing
import datetime

import pytz
import pymongo

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
        print args
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

def run_fspsq(args):
    """A process for running fsps jobs."""
    print "hello"
    libname, dbname, host, port, maxN, fspsqPath, commandPath = args
    thisHost = socket.gethostname() # hostname of current process
    
    # Connect to the library in MongoDB
    connection = pymongo.Connection(host=host, port=port)
    db = connection[dbname]
    collection = db[libname]
    
    commonVarSets = _make_common_var_sets(collection)
    print "commonVarSets:", commonVarSets
    for varSet in commonVarSets:
        while True:
            psets = []
            now = datetime.datetime.utcnow()
            now = now.replace(tzinfo=pytz.utc)
            while len(psets) <= maxN:
                q = {"compute_complete": False, "compute_started": False}
                q.update(varSet)
                print "q", q
                doc = collection.find_and_modify(query=q,
                    update={"$set": {"compute_started": True,
                                     "compute_date": now,
                                     "compute_host": thisHost}},)
                print "doc", doc
                if doc is None: break # no available models
                modelName = str(doc['_id'])
                pset = ParameterSet(modelName, **doc['pset'])
                psets.append(pset)
            if len(psets) == 0: break # empty job queue
            # Startup a computation: write command file and start fspsq
            cmdTxt = "\n".join([p.command() for p in psets])+"\n"
            if os.path.exists(commandPath): os.remove(commandPath)
            f = open(commandPath, 'w')
            f.write(cmdTxt)
            f.close()
            shellCmd = _make_shell_command(fspsqPath, commandPath, varSet)
            print "cmd::", shellCmd
            subprocess.call(shellCmd, shell=True)

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

def _make_shell_command(fspsqPath, commandFilePath, varSet):
    """Crafts the command for running `fspsq`, returning a string."""
    params = ['sfh', 'zmet', 'dust_type', 'imf_type',
        'compute_vega_mags', 'redshift_colors']
    cmd = "%s %s %i %i %i %i %i %i" % (fspsqPath, commandFilePath,
        varSet['pset.sfh'], varSet['pset.zmet'], varSet['pset.dust_type'],
        varSet['pset.imf_type'], varSet['pset.compute_vega_mags'],
        varSet['pset.redshift_colors'])
    return cmd

class ParameterSet(object):
    """An input parameter set for a FSPS model run."""
    def __init__(self, name, **kwargs):
        self.name = name # name of this model
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
        cmd = self.name + " " + " ".join([s % self.p[k] for (k,s) in dt])
        return cmd
    
    def get_doc(self):
        """Returns the document dictionary to insert in MongoDB."""
        return self.p

if __name__ == '__main__':
    pass