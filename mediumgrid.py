"""Set up a test grid of single-metallicity CSPs with different taus, and
difference metallicities.
"""
import os
import cPickle as pickle

import numpy as np
import tables
from bson.binary import Binary

import matplotlib.mlab as mlab # for recarray helpers
import matplotlib.pyplot as plt

import fspsq
from fspsq import FSPSLibrary
from fspsq import ParameterSet

def main():
    mediumLibrary = MediumGrid("mediumgrid", dbname="fsps")
    # mediumLibrary.reset()
    # mediumLibrary.generate_grid()
    # mediumLibrary.compute_models(nThreads=6, maxN=50)
    # mediumLibrary.create_mag_table("medium_table.h5")
    # histogram_2MASSJ("medium_table.h5")
    nir_cmd("medium_table.h5")

class MediumGrid(FSPSLibrary):
    """A small grid SSPs for three metallicities."""
    
    def generate_grid(self):
        """Create the model grid."""
        zmets = range(1,23)
        taus = np.arange(0.1, 10, 0.1)
        print zmets
        print taus
        i = 0
        for zmet in zmets:
            for tau in taus:
                modelName = "model%i" % i
                pset = ParameterSet(modelName, sfh=1, zmet=zmet, tau=float(tau))
                self.register_pset(pset)
                i += 1
        print "There are %i models" % self.count_models()
    
    def create_mag_table(self, outputPath, isocType="pdva", specType="basel"):
        """Create an HDF5 table of that describes a set of magnitudes."""
        if os.path.exists(outputPath): os.remove(outputPath)
        title = os.path.splitext(os.path.basename(outputPath))[0]
        h5file = tables.openFile(outputPath, mode="w", title=title)
        table = h5file.createTable("/", 'mags', MagTableDef, "Mag Model Table")
        print h5file
        docs = self.collection.find({"compute_complete":True,
            "np_data": {"$exists": 1}}) # , limit=2
        print "working on %i docs to read" % docs.count()
        lut = get_metallicity_LUT(isocType, specType)
        for doc in docs:
            print "reading", doc['_id']
            # print doc.keys()
            # print doc['np_data']
            npData = doc['np_data']
            # print npData.dtype
            # binData = Binary(doc['np_data']['data'])
            # print type(binData)
            # npData = pickle.load(binData)
            nRows = len(npData)
            # Append model information (about SFH, dust, etc)
            zmet = doc['pset']['zmet']
            Z = lut[zmet-1]
            zmets = np.ones(nRows, dtype=np.float) * Z
            tau = doc['pset']['tau']
            taus = np.ones(nRows, dtype=np.float) * tau
            npDataAll = mlab.rec_append_fields(npData, ['Z','tau'],[zmets,taus])
            # Trim the recarray to just the desired fields
            npDataTrim = mlab.rec_keep_fields(npDataAll,
                ['Z','tau','age','mass','lbol','sfr','TMASS_J','TMASS_H',
                'TMASS_Ks','MegaCam_u','MegaCam_g','MegaCam_r','MegaCam_i',
                'MegaCam_z','GALEX_NUV','GALEX_FUV'])
            for i in xrange(nRows):
                row = npDataTrim[i]
                print row['Z'], row['tau'],row['TMASS_J'],row['TMASS_Ks']
            # Append to HDF5
            table.append(npDataTrim)
        h5file.flush()
        h5file.close()

def get_metallicity_LUT(isocType, specType):
    """docstring for as_metallicity"""
    if isocType=="pdva" and specType=="basel":
        return (0.0002, 0.0003, 0.0004, 0.0005,0.0006,0.0008,0.0010,
            0.0012,0.0016,0.0020,0.0025,0.0031,0.0039,0.0049,0.0061,
            0.0077,0.0096,0.0120,0.0150,0.0190,0.0240,0.0300)

class MagTableDef(tables.IsDescription):
    """Column definition for a magnitude table. Assumes sfh=1 (a 5-param SFH)"""
    # names correspond to names in the ingested record array
    # 64-bit floats correspond to the default numpy floats
    # data corruption will occur if types mis-match (no casting is done)
    zmet = tables.Float64Col() # logZ/Z_solar
    tau = tables.Float64Col() # tau
    age = tables.Float64Col() # log Age
    mass = tables.Float64Col() # logMass
    lbol = tables.Float64Col() # log Lbol
    sfr = tables.Float64Col()
    TMASS_J = tables.Float64Col()
    TMASS_H = tables.Float64Col()
    TMASS_Ks = tables.Float64Col()
    MegaCam_u = tables.Float64Col()
    MegaCam_g = tables.Float64Col()
    MegaCam_r = tables.Float64Col()
    MegaCam_i = tables.Float64Col()
    MegaCam_z = tables.Float64Col()
    GALEX_NUV = tables.Float64Col()
    GALEX_FUV = tables.Float64Col()

def histogram_2MASSJ(tablePath):
    h5file = tables.openFile(tablePath)
    table = h5file.root.mags
    a = table[:2].view(np.recarray)
    print a,a.dtype
    tmassJ = [x['TMASS_J'] for x in table.where("""(TMASS_J > -10) & (TMASS_J < 30)""")]
    # print tmassJ
    tmassJ = np.array(tmassJ)
    
    plotPath = "hist_J"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(tmassJ, histtype='step') # bins=np.arange(0.,3.,0.1)
    fig.savefig(plotPath+".pdf", format='pdf')
    h5file.close()

def nir_cmd(tablePath):
    h5file = tables.openFile(tablePath)
    table = h5file.root.mags
    mags = np.array([[x['TMASS_J'],x['TMASS_Ks']] for x in table.where("""(TMASS_J > -10) & (TMASS_J < 30)""")])
    c = mags[:,0] - mags[:,1]
    Ks = mags[:,1]
    
    plotPath = "cmd_JK"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(c, Ks, s=1.) # bins=np.arange(0.,3.,0.1)
    fig.savefig(plotPath+".pdf", format='pdf')
    h5file.close()

if __name__ == '__main__':
    main()