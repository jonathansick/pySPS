"""Basic Monte Carlo stellar population library."""
import os
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import bson
import tables

import fspsq
from fspsq import FSPSLibrary
from fspsq import ParameterSet

from mediumgrid import get_metallicity_LUT
from mediumgrid import MagTableDef

from griddata import griddata


def main():
    #library = MonteCarloLibrary("mc.1", dbname='fsps')
    #library.reset()
    #library.define_samples()
    #library.compute_models(nThreads=8, maxN=50)
    #library.age_grid()
    #library.plot_sample_sfh()

    #library = MonteCarloLibrary2("mc.2", dbname='fsps')
    #library.reset()
    #library.define_samples(n=1)
    #library.compute_models(nThreads=1, maxN=50)
    #library.plot_sample_sfh()

    library = MonteCarloLibrary2("mc.3", dbname='fsps')
    #library.reset()
    #library.define_samples(n=10000)
    #library.compute_models(nThreads=6, maxN=50)
    #library.plot_parameter_hists()
    #library.create_mag_table("mc3.h5", t=13.7)
    library.colour_histogram("mc3.h5", ("MegaCam_i","TMASS_Ks"))
    library.bin_cc_index(("MegaCam_g","MegaCam_i"),("MegaCam_i","TMASS_Ks"),
            "mc3.h5")

class MonteCarloLibrary(FSPSLibrary):
    """Demonstration of a Monte Carlo sampled stellar pop library."""
    
    def define_samples(self, n=50000):
        """Define the set of models."""
        for i in xrange(n):
            pset = ParameterSet(None, # automatically create a name
                sfh=1,
                imf_type=1, # Chabrier 2003
                dust_type=2, # Calzetti 2000 attenuation curve
                zmet=int(self._sample_zmet()),
                tau=float(self._sample_tau()),
                const=float(self._sample_const()),
                sf_start=float(self._sample_sf_start()),
                fburst=float(self._sample_fburst()),
                tburst=float(self._sample_tburst()),
                dust2=float(self._sample_dust2())
                )
            self.register_pset(pset)
    
    def _sample_zmet(self):
        """Returns a random metallicity"""
        return np.random.randint(1,23)
    
    def _sample_tau(self):
        """Returns a random e-folding of SFR"""
        return np.random.uniform(0.1, 10.)
    
    def _sample_const(self):
        """Returns fraction of mass formed as a constant mode of SF"""
        return np.random.uniform(0.0,0.1)
    
    def _sample_sf_start(self):
        """Start time of SFH in Gyr"""
        return np.random.uniform(0.,7.)
    
    def _sample_fburst(self):
        """Fraction of mass formed in an instantaneous burst of SF."""
        return np.random.uniform(0.,2.)
    
    def _sample_tburst(self):
        """Time of the burst after the BB."""
        return np.random.uniform(0.,12.5)
    
    def _sample_dust2(self):
        """Optical depth of ambient ISM."""
        return np.random.uniform(0.1,1.)

    def age_grid(self):
        """Print the grid of ages from a random output."""
        doc = self.collection.find_one({"compute_complete":True,
             "np_data": {"$exists": 1}})
        npData = doc['np_data']
        print npData.dtype
        ages = npData['age']
        agesGyr = 10.**ages / 10.**9
        for i, age in enumerate(agesGyr):
            print i, age

    def plot_sample_sfh(self):
        """Plots a random SFH"""
        fig = plt.figure(figsize=(6,6))
        axMass = fig.add_subplot(511)
        axL = fig.add_subplot(512)
        axSFR = fig.add_subplot(513)
        axGI = fig.add_subplot(514)
        axML = fig.add_subplot(515)
        
        #doc = self.collection.find_one({"compute_complete":True,
        #    "np_data": {"$exists": 1}, "pset.tau": {"$lt":1.},
        #    "pset.fburst": {"$lt":0.5}})
        doc = self.collection.find_one({"compute_complete":True,
            "np_data": {"$exists": 1}})
        npData = doc['np_data']

        keys = ['sf_start', 'tburst', 'fburst', 'const']
        for k in keys:
            print k, doc['pset'][k]

        start = doc['pset']['sf_start']

        logAge = npData['age']
        logL = npData['lbol']
        logMass = npData['mass']
        logSFR = npData['sfr']
        g = npData['MegaCam_g']
        i = npData['MegaCam_i']
        gi = g - i
        ML = logMass - logL

        age = 10.**logAge / 10.**9

        axMass.plot(age, logMass, '-')
        axL.plot(age, logL, '-')
        axSFR.plot(age, logSFR, '-')
        axGI.plot(age, gi, '-')
        axML.plot(age, ML, '-')

        no_xticklabels(axMass)
        no_xticklabels(axL)
        no_xticklabels(axSFR)
        no_xticklabels(axGI)
        #axMass.set_xlim(max(logAge),min(logAge))
        #axL.set_xlim(max(logAge),min(logAge))
        #axSFR.set_xlim(max(logAge),min(logAge))
        #axGI.set_xlim(max(logAge),min(logAge))
        #axML.set_xlim(max(logAge),min(logAge))
        axMass.set_xlim(start,13.7)
        axL.set_xlim(start,13.7)
        axSFR.set_xlim(start,13.7)
        axGI.set_xlim(start,13.7)
        axML.set_xlim(start,13.7)
        axML.set_ylim(-1.4,1.)
        axMass.set_ylim(-2,1.)
        axSFR.set_ylim(-12.,-7.)

        axMass.set_ylabel(r"$\log_{10} M$")
        axL.set_ylabel(r"$\log_{10} L$")
        axSFR.set_ylabel(r"$\log_{10} \mathrm{SFR}$")
        axGI.set_ylabel(r"$g-i$")
        axML.set_ylabel(r"$\log_{10} M/L$")
        #axML.set_xlabel(r"$\log_{10} \mathrm{Age}$")
        axML.set_xlabel("Gyr after BB")

        fig.savefig("sfh.pdf", format="pdf")

    def plot_parameter_hists(self):
        """Plot a histogram of each of the parameters in teh library."""
        fig = plt.figure(figsize=(6,6))
        axZ = fig.add_subplot(221)
        axTau = fig.add_subplot(222)
        axFburst = fig.add_subplot(223)
        axTburst = fig.add_subplot(224)

        docs = self.collection.find({"compute_complete":True,
            "np_data": {"$exists": 1}},
            ['pset.tau','pset.zmet','pset.fburst','pset.tburst'])
        taus = []
        zmets = []
        fbursts = []
        tbursts = []
        for doc in docs:
            taus.append(doc['pset']['tau'])
            zmets.append(doc['pset']['zmet'])
            fbursts.append(doc['pset']['fburst'])
            tbursts.append(doc['pset']['tburst'])
        taus = np.array(taus)
        zmets = np.array(zmets)
        fbursts = np.array(fbursts)
        tbursts = np.array(tbursts)
        axZ.hist(zmets, histtype='step')
        axTau.hist(taus, histtype='step')
        axFburst.hist(fbursts, histtype='step')
        axTburst.hist(tbursts, histtype='step')
        fig.savefig("parameters.pdf", format="pdf")

    def colour_histogram(self, h5path, cInd):
        """docstring for colour_histogram"""
        h5file = tables.openFile(h5path, mode="r")
        table = h5file.root.mags
        
        c = np.array([x[cInd[0]]-x[cInd[1]] for x in table])
        c = c[c<5.]

        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111)
        ax.hist(c, bins=100)
        ax.set_xlabel("%s - %s" % cInd)
        fig.savefig("color_hist.pdf", format="pdf")
        h5file.close()

    def create_mag_table(self, outputPath, t=13.7,
            isocType="pdva", specType="basel"):
        """Create an HDF5 table of that describes a set of magnitudes for
        stellar population realizations at a defined age (gyr from BigBang)"""
        if os.path.exists(outputPath): os.remove(outputPath)
        title = os.path.splitext(os.path.basename(outputPath))[0]
        h5file = tables.openFile(outputPath, mode="w", title=title)
        table = h5file.createTable("/", 'mags', MagTableDef, "Mag Model Table")
        print h5file
        docs = self.collection.find({"compute_complete":True,
            "np_data": {"$exists": 1}}) # , limit=2
        print "working on %i docs to read" % docs.count()
        lut = get_metallicity_LUT(isocType, specType)
        rows = []
        cols = ['Z','tau','age','mass','lbol','sfr','TMASS_J','TMASS_H',
                'TMASS_Ks','MegaCam_u','MegaCam_g','MegaCam_r','MegaCam_i',
                'MegaCam_z','GALEX_NUV','GALEX_FUV']
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
            # select row closest to the target age
            ageGyr = 10.**npDataTrim['age'] / 10.**9
            i = np.argmin((ageGyr - t)**2)
            row = np.array(npDataTrim[i], copy=True)
            print i, row.shape, row.dtype
            print row['Z'], row['tau'],row['TMASS_J'],row['TMASS_Ks']
            rows.append(row)
            # Append to HDF5
            for col in cols:
                table.row[col] = row[col]
            table.row.append()
            #table.append(npDataTrim)
        #mlab.recs_join(key, name, rows, jointype='outer', missing=0.0, postfixes=None)
        h5file.flush()
        h5file.close()

    def bin_cc_index(self, c1I, c2I, h5Path):
        """Bin the mag table into a colour colour diagram.

        Produces an color-color table with indexes into models in
        the mags HDF5 table.

        Parameters
        ----------

        c1I : tuple of two str
            Names of the two bands that make the first colour
        c2I : tuple of two str
            Names of the two bands that make the second colour
        """
        h5file = tables.openFile(h5Path, mode='a')
        magTable = h5file.root.mags
        c1 = np.array([x[c1I[0]]-x[c1I[1]] for x in magTable])
        c2 = np.array([x[c2I[0]]-x[c2I[1]] for x in magTable])
        mass = np.array([x['mass'] for x in magTable])
        logL = np.array([x['lbol'] for x in magTable])
        logML = mass - logL
        grid, gridN, wherebin = griddata(c1, c2, logML, binsize=0.05,
                retbin=True, retloc=True)
        print "grid", grid.shape
        # Set up the cc table
        if 'cc' in h5file.root:
            print "cc already exists"
            h5file.root.cc._f_remove()
        ccDtype = np.dtype([('c1',np.int),('c2',np.int),('xi',np.int),
            ('yi',np.int),('ml',np.float)])
        ccTable = h5file.createTable("/", 'cc', ccDtype,
                "CC Table %s-%s %s-%s" % (c1I[0],c1I[1],c2I[0],c2I[1]))
        ny, nx = grid.shape
        c1colors = np.arange(c1.min(), c1.max()+0.05, 0.05)
        c2colors = np.arange(c2.min(), c2.max()+0.05, 0.05)
        print "g-i", c1colors
        print "i-Ks", c2colors
        for i in xrange(ny):
            for j in xrange(nx):
                ccTable.row['c1'] = c1colors[j]
                ccTable.row['c2'] = c2colors[i]
                ccTable.row['xi'] = i
                ccTable.row['yi'] = j
                ccTable.row['ml'] = grid[i,j]
                ccTable.row.append()
        h5file.flush()
        h5file.close()

def no_xticklabels(ax):
    """Removes tick marks from the x axis."""
    for label in ax.xaxis.get_majorticklabels():
        label.set_visible(False)

class MonteCarloLibrary2(MonteCarloLibrary):
    def _sample_zmet(self):
        """Returns a random metallicity"""
        return np.random.randint(1,23)
    
    def _sample_tau(self):
        """Returns a random e-folding of SFR"""
        return np.random.uniform(0.1, 10,)
    
    def _sample_const(self):
        """Returns fraction of mass formed as a constant mode of SF"""
        return np.random.uniform(0.1,0.5)
    
    def _sample_sf_start(self):
        """Start time of SFH in Gyr"""
        return np.random.uniform(0.,1.5)
    
    def _sample_fburst(self):
        """Fraction of mass formed in an instantaneous burst of SF."""
        return np.random.uniform(0.,0.5)
    
    def _sample_tburst(self):
        """Time of the burst after the BB."""
        return np.random.uniform(1.5,12.5)
    
    def _sample_dust2(self):
        """Optical depth of ambient ISM."""
        return np.random.uniform(0.1,1.)


if __name__ == '__main__':
    main()
