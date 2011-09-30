"""Basic Monte Carlo stellar population library."""

import numpy as np
import matplotlib.pyplot as plt
import bson

import fspsq
from fspsq import FSPSLibrary
from fspsq import ParameterSet

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
    library.plot_parameter_hists()

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
        axZ = ax.add_subplot(221)
        axTau = ax.add_subplot(222)
        axFburst = ax.add_subplot(223)
        axTburst = ax.add_subplot(224)

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
