"""Test fspsq with a tiny grid of models--a basic burn in test case."""
from pymongo import ASCENDING
from fsps import FSPSLibrary
from fsps import ParameterSet

import matplotlib.pyplot as plt

def main():
    #tinyLibrary = TinySSPGrid("tinyssp", dbname="fsps")
    #tinyLibrary.reset()
    #tinyLibrary.generate_grid()
    #tinyLibrary.compute_models(nThreads=1)
    #tinyLibrary.create_table("tiny_table.h5")

    #plot_sfh(tinyLibrary)

    tinyTage = TinyTageGrid("tinytage", dbname="fsps")
    #tinyTage.reset()
    #tinyTage.generate_grid()
    #tinyTage.compute_models(nThreads=1)
    tinyTage.create_table("tiny_tage.h5")
    tinyTage.print_npdata()

class TinySSPGrid(FSPSLibrary):
    """A small grid SSPs for three metallicities."""
    
    def generate_grid(self):
        """Create the model grid."""
        taus = [0.1,1.,10.]
        for i, tau in enumerate(taus):
            pset = ParameterSet(None, sfh=1,
                    zmet=20,
                    tau=tau)
            self.register_pset(pset)

class TinyTageGrid(FSPSLibrary):
    """A small grid SSPs for three metallicities."""
    
    def generate_grid(self):
        """Create the model grid."""
        taus = [0.1,1.,10.]
        for i, tau in enumerate(taus):
            pset = ParameterSet(None, sfh=1, tage=13.7,
                    zmet=20,
                    tau=tau)
            self.register_pset(pset)

    def print_npdata(self):
        """Prints out columns from np_data for verification."""
        for doc in self.collection.find():
            print doc['_id']
            npData = doc['np_data']
            print npData['age'], npData['mass'], npData['lbol'], npData['sfr']


def plot_sfh_library(library):
    """Attempt to plot the star formation history using numpy arrays in
    MongoDB"""
    fig = plt.figure(figsize=(6,6))
    axMass = fig.add_subplot(611)
    axL = fig.add_subplot(612)
    axSFR = fig.add_subplot(613)
    axML = fig.add_subplot(614)
    axGI = fig.add_subplot(615)
    axiKs = fig.add_subplot(616)
    
    
    #doc = self.collection.find_one({"compute_complete":True,
    #    "np_data": {"$exists": 1}, "pset.tau": {"$lt":1.},
    #    "pset.fburst": {"$lt":0.5}})
    doc = library.collection.find_one({"compute_complete":True,
        "np_data": {"$exists": 1}})
    lines = []
    labels = []
    for doc in library.collection.find({"compute_complete":True,
        "np_data": {"$exists": 1}}): # , sort=('pset.tau',ASCENDING)
        npData = doc['np_data']

        keys = ['sf_start', 'tburst', 'fburst', 'const']
        for k in keys:
            print k, doc['pset'][k]

        logAge = npData['age']
        logL = npData['lbol']
        logMass = npData['mass']
        logSFR = npData['sfr']
        g = npData['MegaCam_g']
        i = npData['MegaCam_i']
        Ks = npData['TMASS_Ks']
        gi = g - i
        iKs = i - Ks
        ML = logMass - logL

        age = 10.**logAge / 10.**9

        axMass.plot(age, logMass, '-')
        axL.plot(age, logL, '-')
        lines.append(axSFR.plot(age, logSFR, '-'))
        labels.append(r"$\tau=%.1f$"%doc['pset']['tau'])
        axGI.plot(age, gi, '-')
        axiKs.plot(age, iKs, '-')
        axML.plot(age, ML, '-')

    no_xticklabels(axMass)
    no_xticklabels(axL)
    no_xticklabels(axSFR)
    no_xticklabels(axGI)
    no_xticklabels(axML)

    axMass.set_xlim(0.,13.7)
    axL.set_xlim(0.,13.7)
    axSFR.set_xlim(0.,13.7)
    axGI.set_xlim(0.,13.7)
    axML.set_xlim(0.,13.7)
    axiKs.set_xlim(0.,13.7)
    axML.set_ylim(-1.4,1.)
    axMass.set_ylim(-2,1.)
    axSFR.set_ylim(-12.,-7.)

    axSFR.legend(lines,labels, loc='right')

    axMass.set_ylabel(r"$\log_{10} M$")
    axL.set_ylabel(r"$\log_{10} L$")
    axSFR.set_ylabel(r"$\log_{10} \mathrm{SFR}$")
    axGI.set_ylabel(r"$g^\prime-i^\prime$")
    axiKs.set_ylabel(r"$i^\prime-K_s$")
    axML.set_ylabel(r"$\log_{10} M/L$")
    #axML.set_xlabel(r"$\log_{10} \mathrm{Age}$")
    axiKs.set_xlabel("Gyr after BB")

    fig.savefig("sfh.pdf", format="pdf")

def no_xticklabels(ax):
    """Removes tick marks from the x axis."""
    for label in ax.xaxis.get_majorticklabels():
        label.set_visible(False)

if __name__ == '__main__':
    main()
