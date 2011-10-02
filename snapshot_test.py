#!/usr/bin/env python
# encoding: utf-8
"""
Tests the iso-age library

History
-------
2011-10-02 - Created by Jonathan Sick

"""

__all__ = ['']
import os
import tables
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import snapshotlib
from fspsq import ParameterSet

from mclib import MagTableDef
from mclib import get_metallicity_LUT

from griddata import griddata

def main():
    #tauzGrid = TauZGrid("tauz", dbname='fsps')
    #tauzGrid.reset()
    #tauzGrid.generate_grid()
    #tauzGrid.compute_models(nThreads=2, maxN=10, clean=True)
    #tauzGrid.plot_color("tauzgrid", 'MegaCam_g', 'MegaCam_i')
    mclib = MonteCarloLibrary('mc.4')
    mclib.reset()
    mclib.define_samples(n=100)
    mclib.compute_models(nThreads=1, maxN=10, clean=True)


class TauZGrid(snapshotlib.SnapshotLibrary):
    """Build a grid of Zs and Taus. Observe each at the age of the universe."""
    def __init__(self, libname, dbname="fsps", age=13.7):
        super(TauZGrid, self).__init__(libname, dbname=dbname, age=age)

    def generate_grid(self):
        zmets = range(1,23,4)
        #zmets = [3, 5, 10, 15, 20]
        #taus = np.arange(50.,100.,0.5)
        #taus = [0.5,1., 5., 10.]
        taus = np.linspace(0.1,100,10.)
        for zmet in zmets:
            for tau in taus:
                pset = ParameterSet(None, sfh=1,
                        zmet=int(zmet), tau=float(tau), tage=13.7,
                        const=0.5)
                self.register_pset(pset)

    def plot_color(self, plotPath, c1Name, c2Name):
        """Plots colour vs tau for each metallicity."""
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        zmets = self.collection.distinct('pset.zmet')
        zmets.sort()
        print zmets
        for zmet in zmets:
            docs = self.collection.find({"pset.zmet":zmet})
            taus = []
            c1 = []
            c2 = []
            for doc in docs:
                taus.append(doc['pset']['tau'])
                c1.append(doc['obs'][c1Name])
                c2.append(doc['obs'][c2Name])
            taus = np.array(taus)
            sort = np.argsort(taus)
            taus = taus[sort]
            c1 = np.array(c1)
            c2 = np.array(c2)
            c1 = c1[sort]
            c2 = c2[sort]
            c = c1 - c2
            ax.plot(taus, c, '-')

        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel("%s - %s" % (c1Name,c2Name))
        fig.savefig(plotPath+".pdf", format='pdf')

class MonteCarloLibrary(snapshotlib.SnapshotLibrary):
    """docstring for MonteCarloLibrary"""
    def __init__(self, libname, dbname='fsps', age=13.7):
        super(MonteCarloLibrary, self).__init__(libname,dbname=dbname,age=age)
    
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
                dust2=float(self._sample_dust2()),
                tage=13.7
                )
            self.register_pset(pset)

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
        return np.random.uniform(0.5,3.)
    
    def _sample_fburst(self):
        """Fraction of mass formed in an instantaneous burst of SF."""
        return np.random.uniform(0.,0.5)
    
    def _sample_tburst(self):
        """Time of the burst after the BB."""
        return np.random.uniform(1.5,12.5)
    
    def _sample_dust2(self):
        """Optical depth of ambient ISM."""
        return np.random.uniform(0.1,1.)

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

if __name__ == '__main__':
    main()


