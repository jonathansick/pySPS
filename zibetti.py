#!/usr/bin/env python
# encoding: utf-8
"""
This module attempts to replicate the Zibetti (2009) recipe for creating
a colour-colour M/L look up table. Of course, rather than use Charlot
& Bruzual's SPS code, the FSPS engine of Conroy, Gunn and White (2009) is used.

History
-------
2011-10-09 - Created by Jonathan Sick

"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from fsps import FSPSLibrary
from fsps import ParameterSet
import cctable

TUNIVERSE = 13.7 # age of universe supposed in Gyr

def main():
    libname = "zibetti2"
    h5name = libname+".h5"
    define_library = False
    compute_models = False
    make_table = False
    make_cc = True
    plot_ml = True
    plot_scatter = True

    library = Zibetti2Library(libname)
    if define_library:
        library.reset()
        library.define_samples()
    if compute_models:
        library.compute_models(nThreads=8, maxN=100)
    if make_table:
        library.create_table(h5name, clobber=True)
    if make_cc:
        ccTable = cctable.CCTable(h5name)
        ccTable.make("megacam_gi_iK", ("MegaCam_i","TMASS_Ks"),
           ("MegaCam_g","MegaCam_i"), binsize=0.05, clobber=True)
        ccTable = cctable.CCTable(h5name)
        ccTable.open("megacam_gi_iK")
        ccTable.mass_light_table()
        ccTable.median_grid("tau")
        ccTable.median_grid("tau", log=True, medName="logTau")
        ccTable.median_grid("Z", log=True, medName="logZ")
        ccTable.median_grid("dust2")
        ccTable.median_grid("sf_start")
        ccTable.median_grid("tburst")
        ccTable.median_grid("fburst")
        ccTable.zsolar_grid()
        ccTable.gamma_grid()
        ccTable.ml_x_grid("TMASS_Ks", 3.28)
    if plot_ml:
        xlabel = r"$i^\prime-K_s$"
        ylabel = r"$g^\prime-i^\prime$"

        ccTable = cctable.CCTable(h5name)
        ccTable.open("megacam_gi_iK")

        plot = cctable.CCPlot(ccTable, "ML_bol")
        plot.plot(libname+"_grid_ml", xlabel, ylabel, r"$\log \Upsilon_\mathrm{bol}$",
                medMult=0.2, rmsLim=None, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "tau")
        plot.plot(libname+"_grid_tau", xlabel, ylabel, r"$\tau$ Gyr",
                rmsMult=5, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "logTau")
        plot.plot(libname+"_grid_log_tau", xlabel, ylabel, r"$\log \tau$ Gyr",
                medMult=0.2, rmsMult=0.1, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "sf_start")
        plot.plot(libname+"_grid_tform", xlabel, ylabel, r"$t_\mathrm{start}$ Gyr",
                medMult=2, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "tburst")
        plot.plot(libname+"_grid_tburst", xlabel, ylabel, r"$t_\mathrm{burst}$ Gyr",
                medMult=2, rmsMult=0.5, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "fburst")
        plot.plot(libname+"_grid_fburst", xlabel, ylabel, r"$f_\mathrm{burst}$",
                medMult=0.1, rmsMult=0.05, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "logZ")
        plot.plot(libname+"_grid_logZ", xlabel, ylabel, r"$\log Z$",
                medMult=0.5, rmsMult=0.2, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "dust2")
        plot.plot(libname+"_grid_dust2", xlabel, ylabel, r"$\mathrm{dust}_2$",
                medMult=0.1, rmsMult=0.05, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "gamma")
        plot.plot(libname+"_grid_gamma", xlabel, ylabel, r"$\gamma~\mathrm{Gyr}^{-1}$",
                medMult=None, rmsMult=None, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "logZsolar")
        plot.plot(libname+"_grid_zsolar", xlabel, ylabel, r"$\log Z/Z_\odot$",
                medMult=None, rmsMult=None, xLim=(-0.5,2.))

        plot = cctable.CCPlot(ccTable, "ML_TMASS_Ks")
        plot.plot(libname+"_ml_Ks", xlabel, ylabel, r"$\log \Upsilon_{K_s}$",
                medMult=0.1, xLim=(-0.5,2.))
    if plot_scatter:
        ccTable = cctable.CCTable(h5name)
        ccTable.open("megacam_gi_iK")
        diagnostic_scatter(libname+"_diagnostics", ccTable)


class ZibettiLibrary(FSPSLibrary):
    """A Monte Carlo stellar population library designed around the Zibetti
    (2009) priors for SFH and dust. The exact forms of the probability
    distribution functions are best specified in da Cunha, Charlot and Elbaz
    (2008).
    """
    def define_samples(self, n=50000):
        """Define the set of models."""
        for i in xrange(n):
            tstart = float(self._sample_sf_start())
            pset = ParameterSet(None, # automatically create a name
                sfh=1, # tau SFH
                tage=TUNIVERSE, # select only modern-day observations
                imf_type=1, # Chabrier 2003
                dust_type=2, # Calzetti 2000 attenuation curve
                zmet=int(self._sample_zmet()),
                tau=float(self._sample_tau()),
                const=float(0.), # no constant SF component
                sf_start=tstart,
                fburst=float(self._sample_fburst()),
                tburst=float(self._sample_tburst(tstart)),
                dust1=float(self._sample_dust1()),
                dust2=float(self._sample_dust2()),
                )
            self.register_pset(pset)
    
    def _sample_zmet(self):
        """Returns a random metallicity from the Padova catalog."""
        return np.random.randint(1,23)
    
    def _sample_tau(self):
        """Power law timescale of SFH, :math:`SFR(t) \propto exp(-t/\tau)`.

        Note that users of the Charlot and Bruzual use a :math:`\gamma`
        parameter, where :math:`\gamma \equiv 1/\tau`.
        
        da Cunha (2008) uses a p.d.f. of

        .. math:: p(\gamma) = 1-\tanh (8\gamma - 6)

        Here we generate that pdf using von Neumann's aceptance-rejection
        technique.

        Formally, da Cunha samples for :math:`\gamma=0\ldots1`, but the code
        checks to ensure that tau is between 0.1 and 100 Gyr.
        """
        u = 3
        x = 0
        while u >= 1. - np.tanh(8.*x - 6):
            x = np.random.uniform(0.,1.)
            u = np.random.uniform(0.,2.)
        tau = 1. / x
        if tau < 0.1:
            tau = 0.1
        elif tau > 100.:
            tau = 100
        return tau

    def _sample_sf_start(self):
        """Start of star-formation (Gyr. Defined in Kauffmann 2003."""
        return np.random.uniform(0.1, TUNIVERSE-1.5)

    def _sample_tburst(self, tform):
        """Time when the star-burst happens. We take the start of
        star-formation as an input so that there *can* always be a burst.

        This mathematical form of this prior used by Kauffmann et al is
        poorly specified. I quote:

        .. Bursts occur with equal probability at all times after tform and
           we have set the probability so that 50 per cent of the galaxies in
           the library have experienced a burst over the past 2 Gyr.

        The first part of that sentence does not necessarily imply the other.
        Regardless, I simply use a prior that bursts can happend uniformly
        between the start of star foramtion and the modern day.
        """
        return np.random.uniform(tform, TUNIVERSE)

    def _sample_fburst(self):
        """The fraction of stellar mass formed in a burst mode. Kauffmann
        logarithmically sample between 0 and 0.75.
        """
        return np.random.uniform(0., 0.75) # we're cheap and use uniform

    def _sample_dust1(self):
        """Sample the attenuation of young stellar light.
        
        da Cunha (2008) uses a p.d.f. of

        .. math:: p(\tau_V) = 1-\tanh (1.5\tau_V - 6.7)

        Here we generate that pdf using von Neumann's aceptance-rejection
        technique.
        """
        u = 3
        x = 0
        while u >= 1. - np.tanh(1.5*x - 6.7):
            x = np.random.uniform(0.,6.)
            u = np.random.uniform(0.,2.)
        return x

    def _sample_dust2(self):
        """Sample the attenuation due to the ambient ISM.
        
        da Cunha (2008) uses a p.d.f. of

        .. math:: p(\mu) = 1-\tanh (8 \mu - 6)

        Here we generate that pdf using von Neumann's aceptance-rejection
        technique.
        """
        u = 3
        x = 0
        while u >= 1. - np.tanh(8.*x - 6.):
            x = np.random.uniform(0.,6.)
            u = np.random.uniform(0.,2.)
        return x

class Zibetti2Library(FSPSLibrary):
    """A Monte Carlo stellar population library designed around the Zibetti
    (2009) priors for SFH and dust. The exact forms of the probability
    distribution functions are best specified in da Cunha, Charlot and Elbaz
    (2008).

    This is the same as the ZibettiLibrary, except that we use a Milky Way dust
    model, with a power law (-0.7) dust model, rather than a Calzetti model.
    """
    def define_samples(self, n=50000):
        """Define the set of models."""
        for i in xrange(n):
            tstart = float(self._sample_sf_start())
            pset = ParameterSet(None, # automatically create a name
                sfh=1, # tau SFH
                tage=TUNIVERSE, # select only modern-day observations
                imf_type=1, # Chabrier 2003
                dust_type=0, # power-law dust model
                zmet=int(self._sample_zmet()),
                tau=float(self._sample_tau()),
                const=float(0.), # no constant SF component
                sf_start=tstart,
                fburst=float(self._sample_fburst()),
                tburst=float(self._sample_tburst(tstart)),
                dust1=float(self._sample_dust1()),
                dust2=float(self._sample_dust2()),
                dust_index=-0.7
                )
            self.register_pset(pset)
    
    def _sample_zmet(self):
        """Returns a random metallicity from the Padova catalog."""
        return np.random.randint(1,23)
    
    def _sample_tau(self):
        """Power law timescale of SFH, :math:`SFR(t) \propto exp(-t/\tau)`.

        Note that users of the Charlot and Bruzual use a :math:`\gamma`
        parameter, where :math:`\gamma \equiv 1/\tau`.
        
        da Cunha (2008) uses a p.d.f. of

        .. math:: p(\gamma) = 1-\tanh (8\gamma - 6)

        Here we generate that pdf using von Neumann's aceptance-rejection
        technique.

        Formally, da Cunha samples for :math:`\gamma=0\ldots1`, but the code
        checks to ensure that tau is between 0.1 and 100 Gyr.
        """
        u = 3
        x = 0
        while u >= 1. - np.tanh(8.*x - 6):
            x = np.random.uniform(0.,1.)
            u = np.random.uniform(0.,2.)
        tau = 1. / x
        if tau < 0.1:
            tau = 0.1
        elif tau > 100.:
            tau = 100
        return tau

    def _sample_sf_start(self):
        """Start of star-formation (Gyr. Defined in Kauffmann 2003."""
        return np.random.uniform(0.1, TUNIVERSE-1.5)

    def _sample_tburst(self, tform):
        """Time when the star-burst happens. We take the start of
        star-formation as an input so that there *can* always be a burst.

        This mathematical form of this prior used by Kauffmann et al is
        poorly specified. I quote:

        .. Bursts occur with equal probability at all times after tform and
           we have set the probability so that 50 per cent of the galaxies in
           the library have experienced a burst over the past 2 Gyr.

        The first part of that sentence does not necessarily imply the other.
        Regardless, I simply use a prior that bursts can happend uniformly
        between the start of star foramtion and the modern day.
        """
        return np.random.uniform(tform, TUNIVERSE)

    def _sample_fburst(self):
        """The fraction of stellar mass formed in a burst mode. Kauffmann
        logarithmically sample between 0 and 0.75.
        """
        return np.random.uniform(0., 0.75) # we're cheap and use uniform

    def _sample_dust1(self):
        """Sample the attenuation of young stellar light.
        
        da Cunha (2008) uses a p.d.f. of

        .. math:: p(\tau_V) = 1-\tanh (1.5\tau_V - 6.7)

        Here we generate that pdf using von Neumann's aceptance-rejection
        technique.
        """
        u = 3
        x = 0
        while u >= 1. - np.tanh(1.5*x - 6.7):
            x = np.random.uniform(0.,6.)
            u = np.random.uniform(0.,2.)
        return x

    def _sample_dust2(self):
        """Sample the attenuation due to the ambient ISM.
        
        da Cunha (2008) uses a p.d.f. of

        .. math:: p(\mu) = 1-\tanh (8 \mu - 6)

        Here we generate that pdf using von Neumann's aceptance-rejection
        technique.
        """
        u = 3
        x = 0
        while u >= 1. - np.tanh(8.*x - 6.):
            x = np.random.uniform(0.,6.)
            u = np.random.uniform(0.,2.)
        return x

def diagnostic_scatter(plotPath, ccTable):
    """Looks at the covariences of various parameters with M/L using
    scatterplots.
    """
    fig = plt.figure(figsize=(6,8))
    fig.subplots_adjust(left=0.15, bottom=0.1, right=0.95, top=0.95,
                wspace=0.35, hspace=0.1)

    ax_gamma = fig.add_subplot(421)
    ax_tform = fig.add_subplot(422)
    ax_tburst = fig.add_subplot(423)
    ax_fburst = fig.add_subplot(424)
    ax_dust1 = fig.add_subplot(425)
    ax_dust2 = fig.add_subplot(426)
    ax_z = fig.add_subplot(427)

    plot_scatter_gamma(ax_gamma, ccTable, r"$\gamma~\mathrm{Gyr}^{-1}$")
    plot_scatter(ax_tform, ccTable, 'sf_start', r"$t_\mathrm{start}$ Gyr")
    plot_scatter(ax_tburst, ccTable, 'tburst', r"$t_\mathrm{burst}$ Gyr")
    plot_scatter(ax_fburst, ccTable, 'fburst', r"$f_\mathrm{burst}$")
    plot_scatter(ax_dust1, ccTable, 'dust1', r"$\mathrm{dust}_1$")
    plot_scatter(ax_dust2, ccTable, 'dust2', r"$\mathrm{dust}_2$")
    plot_scatter_zsolar(ax_z, ccTable, r"$\log Z/Z_\odot$")
    ax_z.set_xlabel(r"$\log \Upsilon_{K_s}$")
    ax_dust2.set_xlabel(r"$\log \Upsilon_{K_s}$")

    for l in ax_gamma.xaxis.get_majorticklabels(): l.set_visible(False)
    for l in ax_tform.xaxis.get_majorticklabels(): l.set_visible(False)
    for l in ax_fburst.xaxis.get_majorticklabels(): l.set_visible(False)
    for l in ax_tburst.xaxis.get_majorticklabels(): l.set_visible(False)
    for l in ax_dust1.xaxis.get_majorticklabels(): l.set_visible(False)

    fig.savefig(plotPath+".png", format="png")
    #fig.savefig(plotPath+".pdf", format="pdf")

def plot_scatter(ax, ccTable, cname, clabel):
    modelVals = []
    modelMass = []
    modelMag = []
    for x in ccTable.models:
        modelVals.append(x[cname])
        modelMass.append(x['mass'])
        modelMag.append(x['TMASS_Ks'])
    modelVals = np.array(modelVals)
    modelMass = np.array(modelMass) # logMass
    modelMag = np.array(modelMag)
    modelL = -0.4*(modelMag-3.28) # logL
    modelLogML = modelMass - modelL
    good = np.where(np.isfinite(modelVals) & np.isfinite(modelLogML))[0]
    modelVals = modelVals[good]
    modelLogML = modelLogML[good]

    grid, extent = grid_hist(modelLogML, modelVals, binsize=0.05)
    ax.imshow(grid, cmap=mpl.cm.jet, extent=extent, #, interpolation='nearest',
            origin='lower', aspect='auto')
    #ax.scatter(modelLogML, modelVals, c='k', marker='o', s=0.1)
    ax.set_ylabel(clabel)

def plot_scatter_gamma(ax, ccTable, clabel):
    modelTaus = []
    modelMass = []
    modelMag = []
    for x in ccTable.models:
        modelTaus.append(x['tau'])
        modelMass.append(x['mass'])
        modelMag.append(x['TMASS_Ks'])
    modelTaus = np.array(modelTaus)
    modelGamma = 1. / modelTaus
    modelMass = np.array(modelMass) # logMass
    modelMag = np.array(modelMag)
    modelL = -0.4*(modelMag-3.28) # logL
    modelLogML = modelMass - modelL
    good = np.where(np.isfinite(modelTaus) & np.isfinite(modelLogML))[0]

    grid, extent = grid_hist(modelLogML, modelGamma, binsize=0.05)
    ax.imshow(grid, cmap=mpl.cm.jet, extent=extent, #, interpolation='nearest',
            origin='lower', aspect='auto')
    #ax.scatter(modelLogML[good], modelGamma[good], c='k', marker='o', s=0.1)
    ax.set_ylabel(clabel)

def plot_scatter_zsolar(ax, ccTable, clabel):
    modelZ = []
    modelMass = []
    modelMag = []
    for x in ccTable.models:
        modelZ.append(x['Z'])
        modelMass.append(x['mass'])
        modelMag.append(x['TMASS_Ks'])
    modelZ = np.array(modelZ)
    modelZsolar = np.log10(modelZ / 0.019)
    modelMass = np.array(modelMass) # logMass
    modelMag = np.array(modelMag)
    modelL = -0.4*(modelMag-3.28) # logL
    modelLogML = modelMass - modelL
    good = np.where(np.isfinite(modelZsolar) & np.isfinite(modelLogML))[0]

    grid, extent = grid_hist(modelLogML, modelZsolar, binsize=0.05)
    ax.imshow(grid, cmap=mpl.cm.jet, extent=extent, #, interpolation='nearest'
            origin='lower', aspect='auto')
    #ax.scatter(modelLogML[good], modelZsolar[good], c='k', marker='o', s=0.1)
    ax.set_ylabel(clabel)

def grid_hist(x, y, binsize=0.05):
    tbl, members, xGrid, yGrid = cctable.griddata(x, y, binsize=binsize)
    print xGrid, yGrid
    print tbl['xi']
    print tbl['yi']
    ncols = tbl['xi'].max() + 1
    nrows = tbl['yi'].max() + 1
    grid = np.zeros([nrows,ncols], dtype=np.float)
    for i in xrange(len(tbl)):
        x = tbl['xi'][i]
        y = tbl['yi'][i]
        v = tbl['n'][i]
        if v > 0:
            grid[y,x] = v
        else:
            grid[y,x] = np.nan
    extent = [xGrid.min(), xGrid.max(), yGrid.min(), yGrid.max()]
    return grid, extent

if __name__ == '__main__':
    main()


