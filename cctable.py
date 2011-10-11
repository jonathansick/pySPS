#!/usr/bin/env python
# encoding: utf-8
"""
Creates colour-colour look-up-tables, a la Zibetti et al (2009).

History
-------
2011-10-05 - Created by Jonathan Sick

"""
import pyfits
import numpy as np
import tables # pytables for HDF5

import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
    pass

class CCTable(object):
    """Generates colour-colour look up tables of SPS models.
    
    Parameters
    ----------

    modelTablePath : str
       path to an HDF5 document with a model output table.
    """
    def __init__(self, modelTablePath):
        super(CCTable, self).__init__()
        self.modelTablePath = modelTablePath
        self.h5 = tables.openFile(self.modelTablePath, mode='a')
        self.models = self.h5.root.models # points to the model data table

        # These must be assigned by either making the table, or choosing it
        # TODO allow an existing cc table to be chosen on init
        self.group = None
        self.cells = None
        self.membership = None
        self.xgrid = None
        self.ygrid = None

    def open(self, name):
        """Opens an existing colour-colour table stored in the HDF5 document."""
        self.name = name
        if name not in self.h5.root:
            assert "%s does not exist in %s " % (name, self.modelTablePath)
        self.group = getattr(self.h5.root, name)
        self.cells = getattr(self.group, "cells")
        self.membership = getattr(self.group, "membership")
        self.xgrid = getattr(self.group, "xgrid")
        self.ygrid = getattr(self.group, "ygrid")

    def make(self, name, xColourID, yColourID, binsize=0.05, clobber=False):
        """docstring for make"""
        self.name = name
        if name in self.h5.root and clobber:
            print "clobbering %s"%name, getattr(self.h5.root, name)
            getattr(self.h5.root, name)._f_remove(recursive=True)
        # Create a new node for this Colour-Colour grid table
        self.group = self.h5.createGroup("/", name, title='%s CC Table' % name)
        # Bin by colours
        xc = self._compute_colours(xColourID)
        yc = self._compute_colours(yColourID)
        _tbl, _membership, _xGrid, _yGrid = griddata(xc, yc, binsize=0.05)
        print "axis grids:", len(_xGrid), len(_yGrid)
        print _xGrid
        print _yGrid
        print "membership",
        print _membership
        print "cells"
        print _tbl
        # Add the table and membership/xgrid/ygrid arrays
        tblDtype = np.dtype([('yi',np.int),('xi',np.int),
            ('x',np.float),('y',np.float),('n',np.int)])
        self.cells = self.h5.createTable(self.group, 'cells', tblDtype, 'Cell Data')
        self.cells.append(_tbl)
        self.cells.flush()
        self.h5.flush()
        print "self.cells has rows:", self.cells.nrows
        print len(self.cells)
        # Add membership vector
        self.membership = self.h5.createArray(self.group, 'membership',
                _membership, "Membership of models in cells")
        # Add x and y axis grids
        self.xgrid = self.h5.createArray(self.group, 'xgrid', _xGrid, "xaxis")
        self.ygrid = self.h5.createArray(self.group, 'ygrid', _yGrid, 'yaxis')

    def _compute_colours(self, colourID):
        """Extract a colour from the model table."""
        if type(colourID) is not str:
            c = np.array([x[colourID[0]]-x[colourID[1]] for x in self.models])
            print colourID[0], np.array([x[colourID[0]] for x in self.models])
            print colourID[1], np.array([x[colourID[1]] for x in self.models])
            print colourID[0],colourID[1], c
        else:
            c = np.array([x[colourID] for x in self.models])
        return c

    def mass_light_table(self):
        """Adds a median log M/L_bol column to the table.
        
        This is essentially a specialized version of :meth:`median_grid`.
        """
        massArray = []
        LArray = []
        for x in self.models:
            massArray.append(x['mass'])
            LArray.append(x['lbol'])
        massArray = np.array(massArray) # log Mass
        LArray = np.array(LArray) # log L_bol
        modelML = massArray - LArray # log M/L
        medianMLs = np.zeros(self.cells.nrows, dtype=np.float)
        sigmaMLs = np.zeros(self.cells.nrows, dtype=np.float)
        for i in xrange(self.cells.nrows):
            print self.membership
            print np.where(self.membership[:] == i)[0]
            ind = np.where(self.membership[:] == i)[0]
            if len(ind) > 3:
                print "Well populated", len(ind), ind
                sample = modelML[ind]
                sample = sample[np.isfinite(sample)]
                medianML = np.median(sample)
                sigmaML = np.std(sample)
                medianMLs[i] = medianML
                sigmaMLs[i] = sigmaML
            else:
                medianMLs[i] = np.nan
                sigmaMLs[i] = np.nan
        self._append_cell_statistics("ML_bol", medianMLs, sigmaMLs)

    def median_grid(self, cname, log=False, medName=None):
        """Generates a median grid for a column-name `cname`.
        
        The median data for each cell in the colour-colour table is appended
        as a column in the `cells` table.
        
        Parameters
        ----------
        cname : str
           Name of the column in the model table with the parameter; typically
           this is also a name of an FSPS parameter.
        log : bool, optional
           If `True`, then the base-10 logarithm of the parameter will
           be taken before computing the median.
        medName : str, optional
           If set, this is the column name for the median values in the
           colour-colour LUT. Otherwise the parameter's name `cname` is used.
        """
        modelVals = np.array([x[cname] for x in self.models], dtype=np.float)
        medianVals = np.zeros(self.cells.nrows, dtype=np.float)
        sigmaVals = np.zeros(self.cells.nrows, dtype=np.float)
        for i in xrange(self.cells.nrows):
            ind = np.where(self.membership[:] == i)[0]
            if len(ind) > 3:
                print "Well populated", len(ind), ind
                sample = modelVals[ind]
                sample = sample[np.isfinite(sample)]
                if log:
                    sample = np.log10(sample)
                medianVals[i] = np.median(sample)
                sigmaVals[i] = np.std(sample)
            else:
                medianVals[i] = np.nan
                sigmaVals[i] = np.nan
        if medName is not None:
            self._append_cell_statistics(medName, medianVals, sigmaVals)
        else:
            self._append_cell_statistics(cname, medianVals, sigmaVals)
    
    def zsolar_grid(self, cname="Z", medName="logZsolar"):
        """Generates a median grid for log(Z/Zsolar). Assumes Zsolar=0.0190
        
        Essentially this is a specialized version of :meth:`median_grid`.'

        Parameters
        ----------
        cname : str, optional
           Name of the column in the model table with the parameter; typically
           this is also a name of an FSPS parameter.
        log : bool, optional
           If `True`, then the base-10 logarithm of the parameter will
           be taken before computing the median.
        medName : str, optional
           If set, this is the column name for the median values in the
           colour-colour LUT. Otherwise the parameter's name `cname` is used.
        """
        modelVals = np.array([x[cname] for x in self.models], dtype=np.float)
        modelLogSolarVals = np.log10(modelVals / 0.0190)
        medianVals = np.zeros(self.cells.nrows, dtype=np.float)
        sigmaVals = np.zeros(self.cells.nrows, dtype=np.float)
        for i in xrange(self.cells.nrows):
            ind = np.where(self.membership[:] == i)[0]
            if len(ind) > 3:
                print "Well populated", len(ind), ind
                sample = modelLogSolarVals[ind]
                sample = sample[np.isfinite(sample)]
                medianVals[i] = np.median(sample)
                sigmaVals[i] = np.std(sample)
            else:
                medianVals[i] = np.nan
                sigmaVals[i] = np.nan
        self._append_cell_statistics(medName, medianVals, sigmaVals)

    def gamma_grid(self, cname="tau", medName="gamma"):
        """Generates a median grid for gamma.

        gamma is used by the Bruzual and Charlot models to parameterize SFH.
        It is inversely related to the tau parameter of FSPS.

        Essentially this is a specialized version of :meth:`median_grid`.'

        Parameters
        ----------
        cname : str, optional
           Name of the column in the model table with the parameter; typically
           this is also a name of an FSPS parameter.
        medName : str, optional
           If set, this is the column name for the median values in the
           colour-colour LUT. Otherwise the parameter's name `cname` is used.
        """
        modelVals = np.array([x[cname] for x in self.models], dtype=np.float)
        gammaVals = 1/modelVals
        medianVals = np.zeros(self.cells.nrows, dtype=np.float)
        sigmaVals = np.zeros(self.cells.nrows, dtype=np.float)
        for i in xrange(self.cells.nrows):
            ind = np.where(self.membership[:] == i)[0]
            if len(ind) > 3:
                print "Well populated", len(ind), ind
                sample = gammaVals[ind]
                sample = sample[np.isfinite(sample)]
                medianVals[i] = np.median(sample)
                sigmaVals[i] = np.std(sample)
            else:
                medianVals[i] = np.nan
                sigmaVals[i] = np.nan
        self._append_cell_statistics(medName, medianVals, sigmaVals)

    def ml_x_grid(self, bandname, solarMag):
        """Generates a median grid for Mass/Light in a specific band. The
        M/L is stored in the table as ML_bandname

        Essentially this is a specialized version of :meth:`median_grid`.'

        Parameters
        ----------
        bandname : str
           Name of the band, this should be a column in the Models Table. For
           band names, see `fsps.FILTER_LIST`.
        solarMag : float
           The absolute solar magnitude in this band
        """
        massArray = []
        magsArray = []
        for x in self.models:
            massArray.append(x['mass'])
            magsArray.append(x[bandname])
        modelMass = np.array(massArray) # log Mass
        modelMags = np.array(magsArray)
        modelLLsolar = -0.4*(modelMags - solarMag) # log L
        modelML = modelMass - modelLLsolar
        medianVals = np.zeros(self.cells.nrows, dtype=np.float)
        sigmaVals = np.zeros(self.cells.nrows, dtype=np.float)
        for i in xrange(self.cells.nrows):
            ind = np.where(self.membership[:] == i)[0]
            if len(ind) > 3:
                print "Well populated", len(ind), ind
                sample = modelML[ind]
                sample = sample[np.isfinite(sample)]
                medianVals[i] = np.median(sample)
                sigmaVals[i] = np.std(sample)
            else:
                medianVals[i] = np.nan
                sigmaVals[i] = np.nan
        medName = "ML_"+bandname
        self._append_cell_statistics(medName, medianVals, sigmaVals)

    def _append_cell_statistics(self, name, medianArray, sigmaArray):
        """Appends colums to the cell table for the median value and its
        standard deviation"""
        stdName = name+"_std"
        # Get a description of table in dictionary format
        dtype = self.cells.description._v_colObjects
        dtype2 = dtype.copy()

        # Add a column to description
        dtype2[name] = tables.Float64Col(dflt=0.)
        dtype2[stdName] = tables.Float64Col(dflt=0.)

        # Create a new table with the new description
        table2 = self.h5.createTable(self.group, 'cells2', dtype2, "Cell Data")
        # Filters(1)

        # Copy the user attributes
        self.cells.attrs._f_copy(table2)

        # Fill the rows of new table with default values
        for i in xrange(self.cells.nrows):
            table2.row.append()
        # Flush the rows to disk
        table2.flush()

        # Copy the columns of source table to destination
        for col in dtype:
            getattr(table2.cols, col)[:] = getattr(self.cells.cols, col)[:]

        # Fill the new column
        getattr(table2.cols, name)[:] = medianArray
        getattr(table2.cols, stdName)[:] = sigmaArray

        # Remove the original table
        self.cells.remove()

        # Move table2 to table
        table2.move(self.group,'cells')

        self.h5.flush()
        self.open(self.name)

def griddata(x, y, binsize=0.01):
    """
    Place unevenly spaced 2D data on a grid by 2D binning (nearest
    neighbor interpolation).
    
    From http://www.scipy.org/Cookbook/Matplotlib/Gridding_irregularly_spaced_data
    
    Parameters
    ----------
    x : ndarray (1D)
        The idependent data x-axis of the grid.
    y : ndarray (1D)
        The idependent data y-axis of the grid.
    z : ndarray (1D)
    binsize : scalar, optional
        The full width and height of each bin on the grid.  If each
        bin is a cube, then this is the x and y dimension.  This is
        the step in both directions, x and y. Defaults to 0.01.
   
    Returns
    -------
    tbl : ndarray (1D)
       Record array specifying `xi`,`yi`,`x`,`y`,`n` for every bin cell
    membership : ndarray (1D)
       Indices into tbl for each point in data
    xGrid, yGrid: ndarray (1D)
       Arrays specifying x and y axis grids

    Revisions
    ---------
    2010-07-11  ccampo  Initial version
    2011-10-05  jonathansick Customized for use in pySPS
    """
    # get extrema values.
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    # make coordinate arrays.
    xGrid = np.arange(xmin, xmax+binsize, binsize)
    yGrid = np.arange(ymin, ymax+binsize, binsize)
    xi, yi = np.meshgrid(xGrid,yGrid)
    nrow, ncol = xi.shape

    # Array the same length as data specifying which cell each model belongs to
    membership = np.empty(len(x), dtype=np.int)

    # Tabular version of the grid
    tbl = np.empty(nrow*ncol, dtype=[('yi',np.int),('xi',np.int),
        ('x',np.float),('y',np.float),('n',np.int)])

    # fill in the grid.
    k = 0
    for row in range(nrow):
        for col in range(ncol):
            xc = xi[row, col]    # x coordinate.
            yc = yi[row, col]    # y coordinate.

            tbl[k]['yi'] = row
            tbl[k]['xi'] = col
            tbl[k]['y'] = yc
            tbl[k]['x'] = xc

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
            ind  = np.where(ibin == True)[0]

            tbl[k]['n'] = len(ind)
            membership[ind] = k

            k += 1

    return tbl, membership, xGrid, yGrid

class CCPlot(object):
    """Makes plots of colour-colour look up tables."""
    def __init__(self, ccTable, kind):
        super(CCPlot, self).__init__()
        self.ccTable = ccTable
        self.kind = kind # name fo column with median quantity

    def plot(self, plotPath, xLabel, yLabel, unitLabel,
            medMult=None, rmsMult=None, medLim=None, rmsLim=None):
        """General-purpose plot of median value, RMS and bin counts."""
        fig = plt.figure(figsize=(4,8))
        fig.subplots_adjust(left=0.1, bottom=0.07, right=0.85, top=0.95,
                wspace=None, hspace=None)
        axMed = fig.add_subplot(311)
        axRMS = fig.add_subplot(312)
        axN = fig.add_subplot(313)
        
        self.plot_cc_median(axMed, fig, xLabel, yLabel, unitLabel,
                zMult=medMult, zLim=medLim)
        self.plot_cc_rms(axRMS, fig, xLabel, yLabel, r"RMS(%s)" % unitLabel,
                zMult=rmsMult, zLim=rmsLim)
        self.plot_cc_count(axN, fig, xLabel, yLabel, r"$N$")

        axMed.set_xlabel("", visible=False)
        axRMS.set_xlabel("")
        for tickLabel in axMed.xaxis.get_majorticklabels(): tickLabel.set_visible(False)
        for tickLabel in axRMS.xaxis.get_majorticklabels(): tickLabel.set_visible(False)
        fig.savefig(plotPath+".pdf", format="pdf")

    def plot_cc_median(self, ax, fig, xLabel, yLabel, zLabel, zMult=None,zLim=None):
        """Plots the median value colour-colour grid in the provided axes."""
        x = self.ccTable.xgrid
        y = self.ccTable.ygrid
        print "nx", len(x), "ny", len(y)
        extent = [min(x),max(x),min(y),max(y)]
        print "extent:", extent
        grid = self._make_image(self.kind)
        if zLim is not None:
            vmin, vmax = zLim
        else:
            vmin, vmax = None, None
        im = ax.imshow(grid, cmap=mpl.cm.jet, extent=extent, vmin=vmin, vmax=vmax,
            interpolation='nearest', origin='lower', aspect='equal') # , aspect='equal'
        cbar = fig.colorbar(mappable=im, cax=None, ax=ax, orientation='vertical',
            fraction=0.1, pad=0.02, shrink=0.75,)
        cbar.set_label(zLabel)
        if zMult is not None:
            cbar.locator = mpl.ticker.MultipleLocator(base=zMult)
            cbar.update_ticks()
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)

    def plot_cc_rms(self, ax, fig, xLabel, yLabel, zLabel, zMult=None, zLim=None):
        """Plots the median value colour-colour grid in the provided axes."""
        x = self.ccTable.xgrid
        y = self.ccTable.ygrid
        print "nx", len(x), "ny", len(y)
        extent = [min(x),max(x),min(y),max(y)]
        print "extent:", extent
        grid = self._make_image(self.kind+"_std")
        if zLim is not None:
            vmin, vmax = zLim
        else:
            vmin, vmax = None, None
        im = ax.imshow(grid, cmap=mpl.cm.jet, extent=extent, vmin=vmin, vmax=vmax,
            interpolation='nearest', origin='lower',aspect='equal') # , aspect='equal'
        cbar = fig.colorbar(mappable=im, cax=None, ax=ax, orientation='vertical',
            fraction=0.1, pad=0.02, shrink=0.75,)
        cbar.set_label(zLabel)
        if zMult is not None:
            cbar.locator = mpl.ticker.MultipleLocator(base=zMult)
            cbar.update_ticks()
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)

    def plot_cc_count(self, ax, fig, xLabel, yLabel, zLabel):
        """Plots the number of models in the colour-colour grid in the provided axes."""
        x = self.ccTable.xgrid
        y = self.ccTable.ygrid
        print "nx", len(x), "ny", len(y)
        extent = [min(x),max(x),min(y),max(y)]
        print "extent:", extent
        grid = self._make_count_image()
        print "count shape", grid.shape
        print grid.min(), grid.max()
        grid = grid.astype(int)
        #grid[np.where(grid < 4)] = np.nan
        print grid.dtype
        im = ax.imshow(grid, cmap=mpl.cm.jet, extent=extent, vmin=0, vmax=grid.max(),
            interpolation='nearest', origin='lower',aspect='equal') # , aspect='equal'
        cbar = fig.colorbar(mappable=im, cax=None, ax=ax, orientation='vertical',
            fraction=0.1, pad=0.02, shrink=0.75,)
        cbar.set_label(zLabel)
        cbar.locator = mpl.ticker.MultipleLocator(base=100)
        cbar.update_ticks()
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)

    def _make_image(self, colName):
        """`colName` is the name of the column in the colour-colour table."""
        xi = np.array([x['xi'] for x in self.ccTable.cells.iterrows()], dtype=np.int)
        yi = np.array([x['yi'] for x in self.ccTable.cells.iterrows()], dtype=np.int)
        vals = np.array([x[colName] for x in self.ccTable.cells], dtype=np.float)
        print colName, min(vals), max(vals)
        nrows = yi.max() + 1
        ncols = xi.max() + 1
        grid = np.zeros([nrows,ncols], dtype=np.float)
        for x, y, v in zip(xi, yi, vals):
            grid[y,x] = v
        return grid
    
    def _make_count_image(self):
        """`colName` is the name of the column in the colour-colour table."""
        xi = np.array([x['xi'] for x in self.ccTable.cells.iterrows()], dtype=np.int)
        yi = np.array([x['yi'] for x in self.ccTable.cells.iterrows()], dtype=np.int)
        vals = np.array([x['n'] for x in self.ccTable.cells], dtype=np.int)
        print "cells counts", vals
        print min(vals), max(vals)
        nrows = yi.max() + 1
        ncols = xi.max() + 1
        grid = np.zeros([nrows,ncols], dtype=np.int)
        for x, y, v in zip(xi, yi, vals):
            grid[y,x] = v
        print grid
        print grid.shape
        print type(grid)
        #print "N", min(grid), max(grid)
        #grid[grid < 3] = np.nan
        return grid

    def hist(self, plotPath):
        vals = np.array([x[self.kind] for x in self.ccTable.cells], dtype=np.float)
        vals = np.nan_to_num(vals)
        fig = plt.figure(figsize=(4,3))
        ax = fig.add_subplot(111)
        ax.hist(vals, bins=100)
        fig.savefig(plotPath+".pdf", format="pdf")


if __name__ == '__main__':
    main()


