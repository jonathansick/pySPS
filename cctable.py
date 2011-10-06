#!/usr/bin/env python
# encoding: utf-8
"""
Creates colour-colour look-up-tables, a la Zibetti et al (2009).

History
-------
2011-10-05 - Created by Jonathan Sick

"""

import numpy as np
import tables # pytables for HDF5

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

    def make(self, name, xColourID, yColourID, binsize=0.05, clobber=False):
        """docstring for make"""
        if name in self.h5.root and clobber:
            self.h5.root[name]._f_remove()
        # Create a new node for this Colour-Colour grid table
        self.group = self.h5.createGroup("/", name, title='%s CC Table' % name)
        # Bin by colours
        xc = self._compute_colours(xColourID)
        yc = self._compute_colours(yColourID)
        _tbl, _membership, _xGrid, _yGrid = griddata(xc, yc, binsize=0.05)
        # Add the table and membership/xgrid/ygrid arrays
        tblDtype = np.dtype([('yi',np.int),('xi',np.int),
            ('x',np.float),('y',np.float),('n',np.int)])
        self.cells = self.h5.createTable(self.group, 'cells', tblDtype, 'Cell Data')
        self.cells.append(_tbl)
        self.cells.flush()
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
        else:
            c = np.array([x[colourID] for x in self.models])
        return c

    def mass_light_table(self):
        """Adds a median M/L_bol column to the table."""
        modelML = np.array([x['mass']/x['lbol'] for x in self.models],
                dtype=np.float)
        medianMLs = np.zeros(self.cells.nrows, dtype=np.float)
        sigmaMLs = np.zeros(self.cells.nrows, dtype=np.float)
        for i in xrange(self.cells.nrows):
            ind = np.where(self.membership == i)[0]
            if len(ind) > 0:
                sample = modelML[ind]
                medianML = np.median(sample)
                sigmaML = np.std(sample)
                medianMLs[i] = medianML
                sigmaMLs[i] = sigmaML
            else:
                medianMLs[i] = np.nan
                sigmaMLs[i] = np.nan
        self._append_cell_statistics("ML_bol", medianMLs, sigmaMLs)

    def _append_cell_statistics(self, name, medianArray, sigmaArray):
        """Appends colums to the cell table for the median value and its
        standard deviation"""
        stdName = name+"_std"
        # Get a description of table in dictionary format
        dtype = self.cells.description._v_colObjects
        dtype2 = dtype.copy()

        # Add a column to description
        dtype[name] = tables.Float64Col(dflt=0.)
        dtype[stdName] = tables.Float64Col(dflt=0.)

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
        table2.cols[name][:] = medianArray
        table2.cols[stdName][:] = sigmaArray

        # Remove the original table
        self.cells.remove()

        # Move table2 to table
        table2.move(self.group,'cells')

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

            tbl['yi'] = row
            tbl['xi'] = col
            tbl['y'] = yc
            tbl['x'] = xc

            # find the position that xc and yc correspond to.
            posx = np.abs(x - xc)
            posy = np.abs(y - yc)
            ibin = np.logical_and(posx < binsize/2., posy < binsize/2.)
            ind  = np.where(ibin == True)[0]

            tbl['n'] = len(ind)
            membership[ind] = k

            k += 1

    return tbl, membership, xGrid, yGrid

if __name__ == '__main__':
    main()


