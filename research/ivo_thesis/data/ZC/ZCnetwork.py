# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 10:29:06 2020

@author: ivo
"""

"""
In order to calculate the network parameters the SST dataset needs to have the
same format as in the DEM notebook (it's NOAA standard or something)
"""

"""
TODO:
    1. calculate Hamming distance
    2. calculate c2

"""

import xarray as xr
import pandas as pd
import numpy as np

from ninolearn.pathes import processeddir, rawdir
from ninolearn.preprocess.regrid import to2_5x2_5_ZC
from ninolearn.preprocess.network import networkMetricsSeries

from os.path import join

data = xr.open_dataset(join(processeddir, 'sst_ZC_undistorted_anom.nc'))
data25x25 = to2_5x2_5_ZC(data)
data25x25.to_netcdf(join(processeddir, 'sst_ZC_25x25_undistorted_anom.nc'))    

lons = data25x25.lon
newlon = np.zeros(lons.shape[0])
for i in range(lons.shape[0]):
    if lons[i] < 0: 
        new = lons[i] + 360
        newlon[i] = new
    else:
        newlon[i] = lons[i]

### TODO: consider changing to lon [0:360] coordinate system in preprocessing

#%%

# settings for the computation of the network metrics time series
nms = networkMetricsSeries('sst', 'ZC_25x25_undistorted', processed="anom",
                            threshold=0.97, startyear=1951, endyear=1994,
                            window_size=12, lon_min=124, lon_max=-80,
                            lat_min=-19, lat_max=19, verbose=2)

# compute the time series
nms.computeTimeSeries()

# save the time series again with a name following the naming convention
nms.save()