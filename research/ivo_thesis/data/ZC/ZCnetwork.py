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

from ninolearn.IO import read_raw
from ninolearn.preprocess.anomaly import postprocess
from ninolearn.preprocess.regrid import to2_5x2_5


from os.path import join

data = xr.open_dataset(join(processeddir, 'sst_ZC_undistorted_anom.nc'))
data25x25 = to2_5x2_5_ZC(data)['temperature']


data25x25.to_netcdf(join(processeddir, 'sst_ZC_25x25_undistorted_anom.nc'))    



sst_ERSSTv5 = read_raw.sst_ERSSTv5()
sst_ERSSTv5_regrid = to2_5x2_5(sst_ERSSTv5)
postprocess(sst_ERSSTv5_regrid, new=True)

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