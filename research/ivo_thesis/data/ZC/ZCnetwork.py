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

from ninolearn.pathes import processeddir, rawdir
from ninolearn.preprocess.regrid import to2_5x2_5_ZC
from ninolearn.preprocess.network import networkMetricsSeries

from os.path import join

data = xr.open_dataset(join(processeddir, 'ZC/ZC_SST_undistorted.nc'))
data25x25 = to2_5x2_5_ZC(data)


##### FROM PP: 
    

# # settings for the computation of the network metrics time series
# # TODO: this should instead use the artificial SST timeseries
# nms = networkMetricsSeries('sst', 'ERSSTv5', processed="anom",
#                            threshold=0.97, startyear=1949, endyear=2018,
#                            window_size=12, lon_min=120, lon_max=280,
#                            lat_min=-30, lat_max=30, verbose=2)

# # compute the time series
# nms.computeTimeSeries()

# # save the time series again with a name following the naming convention
# nms.save()