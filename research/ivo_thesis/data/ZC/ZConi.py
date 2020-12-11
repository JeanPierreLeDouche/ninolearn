#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:51:07 2020

@author: ivo
"""

"""
TODO: calculate ONI 

"""

import xarray as xr
import pandas as pd
import numpy as np

from ninolearn.pathes import processeddir, rawdir
from ninolearn.preprocess.regrid import to2_5x2_5_ZC
from ninolearn.preprocess.network import networkMetricsSeries
from ninolearn.utils import find_indexes_from_latlon

from os.path import join

data = xr.open_dataset(join(processeddir, 'ZC/ZC_SST_undistorted.nc'))
SST = data['temperature']

# calculate indices of the data belonging 
SST_NIN34_monthly = np.mean( np.mean(SST.loc[:, -5:5, -170:-120], axis = 2), axis = 1).to_dataframe()

### calculate 30 yr climatology (1960-1990)
data_clim = SST_NIN34_monthly.loc['1960-01-01':'1990-01-01']
months = range(1, 13)

climatology = np.zeros(13) # indexes match months so month 0 is empty
for month in months:
    month_data_clim = data_clim[data_clim.index.month == month]
    avg = np.mean(month_data_clim)
    climatology[month] = avg
    
anomalies = pd.DataFrame(np.zeros(SST_NIN34_monthly.shape), index = SST_NIN34_monthly.index, columns=['anomaly'])

#calculate anomalies
for month in months:
    pass
