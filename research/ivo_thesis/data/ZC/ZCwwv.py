#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:52:51 2020

@author: ivo
"""

"""
TODO:
    1. figure out how to calculate wwv from thermocline depth as proxy
    2. calculate wwv from h 


"""

from os.path import join
from ninolearn.pathes import processeddir, rawdir

from os.path import join, exists
import xarray as xr
import xesmf as xe
import pandas as pd
import matplotlib.pyplot as plt

# from ninolearn.pathes import processeddir
# from ninolearn.utils import generateFileName, small_print_header
# from ninolearn.preprocess.anomaly import _get_period, computeMeanClimatology
# from ninolearn.preprocess.prepare import
from ninolearn.IO.read_processed import data_reader
# from ninolearn.IO.read_raw import oni
# from ninolearn.IO.read_raw import nino34detrend_anom
from ninolearn.download import download, sources
from ninolearn.pathes import processeddir, rawdir
from ninolearn.preprocess.anomaly import computeAnomaly, saveAnomaly

# import pandas as pd
import numpy as np
# import xarray as xr
import xesmf as xe
import math
version = 'undistorted'

h = xr.load_dataset(join(processeddir, 'h_ZC_undistorted.nc' ))

### NOTE: 'name' and 'dataset' attrs are metadata and currently not automatically 
### preserved under operations.
### TODO: update ninolearn's handling of metadata (see how Paul does this first)

h = h.assign_attrs(name = 'sst')
h = h.assign_attrs(dataset = 'ZC_'+version)

anom = computeAnomaly(h)
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-19, 20, 1)),
                      'lon': (['lon'], np.arange(124, 281, 1)), })
                    
regridder = xe.Regridder(h, ds_out, 'bilinear') # from long decimal number latlon to 1x1

h = regridder(h)
### WWV is typically calculated for 5S-5N and 120E-280E ( technically 100W ofc)
h_dataset = h['thermocline_height'].loc[:, -5:5, :]
h_means = np.mean( np.mean(h_dataset, axis =2), axis =1)
df_h = pd.DataFrame(h_means, index = np.asarray(h_means.time), columns = ['anom'])

df_h.to_csv(join(processeddir, 'h.csv' ))

### not sure if h is anomaly or value
# anom = regridder(anom)
# h['anom'] = anom['thermocline_height']



