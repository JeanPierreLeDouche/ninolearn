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
import matplotlib.pyplot as plt

from ninolearn.pathes import processeddir, rawdir
from ninolearn.preprocess.regrid import to2_5x2_5_ZC
from ninolearn.preprocess.network import networkMetricsSeries

from ninolearn.IO import read_raw
from ninolearn.preprocess.anomaly import postprocess
from ninolearn.preprocess.regrid import to2_5x2_5

from ninolearn.IO.read_processed import data_reader

from os.path import join

### opening datasets and processing them 

data = xr.open_dataset(join(processeddir, 'sst_ZC_undistorted_anom.nc'))

data25x25 = to2_5x2_5_ZC(data)['temperature']
data25x25 = data25x25.rename('sstAnom')
data25x25 = data25x25.transpose()

data25x25.to_netcdf(join(processeddir, 'sst_ZC_25x25_undistorted_anom.nc'))   

# settings for the computation of the network metrics time series
nms = networkMetricsSeries('sst', 'ZC_25x25_undistorted', processed="anom",
                            threshold=0.97, startyear=1951, endyear=1993,
                            window_size=12, lon_min=124, lon_max=280,
                            lat_min=-19, lat_max=19, verbose=2)

# compute the time series
nms.computeTimeSeries()

# save the time series again with a name following the naming convention
nms.save()

nmsdata = pd.read_csv(join(processeddir, 'network_metrics-sst_ZC_25x25_undistorted_anom.csv'))

#%%

c2 = nmsdata['fraction_clusters_size_2']
H = nmsdata['hamming_distance']
Hs = nmsdata['corrected_hamming_distance']
t = pd.to_datetime(nmsdata['Unnamed: 0'])

#%%

nmsdata_ERSST = pd.read_csv(join(processeddir, 'network_metrics-sst_ERSSTv5_anom.csv' ))
c2_ERSST = nmsdata_ERSST['fraction_clusters_size_2']
H_ERSST = nmsdata_ERSST['hamming_distance']
Hs_ERSST = nmsdata_ERSST['corrected_hamming_distance']
t_ERSST = pd.to_datetime(nmsdata_ERSST['Unnamed: 0'])

# plt.plot(t, c2, label = r'$c_2$')
# plt.plot(t, H, label = 'H')
# plt.plot(t, Hs, label = r'$H^*$')
# plt.title('C2 fractions and Hamming distances')
# plt.legend()
# plt.show()

fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.5)

axs[0,0].plot(t_ERSST, c2_ERSST)
axs[0,0].set_ylabel(r'$c_2$')

axs[1,0].plot(t_ERSST, Hs_ERSST)
axs[1,0].set_ylabel(r'$H^*$')

axs[0,1].plot(t, c2)
axs[0,0].set_ylabel(r'$c_2$')

axs[1,1].plot(t, Hs)
axs[1,0].set_ylabel(r'$H^*$')

plt.show()