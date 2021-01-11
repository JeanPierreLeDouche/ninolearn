#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 11:27:49 2020

@author: ivo
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

### load ZC network metric results
nmsdata = pd.read_csv(join(processeddir, 'network_metrics-sst_ZC_25x25_undistorted_anom.csv'))

c2 = nmsdata['fraction_clusters_size_2']
H = nmsdata['hamming_distance']
Hs = nmsdata['corrected_hamming_distance']
t = pd.to_datetime(nmsdata['Unnamed: 0'])

### load ERSSTv5 network metric results
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

### plot ERSSTv5 and ZC results side by side

fig, axs = plt.subplots(2, 2)
fig.subplots_adjust(hspace=0.5)

axs[0,0].plot(t_ERSST, c2_ERSST, color = 'g')
axs[0,0].set_ylabel(r'$c_2$')
axs[0,0].set_title('ERSSTv5 data')

axs[1,0].set_title('ERSSTv5 data')
axs[1,0].plot(t_ERSST, Hs_ERSST, color = 'g')
axs[1,0].set_ylabel(r'$H^*$')

axs[0,1].plot(t, c2, color = 'r')
axs[0,1].set_ylabel(r'$c_2$')
axs[0,1].set_title('ZC87 data')

axs[1,1].plot(t, Hs, color = 'r')
axs[1,1].set_ylabel(r'$H^*$')
axs[1,1].set_title('ZC87 data')

print('note: ERSSTv5 dataset is about twice the lenght of ZC87')

plt.show()