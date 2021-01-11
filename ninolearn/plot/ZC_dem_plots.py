#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:57:11 2021

@author: ivo
"""

from ninolearn.IO.read_raw import ZC_raw, ZC_h, ZC_oni
from ninolearn.preprocess.prepare import prep_nms

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ninolearn.pathes import processeddir, rawdir, plotdir
from ninolearn.preprocess.regrid import to2_5x2_5_ZC
from ninolearn.preprocess.network import networkMetricsSeries

from ninolearn.IO import read_raw
# from ninolearn.preprocess.anomaly import postprocess
# from ninolearn.preprocess.regrid import to2_5x2_5

from ninolearn.IO.read_processed import data_reader

from os.path import join

def nms_plots(version = 'default'):
    nmsdata = pd.read_csv(join(processeddir, ('network_metrics-sst_ZC_25x25_' + version + '_anom.csv')))
    
    c2 = nmsdata['fraction_clusters_size_2']
    c3 = nmsdata['fraction_clusters_size_3']
    
    H = nmsdata['hamming_distance']
    Hs = nmsdata['corrected_hamming_distance']
    t = pd.to_datetime(nmsdata['Unnamed: 0'])
    
    nmsdata_ERSST = pd.read_csv(join(processeddir, 'network_metrics-sst_ERSSTv5_anom.csv' ))
    c2_ERSST = nmsdata_ERSST['fraction_clusters_size_2']
    c3_ERSST = nmsdata_ERSST['fraction_clusters_size_3']
    H_ERSST = nmsdata_ERSST['hamming_distance']
    Hs_ERSST = nmsdata_ERSST['corrected_hamming_distance']
    t_ERSST = pd.to_datetime(nmsdata_ERSST['Unnamed: 0'])
    
    plt.plot(t, c2, label = r'$c_2$')
    plt.plot(t, H, label = 'H')
    plt.plot(t, Hs, label = r'$H^*$')
    plt.title('C2 fractions and Hamming distances')
    plt.legend()
    plt.show()
    
    fig, axs = plt.subplots(3, 2)
    fig.subplots_adjust(hspace=0.5)
    
    axs[0,0].plot(t_ERSST, c2_ERSST, color = 'g')
    axs[0,0].set_ylabel(r'$c_2$')
    axs[0,0].title.set_text(r'$C_2$ from ERRSTv5')
    
    axs[1,0].plot(t_ERSST, Hs_ERSST, color = 'g')
    axs[1,0].set_ylabel(r'$H^*$')
    axs[1,0].title.set_text(r'$H^*$ from ERRSTv5')

    axs[0,1].plot(t, c2, color = 'r')
    axs[0,0].set_ylabel(r'$c_2$')
    axs[0,1].title.set_text((r'$C_2$ from ZC87 data, version: ' + version))

    axs[1,1].plot(t, Hs, color = 'r')
    axs[1,0].set_ylabel(r'$H^*$')
    axs[1,1].title.set_text((r'$H^*$ from ZC87 data, version: ' + version))

    axs[2,0].plot(t_ERSST, c3_ERSST, color = 'g')
    axs[2,0].set_ylabel(r'$C_3$')
    axs[2,0].title.set_text(r'$C_3$ from ERSSTv5')
    
    axs[2,1].plot(t, c3, color = 'r')
    axs[2,1].title.set_text((r'$C_3$ from ZC87 data, version: ' + version))

    plt.savefig(join(plotdir, ('nms_' + version)))
    plt.show()