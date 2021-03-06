#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:51:07 2020

@author: ivo
"""

"""
TODO: calculate ONI 

"""

### NOTE: this script is also included under ninolearn/preprocess/prepare.py as
# the "prep_ZConi()" function

import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import numpy.ma as ma 

from ninolearn.pathes import processeddir
from ninolearn.IO.read_processed import data_reader


from os.path import join

makeplot = False
comparisonplot = False

filename = 'ZC_SST_undistorted'
suffix = '.nc'

data = xr.open_dataset(join(processeddir, 'ZC/' + filename + suffix))
SST = data['temperature']

# caclulate mean temp in NIN34 box from SST field to get timeseries
SST_NIN34_mly = np.mean( np.mean(SST.loc[:, -5:5, -170:-120], axis = 2), axis = 1).to_dataframe()

### calculate 30 yr climatology (1960-1990)

data_clim = SST_NIN34_mly.loc['1960-01-01':'1990-01-01']
months = range(1, 13) # Januari is 1, December is 12

climatology = np.zeros(13) # indexes match months so month 0 is empty
for month in months:
    month_data_clim = data_clim[data_clim.index.month == month]
    avg = np.mean(month_data_clim)
    climatology[month] = avg
    
anomalies = pd.DataFrame(np.zeros(SST_NIN34_mly.shape), index = SST_NIN34_mly.index, columns=['anomaly'])

### calculate anomalies
for month in months:
    SST_month = SST_NIN34_mly[SST_NIN34_mly.index.month == month]
    anom = SST_month - climatology[month]
    anomalies.loc[SST_month.index] = anom

SST_NIN34_mly['anom'] = anomalies
ONI = SST_NIN34_mly['anom'].rolling(window=3, center = True).mean().dropna()
df_ONI = pd.DataFrame(ONI, index = np.asarray(ONI.index))
df_ONI.index.names = ['time']

df_ONI.to_csv(join(processeddir, 'ZC_oni.csv'))
SST_NIN34_mly.to_csv(join(processeddir, 'SST_NIN34_anomaly' + filename + suffix))



##############################################################################




### plot ONI (sloppy implementation)
reader = data_reader(startdate='1950-02', enddate='2018-12')
oni_ERSSTv5 = reader.read_csv('oni')

plt.plot(ONI.index , ONI, color = 'black')

ma1 = ma.masked_array(ONI,  np.logical_not(ONI > 0.5))
plt.plot(ONI.index , ma1, color = 'red')

ma2 = ma.masked_array(ONI, np.logical_not((ONI < -0.5)))
plt.plot(ONI.index , ma2, color = 'blue')

plt.hlines([-0.5,0.5], xmin = ONI.index[0], xmax = ONI.index[-1], ls ='-.')
plt.xlim([ONI.index[0], ONI.index[-1]])

ma3 = ma.masked_array(ONI, np.logical_not( np.logical_and(-0.5 < ONI, ONI < 0.5)))       
plt.plot(ONI.index , ma3, color = 'black')

# plt.plot(oni_ERSSTv5.index, oni_ERSSTv5, color = 'g', ls = '--')

plt.ylabel('ONI')
plt.xlabel('ZC model time (arbitrary) [years]')
plt.savefig(join('/home/ivo/Documents/GitHub/ninolearn/research/ivo_thesis/plots/', 'ZC_oni_' + filename))

if makeplot == True:
    plt.show()

if comparisonplot is True:
    plt.plot(ONI.index , ONI, color = 'black', label = 'ZC')
    plt.plot(oni_ERSSTv5.index, oni_ERSSTv5, color = 'g', ls = '--', label = 'ERSSTv5')
    plt.legend()
    plt.ylabel('ONI')
    plt.xlabel('time [years]')
    
    plt.show()
    plt.savefig(join('/home/ivo/Documents/GitHub/ninolearn/research/ivo_thesis/plots/', 'oni_comparison' + filename))


    
    
    