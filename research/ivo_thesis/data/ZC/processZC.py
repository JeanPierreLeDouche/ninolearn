#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:34:03 2020

@author: ivo
"""

from os.path import join
import xarray as xr
import pandas as pd
import numpy as np
import xesmf as xe

from ninolearn.pathes import processeddir, rawdir
from ninolearn.utils import find_lat_from_dist, find_lon_from_dist
from ninolearn.preprocess.anomaly import computeMeanClimatology, computeAnomaly

r_e = 6371 * 1e3 # m # radius of the earth

###---------------------------------------------------------------------------
####################### specify version !: ##################################
###--------------------------------------------------------------------------

version = 'undistorted'

#
#
#

data = pd.read_csv(join(rawdir,  f'ZCraw_44yrs_{version}'), index_col=0)
data['time'] = pd.to_datetime(data['time'], unit = 's') + pd.DateOffset(years = -20)    

yvals, xvals, tvals = np.unique(data['y']), np.unique(data['x']), np.unique(data['time'])

"""
make grids in the funky ZC cartesian system of the h and T (add if needed)

"""

ZCgridT = np.zeros(( xvals.shape[0], yvals.shape[0], tvals.shape[0]))
ZCgridh = np.zeros(( xvals.shape[0], yvals.shape[0], tvals.shape[0]))

for t in enumerate(tvals):
    data_t = data[data['time'] == t[1]]
    
    for y in enumerate(yvals):
        data_y = data_t[data_t['y'] == y[1]]   
        
        ZCgridT[:,y[0],t[0]] = data_y['T']
        ZCgridh[:,y[0],t[0]] = data_y['h']
        
""" 
make lat lon grids out of the ZC cartesian system 
"""

lons_ZC = np.zeros(30)
lats_ZC = np.zeros(31)

lons_ZC[0] = 124
lats_ZC[0] = -29

#lons first
for x_i in range(xvals.shape[0]-1):
    new_lon, _ = find_lon_from_dist(124 , 0, ( xvals[x_i] - xvals[0] ) )
    lons_ZC[x_i] = new_lon  
lons_ZC[-1] = -80.

#lats similarly    
for y_i in range(yvals.shape[0]-1):
    _, new_lat = find_lat_from_dist(0, -19 , ( yvals[y_i] - yvals[0] ) )
    lats_ZC[y_i] = new_lat
lats_ZC[-1] = 19

lats = np.zeros((31, 30))
lons = np.zeros((31, 30))
for i in np.arange(30): lats[:,i] = lats_ZC
for i in np.arange(31): lons[i,:] = lons_ZC   

ds = xr.Dataset(
    {"temperature": (["lon", "lat", "time"], ZCgridT),
     "thermocline_height": (["lon", "lat", "time"], ZCgridh),},
    coords = { "lat": (['lat'], lats_ZC),
              "lon": (['lon'], lons_ZC),
              "time": tvals}, )

# interpolate to first day of each month
ds_int = ds.interp(time=pd.period_range('1/1/1950', '1/11/1994', freq= 'M').to_timestamp())
ds_int = ds_int.transpose()

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-19, 20, 1)),
                     'lon': (['lon'], np.concatenate((np.arange(124,181, 1), np.arange(-179,-79,1),))),   })

regridder = xe.Regridder(ds_int, ds_out, 'bilinear') # from long decimal number latlon to 1x1

ds_new = regridder(ds_int)
ds_new = ds_new.dropna(dim = 'time') # first value cannot be interpolated because there is no lower bound, remove this
 
sst = ds_new.drop_vars(['thermocline_height'])
h = ds_new.drop_vars(['temperature'])

sst.to_netcdf(join(processeddir, f'sst_ZC_{version}.nc')) # should check if this still works after changing code above
h.to_netcdf(join(processeddir, f'h_ZC_{version}.nc')) # should check if this still works after changing code above

### also calculate anomalys grid
# TO DO: make pauls code work with my data or write my own functions for climatology and such
sst = sst.assign_attrs(name = 'sst')
sst = sst.assign_attrs(dataset = 'ZC_'+version)

sst_anom = computeAnomaly(sst)
sst_anom.to_netcdf(join(processeddir, f'sst_ZC_{version}_anom.nc')) # should check if this still works after changing code above

sst['anomaly'] = sst_anom['temperature']

sst.to_netcdf(join(processeddir, f'sst_ZC_{version}_full.nc')) # should check if this still works after changing code above

