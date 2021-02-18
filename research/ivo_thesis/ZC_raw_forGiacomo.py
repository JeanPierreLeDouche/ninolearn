#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:08:41 2021

@author: ivo
"""

from os.path import join
import pandas as pd
import xarray as xr
from scipy.io import loadmat

from ninolearn.pathes import rawdir

import numpy as np
from ninolearn.utils import find_lat_from_dist, find_lon_from_dist
import xesmf as xe
from ninolearn.pathes import processeddir
from ninolearn.preprocess.anomaly import computeAnomaly

def ZC_raw(version = 'default'):

    """ 
    Loads data as produced by the Zebiak Cane model (fort.149 output) and 
    unpacks the .csv into a 3D grid with dimension 3 being time, also regrids to
    1x1 degree and exports h and sst fields. 
    """
    
    if version == 'default': print('must specify version!')
    
    path = join(rawdir ,('fort.149_' + version))
    headers = ['time' , 'x' ,'y' , 'h' , 'T' ,'u_A' ,'T_0' ,'wind' ,'tau_x']
    data = pd.read_csv(path, sep = '\s+', names = headers)
    data['time'] = pd.to_datetime(data['time'], unit = 's') + pd.DateOffset(years = -20)   
    
    if any(data.dtypes == 'O'):
        data = data.apply(pd.to_numeric, errors = 'coerce')
        data = data.dropna()
    
    
    yvals = np.unique(data['y'])
    xvals = np.unique(data['x'])
    tvals = np.unique(data['time'])
    

    ### make grids in the funky ZC cartesian system of the h and T (add if needed)
    
    ZCgridT = np.zeros(( xvals.shape[0], yvals.shape[0], tvals.shape[0]))
    ZCgridh = np.zeros(( xvals.shape[0], yvals.shape[0], tvals.shape[0]))
    

    
    for t in enumerate(tvals):
        data_t = data[data['time'] == t[1]]
        
        for y in enumerate(yvals):
            data_y = data_t[data_t['y'] == y[1]]   
            
            ZCgridT[:,y[0],t[0]] = data_y['T']
            ZCgridh[:,y[0],t[0]] = data_y['h']
    
    ### make lat lon grids out of the ZC cartesian system by calculating the latitude 
    ### from the distance  with respect to a starting point at (-29,124). 
    
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
     
    """
    the ZC data needs to be converted into an xarray in order to use the xesmf 
    regridding tool
    """
    
    ds = xr.Dataset({"temperature": (["lon", "lat", "time"], ZCgridT),
         "thermocline_height": (["lon", "lat", "time"], ZCgridh), },
        coords = { "lat": (['lat'], lats_ZC),
                  "lon": (['lon'], lons_ZC),
                  "time": tvals},)
    
    ds = ds.transpose()
    
    # regridding to a simple grid of 1x1 degrees defined here
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-19, 20, 1)),
                         'lon': (['lon'], np.concatenate((np.arange(124, 181, 1), np.arange(-179,-79,1),))),   })
    
    # regrid to 1x1 degrees
    regridder = xe.Regridder(ds, ds_out, 'bilinear')
    ds_new = regridder(ds)
    
    # interpolate to first of month
    ds_new = ds_new.interp(time=pd.period_range('1/1/1950', '1/11/1994', freq= 'M').to_timestamp())
    ds_new = ds_new.dropna(dim = 'time') # first value cannot be interpolated because there is no lower bound, remove this
    
    # export 1x1 grid (currently not used)
    filename = 'ZC1x1_' + version
    ds_new.to_netcdf(join(processeddir, filename))
    
    # output h anomaly (no need to calculate, this is straight from ZC)
    h = ds_new.drop_vars(['temperature'])
    h = h.assign_attrs(name = 'h')
    h = h.assign_attrs(dataset = 'ZC_'+version)
    
    h.to_netcdf(join(processeddir, f'h_ZC_{version}.nc'))
    
    # output sst and sst anomaly
    sst = ds_new.drop_vars(['thermocline_height'])
    sst = sst.assign_attrs(name = 'sst')
    sst = sst.assign_attrs(dataset = 'ZC_'+version)
    
    sst_anom = computeAnomaly(sst)
    sst['anomaly'] = sst_anom['temperature']
    
    sst.to_netcdf(join(processeddir, f'sst_ZC_{version}.nc')) 