#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:10:46 2020

@author: ivo
"""
import pandas as pd
import numpy as np
import xarray as xr
import xesmf as xe
import math

r_e = 6371 * 1e3 # m

# units supposedly dimensional 

path = r'/home/ivo/fort.149'
headers = ['time', 'x', 'y', 'h', 'T', 'u_A', 'T_0', 'wind', 'tau_x']
data = pd.DataFrame(np.genfromtxt(path), columns=headers)

def find_distance(lat1, lat2, lon1, lon2):
    r = 6371 * 1e3
    d = 2*r * np.arcsin(np.sqrt(np.sin(math.radians(lat2-lat1))**2 + np.cos(math.radians(lat1))*np.cos(math.radians(lat2))*np.sin((math.radians(lon2-lon1))/2)**2))
    print('distance = {d*1e-3} m' )
    return d

def find_lon_from_dist(lon, lat, distance):
    # defined moving easttward
    onedegreelon = 2 * np.pi * r_e * np.cos(math.radians(lat)) / 360
    frac = distance/onedegreelon
    newlon = lon + frac
    if newlon > 180:
        newlon -= 360
    return newlon, lat

def find_lat_from_dist(lon, lat, distance):
    # defined moving northward
    onedegreelat = 2 * np.pi * r_e / 360

    frac = distance/onedegreelat
    newlat = lat + frac 
    return lon, newlat

# TO DO: somehow change these 40 km steps into a lat lon grid or anything I 
# can use to calculate the ONI and network characteristics
""" original data is a LxW box of 15 000 km x 4200 km where the short direction 
 runs from -2109 to 2109. According to the ZC1987 paper this corresponds to 
 124E-80W by 29S-29N (which is in reality 17335km x 6445km). 
 
 NOTE: the long dimension of 15 000 km is the average between the horizontal 
 extend at the equator and 29N, however the 4200 km can only apply to a box from 
 19S to 19N which is also shown in the papers figures
"""

yvals = np.unique(data['y'])
xvals = np.unique(data['x'])
tvals = np.unique(data['time'])

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

#%%

"""
try to somehow make lat lon grids out of the ZC cartesian system 
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

for i in np.arange(30):
    lats[:,i] = lats_ZC

for i in np.arange(31):
    lons[i,:] = lons_ZC   

ds = xr.Dataset(
    { 
     "temperature": (["lon", "lat", "time"], ZCgridT),
     "thermocline_height": (["lon", "lat", "time"], ZCgridh),
     },
    coords = {
        "lat": (['lat'], lats_ZC),
                "lon": (['lon'], lons_ZC),

        "time": tvals
        },
    )

ds = ds.transpose()

ds_out = xr.Dataset({'lat': (['lat'], np.arange(-19, 20, 1.0)),
                     'lon': (['lon'], np.concatenate((np.arange(124,181, 1), np.arange(-179,-79,1),))),
                     
                    }
                   )

regridder = xe.Regridder(ds, ds_out, 'bilinear')
ds_new = regridder(ds)

