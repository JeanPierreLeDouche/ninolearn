#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:10:46 2020

@author: ivo
"""

from os.path import join, exists
import xarray as xr
import pandas as pd

# from ninolearn.pathes import processeddir
# from ninolearn.utils import generateFileName, small_print_header
# from ninolearn.preprocess.anomaly import _get_period, computeMeanClimatology
# from ninolearn.preprocess.prepare import
from ninolearn.IO.read_processed import data_reader
# from ninolearn.IO.read_raw import oni
# from ninolearn.IO.read_raw import nino34detrend_anom
from ninolearn.download import download, sources
from ninolearn.pathes import processeddir, rawdir

# import pandas as pd
import numpy as np
# import xarray as xr
import xesmf as xe
import math

r_e = 6371 * 1e3 # m

# units supposedly dimensional 

path = r'/home/ivo/fort.149'
headers = ['time', 'x', 'y', 'h', 'T', 'u_A', 'T_0', 'wind', 'tau_x']
data = pd.DataFrame(np.genfromtxt(path), columns=headers)

def find_index(array, element): 
    for i in np.arange(0, array.shape[0]):
        if array[i] ==  element:
            index = i
    return index

def find_indexes_from_latlon(lats, lons, lon_right, lat_top, lon_left, lat_bot):
    indexes = [0, 0, 0, 0]
    for i in range(0, lats.shape[0]):
        if lats[i] == lat_bot:
            indexes[3] = i
        elif lats[i] == lat_top:
            indexes[1] = i            
    for j in range(0, lons.shape[0]):
        if lons[j] == lon_left:
            indexes[2] = j
        elif lons[j] == lon_right:
            indexes[0] = j
    return indexes 

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
 19S to 19N which is also shown in the papers' figures
"""

data['time'] = pd.to_datetime(data['time'], unit = 's') + pd.DateOffset(years = 10)    

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


""" 
make lat lon grids out of the ZC cartesian system by calculating the latitude 
from the distance  with respect to a starting point at (-29,124). 
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

"""
the ZC data needs to be converted into an xarray in order to use the xesmf 
regridding tool
"""

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
# regridding to a simple grid of 1x1 degrees defined here
ds_out = xr.Dataset({'lat': (['lat'], np.arange(-19, 20, 1.0)),
                     'lon': (['lon'], np.concatenate((np.arange(124,181, 1), np.arange(-179,-79,1),))),        
                    }
                   )

regridder = xe.Regridder(ds, ds_out, 'bilinear')
ds_new = regridder(ds)

ds_new.to_netcdf(join(processeddir, 'ZCregrid1x1.nc'))

### now we have nice grids
ZC_T_grid = ds_new['temperature']
ZC_h_grid = ds_new['thermocline_height']

###NINO34 box indices
NIN34inds = find_indexes_from_latlon(ZC_T_grid['lat'], ZC_T_grid['lon'], -120, 5, -170, -5)

### take means of slices only within the NINO34 area
ZC_T_NIN34 = np.mean(np.mean(ZC_T_grid[:, NIN34inds[3]:NIN34inds[1], NIN34inds[2]:NIN34inds[0]], axis =2 ), axis =1 )
ZC_h_NIN34 = ZC_h_grid[:, NIN34inds[3]:NIN34inds[1], NIN34inds[2]:NIN34inds[0]]

### introduce a time axis that has been previously used when supplying data to the 
### DEM (first of every month)
new_t_axis = pd.period_range('1/1/1975', '12/1/2019', freq= 'M')
ZC_SST= pd.DataFrame(ZC_T_NIN34,index = new_t_axis, columns = ['SST']).interpolate(method='time')

### now to get the actual NINO34 we need to subtract the centered 30 year mean climatology 
# download(sources.ONI)
reader = data_reader(startdate='1951-01', enddate='2019-12')
oni = reader.read_csv('oni', processed = 'TOTAL').transpose() # values of januari belong to NDJ

running_means = np.zeros(ZC_SST.shape[0])
running_means[0], running_means[1] = ZC_SST.iloc[0], (ZC_SST.iloc[1]+ZC_SST.iloc[0])/2

climatology_eras = np.arange(1975,2019,5)
climatologies = pd.DataFrame(np.zeros((climatology_eras.shape[0], 12)), climatology_eras, columns = range(1,13))

# calculate running means
for i in range(ZC_SST.shape[0]-2):
    avg = (ZC_SST.iloc[i] + ZC_SST.iloc[i+1] + ZC_SST.iloc[i+2] )/3
    running_means[i+2] = avg

df_running_means = pd.DataFrame(running_means, index = new_t_axis, columns =['runningmean'] )

# calculate climatologies for each specific window, defining an era from the 
# first 5 divisble year onward: so 1965-1970 falls under era 1965
for era in climatology_eras:
    start = era - 15
    end = era + 10
    
    oni_data = oni[str(start):str(end)]
    for month in range(1, 13):
        month_data = oni_data[oni_data.index.month == month]
        month_avg = np.mean(month_data)
        climatologies.loc[era][month] = month_avg 

ZC_oni = pd.DataFrame(np.zeros(ZC_SST.shape[0]), index = new_t_axis, columns = ['anom'])
for era in climatology_eras:
    era_data = ZC_SST[str(era):str(era+4)]
    for i in range(era_data.shape[0]):
        month = era_data.index.month[i]
        oni_val = era_data.iloc[i] - climatologies.loc[era][month]    
        ZC_oni.loc[era_data.index[i]] = oni_val[0]
        
ZC_oni.to_csv(join(processeddir, 'ZC_oni.csv'))




