"""
In this module, methods are collected which have not found a proper module yet
to which they belong. Help them find there home!
"""

import numpy as np
from scipy.stats import spearmanr, pearsonr
import math
import pylab as pl

def print_header(string):
    print()
    print("##################################################################")
    print(string)
    print("##################################################################")
    print()


def small_print_header(string):
    print(string)
    print("--------------------------------------")


def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def lowest_indices(ary, n):
    """Returns the n lowest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, n)[:n]
    indices = indices[np.argsort(flat[indices])]
    return np.unravel_index(indices, ary.shape)


def generateFileName(variable, dataset, processed='', suffix=None):
    """
    generates a file name
    """
    filenamelist = [variable, dataset, processed]

    # remove ''  entries from list
    filenamelist = list(filter(lambda a: a != '', filenamelist))

    filename = '_'.join(filenamelist)

    if suffix is not None:
        filename = '.'.join([filename, suffix])

    return filename


def scale(x):
    """
    scale a time series
    """
    return (x-x.mean())/x.std()


def scaleMax(x):
    """
    sacle timeseries by absolute maximum
    """
    return x/np.max(np.abs(x))


"""
here I want to implement the code for the MLP regression and classification
"""


def include_time_lag(X, n_lags=0, step=1):
    Xnew = np.copy(X[n_lags*step:])
    for i in range (1, n_lags):
        Xnew = np.concatenate((Xnew, X[(n_lags-i)*step:-i*step]), axis=1)
    return Xnew


def nino_to_category(nino, categories=None, threshold=None):
    """
    This method translates a NINO index value into a category. NOTE: Either the
    categories OR threshold method can be used!

    :param nino: the timeseries of the NINO index.

    :param categories: The number of categories.

    :param threshod: The threshold for the.
    """
    if categories != None and threshold != None:
        raise Exception("Either categories OR threshold method can be used!")

    if threshold == None:
        sorted_arr = np.sort(nino)
        n = len(sorted_arr)
        n_cat = n//categories
        bounds = np.zeros(categories+1)

        for i in range(1,categories):
            bounds[i] = sorted_arr[i*n_cat]
        bounds[0] = sorted_arr[0] -1
        bounds[-1] = sorted_arr[-1]+1

        nino_cat = np.zeros_like(nino, dtype=int) + categories

        for j in range(categories):
            nino_cat[(nino>bounds[j]) & (nino<=bounds[j+1])] = j

        assert (nino_cat != categories).all()
    else:
        nino_cat = np.zeros_like(nino, dtype=int) + 1
        nino_cat[nino>threshold] = 2
        nino_cat[nino<-threshold] = 0
    return nino_cat


def basin_means(data, lat1=2.5, lat2=-2.5):
    """
    Computes the mean in different basins of the equatorial Pacific

    :param data: The data for which the Basins means shall be computed with
    dimension (time, lat, lon).

    :param lat1, lat2: The latidual bounds

    :returns: The mean in the west Pacific (120E- 160E), the central Pacifc (160E-180E),
    east Pacifc  (180E- 240W).
    """
    data_WP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(120, 160))]
    data_WP_mean = data_WP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    data_CP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(160, 180))]
    data_CP_mean = data_CP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    data_EP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(180, 240))]
    data_EP_mean = data_EP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    return data_WP_mean, data_CP_mean, data_EP_mean

def spearman_lag(x, y, max_lags=80):
    """
    Computes the Spearman lag correlation coefficents using  of x and y until a maximum number of lag time
    steps.

    :param x: The variable that leads.

    :param y: The variable that lags.

    :param max_lags: The maximum number of time steps the for which the
    lag correlation is computed.

    :returns: A timeseries with the lag correlations.
    """
    r = np.zeros(max_lags)
    r[0] = spearmanr(x[:], y[:])[0]
    for i in np.arange(1, max_lags):
        r[i] = spearmanr(x[i:], y[:-i])[0]
    return r

def pearson_lag(x, y, max_lags=28):
    """
    Computes the Pearson lag correlation coefficents using  of x and y until a maximum number of lag time
    steps.

    :param x: The variable that leads.

    :param y: The variable that lags.

    :param max_lags: The maximum number of time steps the for which the
    lag correlation is computed.

    :returns: A timeseries with the lag correlations and the corresponding p-value.
    """
    r, p = np.zeros(max_lags+1), np.zeros(max_lags+1)
    r[0], p[0] = pearsonr(x[:], y[:])
    for i in np.arange(1, max_lags+1):
         r[i], p[i] =  pearsonr(x[i:], y[:-i])
    return r, p

def find_index(array, element):
    """
    Finds index belonging to element in array, duplicates not supported
    """ 
    for i in np.arange(0, array.shape[0]):
        if array[i] ==  element:
            index = i
    return index

def find_indexes_from_latlon(lats, lons, lon_right, lat_top, lon_left, lat_bot):
    """
    NOTE: this method is entirely redundant if you know how to use the .loc method of pandas

    Takes lats an lons arrays (in whatever convention) and returns indices belonging to 
    sides of a box around the coordinates specified as arguments
    """    

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
    # first index: right bound, second index: top bound, third index: left bound, fourth index: bottom bound
    # (indices anti-clockwise around the rectangle starting on the right side    
    return indexes 

def find_distance(lat1, lat2, lon1, lon2):
    """
    Input two coordinates (lat1, lon1), (lat2, lon2) and find the distance between them 
    on a spherical earth
    """
    r = 6371 * 1e3
    d = 2*r * np.arcsin(np.sqrt(np.sin(math.radians(lat2-lat1))**2 + np.cos(math.radians(lat1))*np.cos(math.radians(lat2))*np.sin((math.radians(lon2-lon1))/2)**2))
    print('distance = {d*1e-3} m' )
    return d

def find_lon_from_dist(lon, lat, distance):
    """
    Calculates a longitude found by moving a certain distance among a line
    of equal latitude 
    """    
    r_e = 6371 * 1e3 # m # radius of the earth

    # defined moving easttward
    onedegreelon = 2 * np.pi * r_e * np.cos(math.radians(lat)) / 360
    frac = distance/onedegreelon
    newlon = lon + frac
    if newlon > 180:
        newlon -= 360
    return newlon, lat

def find_lat_from_dist(lon, lat, distance):
    """
    Calculates a latitude found by moving a certain distance among a line
    of equal longitude 
    """
    r_e = 6371 * 1e3 # m # radius of the earth
    
    # defined moving northward
    onedegreelat = 2 * np.pi * r_e / 360

    frac = distance/onedegreelat
    newlat = lat + frac 
    return lon, newlat


def threshold_plot(x, y, th_lo, th_hi, fmt_lo, fmt_mi, fmt_hi):
    idx = np.where(np.diff(y > th))[0]
    x_insert = x[idx] + (th - y[idx]) / (y[idx+1] - y[idx]) * (x[idx+1] - x[idx])
    y_insert = np.full_like(x_insert, th)

    xn, yn = np.insert(x, idx+1, x_insert), np.insert(y, idx+1, y_insert)

    mask = yn > th_hi
    pl.plot(np.ma.masked_where(mask, xn), np.ma.masked_where(mask, yn), fmt_hi, lw=2)

    mask = np.logical_and(th_lo < yn, yn < th_hi)
    pl.plot(np.ma.masked_where(mask, xn), np.ma.masked_where(mask, yn), fmt_mi)
    
    mask = yn < th_lo  
    pl.plot(np.ma.masked_where(mask, xn), np.ma.masked_where(mask, yn), fmt_lo, lw=2)  

    pl.axhline(th_lo, color="black", linestyle="--")
    pl.axhline(th_hi, color="black", linestyle="--")



