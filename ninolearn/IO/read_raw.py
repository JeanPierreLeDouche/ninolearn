from os.path import join
import pandas as pd
import xarray as xr
from scipy.io import loadmat

from ninolearn.pathes import rawdir

"""
This module collects a bunch methods to read the raw data files.
"""


def nino34_anom():
    """
    Get the Nino3.4 Index anomaly.
    """
    data = pd.read_csv(join(rawdir, "nino34.txt"), delim_whitespace=True)
    return data

def nino34detrend_anom():
    """
    Get the detrended Nino3.4 Index anomaly.
    """
    data = pd.read_csv(join(rawdir, "nino34detrend.txt"), delim_whitespace=True)
    return data

def oni():
    return nino_anom(index="3.4", period ="S")

def nino_anom(index="3.4", period ="S", detrend=False):
    """
    read various Nino indeces from the raw directory
    """
    try:
        if period == "S":
            if index == "3.4" and not detrend:
                data = pd.read_csv(join(rawdir, "oni.txt"),
                                   delim_whitespace=True)
            else:
                msg = "Only not detrended Nino3.4 index is available for seasonal records"
                raise Exception(msg)


        elif period == "M":
            if detrend and index == "3.4":
                data = pd.read_csv(join(rawdir, "nino34detrend.txt"),
                               delim_whitespace=True)
            elif not detrend:
                data = pd.read_csv(join(rawdir, "nino_1_4.txt"),
                                           delim_whitespace=True)
        return data

    except UnboundLocalError:
        raise Exception("The desired NINO index is not available.")


def wwv_anom(cardinal_direction=""):
    """
    get the warm water volume anomaly
    """
    if cardinal_direction != "":
        filename = f"wwv_{cardinal_direction}.dat"
    else:
        filename = "wwv.dat"

    data = pd.read_csv(join(rawdir, filename),
                       delim_whitespace=True, header=4)
    return data

def iod():
    """
    get IOD index data
    """
    data = pd.read_csv(join(rawdir, "iod.txt"),
                       delim_whitespace=True, header=None, skiprows=1, skipfooter=7,
                       index_col=0, engine='python')
    return data

def K_index():
    data = loadmat(join(rawdir, "Kindex.mat"))

    kindex = data['Kindex2_mon_anom'][:,0]
    time = pd.date_range(start='1955-01-01', end='2011-12-01', freq='MS')
    ds = pd.Series(data=kindex, index=time)
    return ds

def sst_ERSSTv5():
    """
    get the sea surface temperature from the ERSST-v5 data set
    """
    data = xr.open_dataset(join(rawdir, 'sst.mnmean.nc'))
    data.sst.attrs['dataset'] = 'ERSSTv5'
    return data.sst


def sst_HadISST():
    """
    get the sea surface temperature from the ERSST-v5 data set and directly
    manipulate the time axis in such a way that the monthly mean values are
    assigned to the beginning of a month as this is the default for the other
    data sets
    """
    data = xr.open_dataset(join(rawdir, "HadISST_sst.nc"))
    maxtime = pd.to_datetime(data.time.values.max()).date()
    data['time'] = pd.date_range(start='1870-01-01', end=maxtime, freq='MS')
    data.sst.attrs['dataset'] = 'HadISST'
    return data.sst

def ustr():
    """
    get u-wind stress from ICOADS 1-degree Enhanced

    """
    data = xr.open_dataset(join(rawdir, "upstr.mean.nc"))
    data.upstr.attrs['dataset'] = 'ICOADS'
    return data.upstr

def uwind():
    """
    get u-wind from NCEP/NCAR reanalysis
    """
    data = xr.open_dataset(join(rawdir, "uwnd.mon.mean.nc"))
    data.uwnd.attrs['dataset'] = 'NCEP'
    return data.uwnd


def vwind():
    """
    get v-wind from NCEP/NCAR reanalysis
    """
    data = xr.open_dataset(join(rawdir, "vwnd.mon.mean.nc"))
    data.vwnd.attrs['dataset'] = 'NCEP'
    return data.vwnd


def sat(mean='monthly'):
    """
    Get the surface air temperature from NCEP/NCAR Reanalysis

    :param mean: Choose between daily and monthly mean fields
    """
    if mean == 'monthly':
        data = xr.open_dataset(join(rawdir, "air.mon.mean.nc"))
        data.air.attrs['dataset'] = 'NCEP'
        return data.air

    elif mean == 'daily':
        data = xr.open_mfdataset(join(rawdir, 'sat', '*.nc'))
        data_return = data.air

        data_return.attrs['dataset'] = 'NCEP'
        data_return.name = 'air_daily'
        return data_return


def olr():
    """
    get v-wind from NCEP/NCAR reanalysis
    """
    data = xr.open_dataset(join(rawdir, "olr.mon.mean.nc"))
    data.olr.attrs['dataset'] = 'NCAR'
    return data.olr


def ssh():
    """
    Get sea surface height. And change some attirbutes and coordinate names
    """
    data = xr.open_mfdataset(join(rawdir, 'ssh', '*.nc'),
                             concat_dim='time_counter')
    data_return = data.sossheig.rename({'time_counter': 'time'})
    maxtime = pd.to_datetime(data_return.time.values.max()).date()
    data_return['time'] = pd.date_range(start='1979-01-01',
                                        end=maxtime,
                                        freq='MS')
    data_return.attrs['dataset'] = 'ORAP5'
    data_return.name = 'ssh'
    return data_return

def godas(variable="sshg"):
    ds = xr.open_mfdataset(join(rawdir, f'{variable}_godas', '*.nc'),
                             concat_dim='time')

    if len(ds[variable].shape)==4:
        data = ds.loc[dict(level=5)].load()
    else:
        data = ds.load()

    data[variable].attrs['dataset'] = 'GODAS'
    return data[variable]

def oras4():
    ds = xr.open_mfdataset(join(rawdir, f'ssh_oras4', '*.nc'),
                             concat_dim='time')
    data = ds.load()
    data.zos.attrs['dataset'] = 'ORAS4'
    return data.zos

def sat_gfdl():
    data = xr.open_mfdataset(join(rawdir, 'sat_gfdl', '*.nc'),
                             concat_dim='time')

    data = data.load()
    data.tas.attrs['dataset'] = 'GFDL-CM3'

    # this change needs to be done to prevent OutOfBoundsError
    data['time'] = pd.date_range(start='1700-01-01', end='2199-12-01',freq='MS')
    return data.tas

def ssh_gfdl():
    data = xr.open_mfdataset(join(rawdir, 'ssh_gfdl', '*.nc'),
                             concat_dim='time')
    #data = data.load()
    data.zos.attrs['dataset'] = 'GFDL-CM3'

    # this change needs to be done to prevent OutOfBoundsError
    data['time'] = pd.date_range(start='1700-01-01', end='2199-12-01',freq='MS')
    return data.zos


def sst_gfdl():
    data = xr.open_mfdataset(join(rawdir, 'sst_gfdl', '*.nc'),
                             concat_dim='time')
    #data = data.load()
    data.tos.attrs['dataset'] = 'GFDL-CM3'

    # this change needs to be done to prevent OutOfBoundsError
    data['time'] = pd.date_range(start='1700-01-01', end='2199-12-01',freq='MS')
    return data.tos

def hca_mon():
    """
    heat content anomaly, seasonal variable to the first day of the middle season
    and upsample the data
    """
    data = xr.open_dataset(join(rawdir, "hca.nc"), decode_times=False)
    data['time'] = pd.date_range(start='1955-02-01', end='2019-02-01', freq='3MS')
    data.h18_hc.attrs['dataset'] = 'NODC'

    data_raw = data.h18_hc[:,0,:,:]
    data_upsampled = data_raw.resample(time='MS').interpolate('linear')
    data_upsampled.name = 'hca'
    return data_upsampled


def other_forecasts():
    data = pd.read_csv(join(rawdir, "other_forecasts.csv"), error_bad_lines=False,
                       header=None, names=['row'], delimiter=';')

    return data

import numpy as np
from ninolearn.utils import find_lat_from_dist, find_lon_from_dist
import xesmf as xe
from ninolearn.pathes import processeddir, rawdir
from ninolearn.preprocess.anomaly import computeMeanClimatology, computeAnomaly


def ZC_raw(version = 'default'):

    """ 
    Loads data as produced by the Zebiak Cane model (fort.149 output) and 
    unpacks the .csv into a 3D grid with dimension 3 being time, also regrids to
    1x1 degree and exports h and sst fields. 
    """
    
    if version == 'default': print('must specify version!')
    
    path = join(rawdir ,('fort.149_' + version))
    data = pd.read_csv(path, index_col=0)
    data['time'] = pd.to_datetime(data['time'], unit = 's') + pd.DateOffset(years = -20)    
    
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
    
    h.to_netcdf(join(processeddir, f'h_ZC_{version}.nc')) # should check if this still works after changing code above
    
    # output sst and sst anomaly
    sst = ds_new.drop_vars(['thermocline_height'])
    sst = sst.assign_attrs(name = 'sst')
    sst = sst.assign_attrs(dataset = 'ZC_'+version)
    
    sst_anom = computeAnomaly(sst)
    sst['anomaly'] = sst_anom['temperature']
    
    sst.to_netcdf(join(processeddir, f'sst_ZC_{version}.nc')) ### INPUT FOR NMS (and ONI), NAMING NOW DIFFERENT
        
def ZC_h(version = 'default'):
    '''
    ZC does not produce a warm water volume so thermocline heigh anomaly
    in the same area is used
    '''
    if version == 'default': print('must specify version!')

    rawfilename = 'h_ZC_' + version + '.nc'
    h = xr.load_dataset(join(processeddir, rawfilename))    
    
    ### WWV is typically calculated for 5S-5N and 120E-280E ( technically 100W ofc)
    
    h_dataset = h['thermocline_height'].loc[:, -5:5, :]
    h_means = np.mean( np.mean(h_dataset, axis =2), axis =1)
    df_h = pd.DataFrame(h_means, index = np.asarray(h_means.time), columns = ['anom'])
    
    df_h.to_csv(join(processeddir, 'h_mean_ZC_' + version + '.csv' ))

def ZC_oni(version = 'default', NIN34=False):
    """
    calculates oni using climatology from 1960 to 1990. This code is redundant 
    and can be replaced by using PP's computeAnomaly function I think
    TODO: see if this can be replaced
    
    """    
    if version == 'default': print('must specify version!')
    
    inputfilename = 'sst_ZC_'+ version + '.nc'
    data = xr.open_dataset(join(processeddir, inputfilename))
    data = data.drop_vars('month')
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
    
    df_ONI.to_csv(join(processeddir, 'oni_ZC_' + version + '.csv'))
    
    if NIN34 == True:
        SST_NIN34_mly.to_csv(join(processeddir, 'nin34' + version + '.csv')) 
    

    

    

    
