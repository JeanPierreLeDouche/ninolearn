from os.path import join
import pandas as pd
import xarray as xr
import gc

from ninolearn.pathes import postdir
from ninolearn.utils import generateFileName

class data_reader(object):
    def __init__(self, startdate = '1980-01', enddate = '2017-12', lon_min = 120, lon_max = 260, lat_min = -30, lat_max = 30):
        """
        Data reader for different kind of El Nino related data.
        
        :param startdate:year and month from which on data should be loaded 
        :param enddate: year and month to which data should be loaded 
        :lon_min: eastern boundary of data set in degrees east
        :lon_max: western boundary of data set in degrees east
        :lat_min: southern boundary of data set in degrees north
        :lat_max: northern boundary of data set in degrees north
        """
        self.startdate = pd.to_datetime(startdate)
        self.enddate = pd.to_datetime(enddate) + pd.tseries.offsets.MonthEnd(0)
        
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        
    def __del__(self):
        gc.collect()
        
    def shift_window(self, month=1):
        self.startdate =self.startdate + pd.DateOffset(months=month)
        self.enddate = self.enddate + pd.DateOffset(months=month) + pd.tseries.offsets.MonthEnd(0)
        
    def nino34_anom(self):
        """
        get the Nino3.4 Index anomaly
        """        
        data = pd.read_csv(join(postdir,"nino34.csv"),index_col=0, parse_dates=True)
        self._check_dates(data, "Nino3.4")
        
        return data.ANOM.loc[self.startdate:self.enddate]
    
    def wwv_anom(self):
        """
        get the warm water volume anomaly
        """
        data = pd.read_csv(join(postdir,"wwv.csv"),index_col=0, parse_dates=True)
        self._check_dates(data, "WWV")
        
        return data.Anomaly.loc[self.startdate:self.enddate]
    
    def read_netcdf(self, variable, dataset='', processed=''):
        
        filename = generateFileName(variable, dataset, processed,'nc')
        
        try: 
            data = xr.open_dataarray(join(postdir,filename))
        except:
            raise Exception(f'Data for {filename[:-4]} not found!')
        
        self._check_dates(data, f'{filename[:-3]}')
            
        return data.loc[self.startdate:self.enddate,
                                self.lat_max:self.lat_min,
                                self.lon_min:self.lon_max]
    
    def _check_dates(self,data, name):
        """
        Checks if provided start and end date are in the bounds of the data that 
        should be read.
        """
        if isinstance(data,xr.DataArray):
            if self.startdate < data.time.values.min():
                raise IndexError("The startdate is out of bounds for %s data!"%name)
            if self.enddate > data.time.values.max():
                raise IndexError("The enddate is out of bounds for %s data!"%name)
        
        if isinstance(data,pd.DataFrame):
            if self.startdate < data.index.values.min():
                raise IndexError("The startdate is out of bounds for %s data!"%name)
            if self.enddate > data.index.values.max():
                raise IndexError("The enddate is out of bounds for %s data!"%name)

if __name__ == "__main__":
    reader = data_reader(startdate="1981-01", enddate='1990-12')
    nino34 = reader.nino34_anom()
    #wwv = reader.wwv_anom()
    sst_ERSST = reader.read_netcdf('sst', 'ERSSTv5','norm')