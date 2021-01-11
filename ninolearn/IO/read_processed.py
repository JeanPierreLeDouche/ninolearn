from os.path import join
import pandas as pd
import xarray as xr
import gc

from ninolearn.pathes import processeddir
from ninolearn.utils import generateFileName

#TODO: Write a routine that generates this list
csv_vars = ['nino3.4M','nino3.4S', 'wwv']


class data_reader(object):
    def __init__(self, startdate='1980-01', enddate='2018-12',
                 lon_min=120, lon_max=280, lat_min=-30, lat_max=30):
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
        self.startdate = self.startdate + pd.DateOffset(months=month)
        self.enddate = self.enddate + pd.DateOffset(months=month) \
            + pd.tseries.offsets.MonthEnd(0)

    def read_csv(self, variable, processed='anom'):
        """
        get data from processed csv
        """
        data = pd.read_csv(join(processeddir, f"{variable}.csv"),
                           index_col=0, parse_dates=True)


        self._check_dates(data, f"{variable}")

        return data[processed].loc[self.startdate:self.enddate]

    def read_netcdf(self, variable, dataset='', processed='', chunks=None):
        """
        wrapper for xarray.open_dataarray.

        :param variable: the name of the variable
        :param dataset: the name of the dataset
        :param processed: the postprocessing that was applied
        :param chunks: same as for xarray.open_dataarray
        """
        filename = generateFileName(variable, dataset,
                                    processed=processed, suffix="nc")

        data = xr.open_dataarray(join(processeddir, filename), chunks=chunks)

        regrided = ['GODAS', 'ERSSTv5', 'ORAS4', 'NODC', 'NCAR']
        ZCdatasets = [ 'ZC_25x25_undistorted']

        if processed=='meanclim':
            return data

        else:
            self._check_dates(data, f'{filename[:-3]}')
            if dataset not in regrided  and dataset!='ORAP5' and  dataset != 'GFDL-CM3' and dataset not in ZCdatasets:
                return data.loc[self.startdate:self.enddate,
                                self.lat_max:self.lat_min,
                                self.lon_min:self.lon_max]

            elif dataset not in regrided  and dataset!='ORAP5' and  dataset != 'GFDL-CM3' and dataset in ZCdatasets:
                print('Reading ZC model data') 
                # NOTE: the order of arguments to the .loc method is essential!
                return data.loc[self.lon_min:self.lon_max,
                                self.lat_min:self.lat_max,
                                self.startdate:self.enddate]

            elif dataset in regrided or dataset == 'GFDL-CM3':
                return data.loc[self.startdate:self.enddate,
                                self.lat_min:self.lat_max,
                                self.lon_min:self.lon_max]
            elif dataset=='ORAP5':
                return data.loc[self.startdate: self.enddate, :, :].where(
                       (data.nav_lat > self.lat_min) &
                       (data.nav_lat < self.lat_max) &
                       (data.nav_lon > self.lon_min) &
                       (data.nav_lon < self.lon_max),
                       drop=True)

    def read_statistic(self, statistic, variable, dataset='', processed=''):

        filename = generateFileName(variable, dataset,
                                    processed=processed, suffix="csv")
        filename = '-'.join([statistic, filename])

        data = pd.read_csv(join(processeddir, filename),
                           index_col=0, parse_dates=True)
        self._check_dates(data, f"{variable} - {statistic}" )
        return data.loc[self.startdate:self.enddate]

    def read_forecasts(self, model_name, lead, filename=None):
        """
        Read forecasts from a NinoLearn model.
        """
        if filename is None:
            ds = xr.open_dataset(join(processeddir, f'{model_name}_forecasts.nc'))
        else:
            ds = xr.open_dataset(join(processeddir, filename))

        data = ds.loc[{'target_season': slice(self.startdate, self.enddate), 'lead': lead}]
        return data

    def read_other_forecasts(self, model, lead):
        """
        Read forecasts from other models.

        :type model: str
        :param model: Model name.
        """
        ds = xr.open_dataset(join(processeddir, f'other_forecasts.nc'))
        data = ds[model].loc[self.startdate:self.enddate, lead]
        return data

    def _check_dates(self, data, name):
        """
        Checks if provided start and end date are in the bounds of the data
        that should be read.
        """
        if isinstance(data, xr.DataArray):
            if self.startdate < data.time.values.min():
                raise IndexError("The startdate is out of\
                                 bounds for %s data!" % name)
            if self.enddate > pd.to_datetime(data.time.values.max()) + pd.tseries.offsets.MonthEnd(0):
                print(data.time.values.max())
                print(self.enddate)
                raise IndexError("The enddate is out of bounds for %s data!" % name)

        if isinstance(data, pd.DataFrame):
            if self.startdate < data.index.values.min():
                msg = f"The startdate is out of bounds for {name} data!"
                raise IndexError(msg)
            if self.enddate > pd.to_datetime(data.index.values.max()) + pd.tseries.offsets.MonthEnd(0):
                print( self.enddate )
                print(data.index.values.max())
                raise IndexError("The enddate is out of bounds for %s data!" % name)


if __name__ == "__main__":
    reader = data_reader(startdate="1981-01", enddate='2018-12',
                         lon_min=120, lon_max=380, lat_min=-30, lat_max=30)

    data = reader.read_netcdf('sshg', dataset='GODAS', processed='anom')
    data2 = reader.read_netcdf('zos', dataset='GFDL-CM3', processed='anom')

