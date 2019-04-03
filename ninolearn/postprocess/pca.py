import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from os.path import join
from sklearn.decomposition.pca import PCA
from mpl_toolkits.basemap import Basemap
from matplotlib import cm

from ninolearn.IO.read_post import data_reader
from ninolearn.pathes import postdir
from ninolearn.utils import generateFileName, scaleMax
from ninolearn.plot.nino_timeseries import nino_background


class pca(PCA):
    """
    This class extends the PCA class of the sklearn.decomposition.pca modlue.
    It facilitates the loading of the data from the postprocessed directory,
    wraps the fit function of the PCA class, has a saving routine for the
    computed pca component and can plot the EOF to get more insight into the
    results.
    """
    def load_data(self, variable, dataset, processed='anom',
                  startyear=1949, endyear=2018, lon_min=120, lon_max=280,
                  lat_min=-30, lat_max=30):
        """
        Load data for PCA analysis from the desired postprocessed data set

        :type variable: str
        :param variable: the variable for which the network time series should
        be computed

        :type dataset: str
        :param dataset: the dataset that should be used to build the network

        :type processed: str
        :param processed: either '','anom' or 'normanom'

        :param startyear: the first year for which the network analysis should
        be done

        :param endyear: the last year for which the network analysis should be
        done

        :param lon_min,lon_max: the min and the max values of the longitude
        grid for which the metrics shell be computed (from 0 to 360 degrees
        east)

        :param lat_min,lat_max:the min and the max values of the latitude grid
        for which the metrics shell be computed (from -180 to 180 degrees east)

        :type plot: bool
        :param: make some plots to view the EOFs
        """

        self.variable = variable
        self.dataset = dataset
        self.processed = processed

        self.startdate = pd.to_datetime(str(startyear))
        self.enddate = (pd.to_datetime(str(endyear)) +
                        pd.tseries.offsets.YearEnd(0))

        self.reader = data_reader(startdate=self.startdate,
                                  enddate=self.enddate)
        data = self.reader.read_netcdf(variable, dataset, processed)

        self.time = data['time']
        self.lon = data['lon']
        self.lat = data['lat']

        EOFarr = np.array(data[:, :, :])

        self.len_time = len(self.time)
        self.len_lat = len(self.lat)
        self.len_lon = len(self.lon)

        self.EOFarr = EOFarr.reshape((self.len_time,
                                      self.len_lat * self.len_lon))

    def compute_pca(self):
        """
        Simple wrapper around the PCA.fit() method.
        """
        self.fit(self.EOFarr)

    def save(self):
        """
        save the first three pca components to a csvfile
        """
        # save data to first day of month ahead
        save_index = self.time.to_index()+pd.tseries.offsets.MonthBegin(1)

        pca1 = pd.Series(np.matmul(self.EOFarr, self.components_[0, :]),
                         index=save_index)
        pca2 = pd.Series(np.matmul(self.EOFarr, self.components_[1, :]),
                         index=save_index)
        pca3 = pd.Series(np.matmul(self.EOFarr, self.components_[2, :]),
                         index=save_index)

        df = pd.DataFrame({'pca1': pca1, 'pca2': pca2, 'pca3': pca3})

        filename = generateFileName(self.variable, self.dataset,
                                    self.processed, suffix='csv')
        filename = '-'.join(['pca', filename])

        df.to_csv(join(postdir, filename))

    def plot_eof(self):
        """
        Make a plot for the first leading EOFs.
        """
        lon2, lat2 = np.meshgrid(self.lon, self.lat)

        nino34 = self.reader.read_csv('nino34')

        fig = plt.figure(figsize=(15, 7))

        for i in range(0, 2):
            fig.add_subplot(221+i)
            plt.title("EOF"+str(i+1))
            m = Basemap(projection='robin', lon_0=180, resolution='c')
            x, y = m(lon2, lat2)

            m.drawparallels(np.arange(-90., 120., 30.))
            m.drawmeridians(np.arange(0., 360., 60.))
            m.drawmapboundary(fill_color='white')
            m.drawcoastlines()

            norm = cm.colors.Normalize(vmax=-1, vmin=1.)
            cmap = cm.bwr
            cs = m.pcolormesh(x, y, scaleMax(
                              self.components_[i, :].reshape(self.len_lat,
                                                             self.len_lon)),
                              cmap=cmap, norm=norm)
            m.colorbar(cs)

        for i in range(0, 2):
            fig.add_subplot(223+i)
            projection = np.matmul(self.EOFarr, self.components_[i, :])
            nino_background(nino34)
            plt.plot(self.time, projection)


if __name__ == "__main__":
    pca_obj = pca(n_components=6)
    pca_obj.load_data('air', 'NCEP', 'anom', startyear=1950, endyear=2018)
    pca_obj.compute_pca()
    pca_obj.save()
    pca_obj.plot_eof()