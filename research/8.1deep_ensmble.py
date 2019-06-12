# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

import keras.backend as K

from sklearn.preprocessing import StandardScaler

from ninolearn.IO.read_post import data_reader
from ninolearn.plot.evaluation  import plot_correlation
from ninolearn.plot.prediction import plot_prediction
from ninolearn.learn.evaluation import rmse
from ninolearn.learn.mlp import include_time_lag
from ninolearn.learn.dem import DEM
from ninolearn.utils import print_header
from ninolearn.pathes import modeldir

#%%
K.clear_session()

def basin_means(data, lat1=2.5, lat2=-2.5):
    data_WP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(120, 160))]
    data_WP_mean = data_WP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    data_CP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(160, 180))]
    data_CP_mean = data_CP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    data_EP = data.loc[dict(lat=slice(lat1, lat2), lon=slice(180, 240))]
    data_EP_mean = data_EP.mean(dim='lat', skipna=True).mean(dim='lon', skipna=True)

    return data_WP_mean, data_CP_mean, data_EP_mean
#%% =============================================================================
# read data
# =============================================================================
reader = data_reader(startdate='1981-01')

# NINO3.4 Index
nino34 = reader.read_csv('nino3.4S')

# Other indeces
iod = reader.read_csv('iod')
wwv = reader.read_csv('wwv')
wwv_west = reader.read_csv('wwvwest')

# seasonal cycle
sc = np.cos(np.arange(len(nino34))/12*2*np.pi)

# SSH network metrics
network_ssh = reader.read_statistic('network_metrics', variable='sshg',
                           dataset='GODAS', processed="anom")
c2_ssh = network_ssh['fraction_clusters_size_2']
H_ssh = network_ssh['corrected_hamming_distance']
S = network_ssh['fraction_giant_component']
H = network_ssh['corrected_hamming_distance']
T_ssh = network_ssh['global_transitivity']
C = network_ssh['avelocal_transmissivity']
L_ssh = network_ssh['average_path_length']

#wind stress
taux = reader.read_netcdf('taux', dataset='NCEP', processed='anom')
taux_WP_mean, taux_CP_mean, taux_EP_mean = basin_means(taux)

ucur = reader.read_netcdf('ucur', dataset='GODAS', processed='anom')
ucur_WP_mean, ucur_CP_mean, ucur_EP_mean = basin_means(ucur, lat1=-7.5, lat2=7.5)


#%% =============================================================================
#  process data
# =============================================================================
time_lag = 2
lead_time = 6
shift = 3

feature_unscaled = np.stack((nino34, sc,  iod,
                             wwv, wwv_west,
                             H_ssh, c2_ssh, #L_ssh, T_ssh,
                             ucur_WP_mean, #ucur_CP_mean, ucur_EP_mean,
                             taux_WP_mean, #taux_CP_mean, taux_EP_mean
                             ), axis=1)

scaler = StandardScaler()
Xorg = scaler.fit_transform(feature_unscaled)

Xorg = np.nan_to_num(Xorg)

X = Xorg[:-lead_time-shift,:]
futureX = Xorg[-lead_time-shift-time_lag:,:]

X = include_time_lag(X, max_lag=time_lag)
futureX =  include_time_lag(futureX, max_lag=time_lag)

yorg = nino34.values
y = yorg[lead_time + time_lag + shift:]

timey = nino34.index[lead_time + time_lag + shift:]
futuretime = pd.date_range(start='2019-01-01',
                                        end=pd.to_datetime('2019-01-01')+pd.tseries.offsets.MonthEnd(lead_time+shift),
                                        freq='MS')

test_indeces = (timey>='2002-01-01') & (timey<='2011-12-01')
#test_indeces = (timey>='2012-01-01') & (timey<='2018-12-01')
train_indeces = np.invert(test_indeces)

trainX, trainy, traintimey = X[train_indeces,:], y[train_indeces], timey[train_indeces]
testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

#%% =============================================================================
# Deep ensemble
# =============================================================================
model = DEM()

model.set_parameters(layers=1, dropout=0.2, noise=0.2, l1_hidden=0.0,
            l2_hidden=0.2, l1_mu=0., l2_mu=0.2, l1_sigma=0., l2_sigma=0.2,
            lr=0.001, batch_size=1, epochs=500, n_segments=3, n_members_segment=1, patience=30, verbose=1, std=True)

#model.get_pretrained_weights(location=modeldir, dir_name=f'pre_ensemble_lead{lead_time}')

model.fit(trainX, trainy, testX, testy, use_pretrained=False)


#%%
pred_mean, pred_std = model.predict(testX)

score = model.evaluate(testy, pred_mean, pred_std)
print_header(f"Score: {score}")

if model.std:
    ens_dir=f'ensemble_lead{lead_time}'
else:
    ens_dir=f'simple_ensemble_lead{lead_time}'

model.save(location=modeldir, dir_name=ens_dir)


#%% =============================================================================
# Plots
# =============================================================================
plt.close("all")

# Scores during trianing
plt.subplots()

for h in model.history:
    plt.plot(h.history[f'val_{model.loss_name}'], c='k')
    plt.plot(h.history[f'{model.loss_name}'], c='r')

plt.plot(np.nan, c='k', label='test')
plt.plot(np.nan, c='r', label='train')
plt.legend()

#%% just for testing the loading function delete and load the model
plt.close("all")
del model
model = DEM()
model.load(location=modeldir, dir_name=ens_dir)

pred_mean, pred_std = model.predict(testX)
predtrain_mean, predtrain_std = model.predict(trainX)
predfuture_mean, predfuture_std =  model.predict(futureX)

# Predictions
plt.subplots(figsize=(15,3.5))
plt.axhspan(-0.5,
            -6,
            facecolor='blue',
            alpha=0.1,zorder=0)

plt.axhspan(0.5,
            6,
            facecolor='red',
            alpha=0.1,zorder=0)

plt.xlim(timey[0],futuretime[-1])
plt.ylim(-3,3)

# test
plot_prediction(testtimey, pred_mean, std=pred_std, facecolor='royalblue', line_color='navy')

# train
#predtrain_mean[traintimey=='2002-02-01'] = np.nan
plot_prediction(traintimey, predtrain_mean, std=predtrain_std, facecolor='lime', line_color='g')

# future
plot_prediction(futuretime, predfuture_mean, std=predfuture_std, facecolor='orange', line_color='darkorange')

# observation
plt.plot(timey, y, "k")

pred_rmse = round(rmse(testy, pred_mean),2)
plt.title(f"Lead time: {lead_time} month, RMSE (of mean): {pred_rmse}")
plt.grid()
plt.xlabel('Year')
plt.ylabel('NINO3.4 [K]')


# Seaonality of Standard deviations
plt.subplots()

xr_nino34 = xr.DataArray(nino34)
std_data = xr_nino34.groupby('time.month').std(dim='time')

pd_pred_std = pd.Series(data=pred_std, index = testtimey)
xr_pred_std = xr.DataArray(pd_pred_std)
std_pred = xr_pred_std.groupby('time.month').mean(dim='time')

std_data.plot()
std_pred.plot()


# plot explained variance
plot_correlation(testy, pred_mean, testtimey - pd.tseries.offsets.MonthBegin(1), title="")


# Error distribution
plt.subplots()
plt.title("Error distribution")
error = pred_mean - testy

plt.hist(error, bins=16)

#%%
in_or_out = np.zeros((len(pred_mean)))
in_or_out[(testy>pred_mean-pred_std) & (testy<pred_mean+pred_std)] = 1
in_frac_1std = np.sum(in_or_out)/len(testy)

in_or_out = np.zeros((len(pred_mean)))
in_or_out[(testy>pred_mean-2*pred_std) & (testy<pred_mean+2*pred_std)] = 1
in_frac_2std = np.sum(in_or_out)/len(testy)


"""
Archived:
#in_or_out = np.zeros((len(pred_mean)))
#in_or_out[(testy>predicty_m1std) & (testy<predicty_p1std)] = 1
#in_frac = np.sum(in_or_out)/len(testy)
#
#in_or_out_train = np.zeros((len(predtrain_mean)))
#in_or_out_train[(trainy>predicttrainy_m1std) & (trainy<predicttrainy_p1std)] = 1
#in_frac_train = np.sum(in_or_out_train)/len(trainy)
#plt.title(f"train:{round(in_frac_train,2)*100}%, test:{round(in_frac*100,2)}%, RMSE (of mean): {pred_rmse}")
"""