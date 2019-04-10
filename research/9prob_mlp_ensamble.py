# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers.core import Lambda

from sklearn.preprocessing import MinMaxScaler

from ninolearn.IO.read_post import data_reader
from ninolearn.plot.evaluation  import plot_explained_variance
from ninolearn.learn.evaluation import nrmse
from ninolearn.learn.mlp import include_time_lag
from ninolearn.learn.losses import nll_gaussian
K.clear_session()

# =============================================================================
# #%% read data
# =============================================================================
reader = data_reader(startdate='1981-01')

nino4 = reader.read_csv('nino4M')
nino34 = reader.read_csv('nino3.4M')
nino12 = reader.read_csv('nino1+2M')
nino3 = reader.read_csv('nino3M')

len_ts = len(nino34)
sc = np.cos(np.arange(len_ts)/12*2*np.pi)
yr =  np.arange(len_ts) % 12
yr3 = np.arange(len_ts) % 36
yr4 = np.arange(len_ts) % 48
yr5 = np.arange(len_ts) % 60


wwv = reader.read_csv('wwv')
network = reader.read_statistic('network_metrics', variable='air',
                           dataset='NCEP', processed="anom")

network_ssh = reader.read_statistic('network_metrics', variable='sshg',
                           dataset='GODAS', processed="anom")

pca_air = reader.read_statistic('pca', variable='air',
                           dataset='NCEP', processed="anom")
pca_u = reader.read_statistic('pca', variable='uwnd',
                           dataset='NCEP', processed="anom")
pca_v = reader.read_statistic('pca', variable='vwnd',
                           dataset='NCEP', processed="anom")


c2 = network['fraction_clusters_size_2']
c3 = network['fraction_clusters_size_3']
c5 = network['fraction_clusters_size_5']
S = network['fraction_giant_component']
H = network['corrected_hamming_distance']
T = network['global_transitivity']
C = network['avelocal_transmissivity']
L = network['average_path_length']
nwt = network['threshold']
pca1_air = pca_air['pca1']
pca2_air = pca_air['pca2']
pca3_air = pca_air['pca3']
pca1_u = pca_u['pca1']
pca2_u = pca_u['pca2']
pca3_u = pca_u['pca3']
pca1_v = pca_v['pca1']
pca2_v = pca_v['pca2']
pca3_v = pca_v['pca3']

c2ssh = network_ssh['fraction_clusters_size_2']

#%% =============================================================================
# # process data
# =============================================================================
time_lag = 2
lead_time = 3
train_frac = 0.7
feature_unscaled = np.stack((nino34.values, #c2ssh.values, # nino12.values , nino3.values, nino4.values,
                             wwv.values, sc,   #yr # nwt.values#, c2.values,c3.values, c5.values,
#                            S.values, H.values, T.values, C.values, L.values,
#                            pca1_air.values, pca2_air.values, pca3_air.values,
#                             pca1_u.values, pca2_u.values, pca3_u.values,
#                             pca1_v.values, pca2_v.values, pca3_v.values
                             ), axis=1)


scaler = MinMaxScaler(feature_range=(-1,1))
Xorg = scaler.fit_transform(feature_unscaled)

X = Xorg[:-lead_time,:]
futureX = Xorg[-lead_time-time_lag:,:]

X = include_time_lag(X, max_lag=time_lag)
futureX =  include_time_lag(futureX, max_lag=time_lag)


yorg = nino34.values
y = yorg[lead_time + time_lag:]

timey = nino34.index[lead_time + time_lag:]
futuretime = pd.date_range(start='2019-01-01',
                                        end=pd.to_datetime('2019-01-01')+pd.tseries.offsets.MonthEnd(lead_time),
                                        freq='MS')


train_end = int(train_frac * X.shape[0])
trainX, testX = X[:train_end,:], X[train_end:,:]
trainy, testy= y[:train_end], y[train_end:]
traintimey, testtimey = timey[:train_end], timey[train_end:]

#%% =============================================================================
# # neural network
# =============================================================================

n_ens = 10
model_ens = []
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0, amsgrad=False)
es = EarlyStopping(monitor='val_loss',
                              min_delta=0.0,
                              patience=20,
                              verbose=0, mode='auto')
for i in range(n_ens):
    model_ens.append(Sequential())

    model_ens[i].add(Dense(8, input_dim=X.shape[1],activation='relu', kernel_regularizer=regularizers.l1_l2(0.1,0.1)))
    model_ens[i].add(Dense(2, activation='linear'))

    model_ens[i].compile(loss=nll_gaussian, optimizer=optimizer)

    history = model_ens[i].fit(trainX, trainy, epochs=300, batch_size=1,verbose=1,
                        shuffle=True, callbacks=[es],
                        validation_data=(testX, testy))



#%%
def mixture(pred):
    """
    returns the ensemble mixture results
    """
    mix_mean = pred[:,0,:].mean(axis=1)
    mix_var = np.mean(pred[:,0,:]**2 + pred[:,1,:]**2, axis=1)  - mix_mean**2
    mix_std = np.sqrt(mix_var)
    return mix_mean, mix_std



pred_ens = np.zeros((len(testy),2, n_ens))
predtrain_ens = np.zeros((len(trainy),2, n_ens))
predfuture_ens = np.zeros((lead_time,2, n_ens))

for i in range(n_ens):
    pred_ens[:,:,i] = model_ens[i].predict(testX)
    predtrain_ens[:,:,i] = model_ens[i].predict(trainX)
    predfuture_ens[:,:,i] = model_ens[i].predict(futureX)


pred_mean, pred_std = mixture(pred_ens)
predtrain_mean, predtrain_std = mixture(predtrain_ens)
predfuture_mean, predfuture_std = mixture(predfuture_ens)


#%% =============================================================================
# Plot
# =============================================================================
plt.close("all")
plt.subplots()
plt.plot(history.history['val_loss'],label = "val")
plt.plot(history.history['loss'], label= "train")
plt.legend()



plt.subplots(figsize=(12,4))


std = 1
predicty_max = pred_mean + std * np.abs(pred_std)
predicty_min = pred_mean - std * np.abs(pred_std)

plt.plot(testtimey,pred_mean, "b")
plt.fill_between(testtimey,predicty_min,predicty_max ,facecolor='blue', alpha=0.3)




predicttrainy_max = predtrain_mean + std * np.abs(predtrain_std)
predicttrainy_min = predtrain_mean - std * np.abs(predtrain_std)

plt.plot(traintimey, predtrain_mean, "lime")
plt.fill_between(traintimey,predicttrainy_min,predicttrainy_max ,facecolor='lime', alpha=0.3)

plt.plot(timey, y, "k")

in_or_out = np.zeros((len(pred_mean)))
in_or_out[(testy>predicty_min) & (testy<predicty_max)] = 1
in_frac = np.sum(in_or_out)/len(testy)

in_or_out_train = np.zeros((len(predtrain_mean)))
in_or_out_train[(trainy>predicttrainy_min) & (trainy<predicttrainy_max)] = 1
in_frac_train = np.sum(in_or_out_train)/len(trainy)

plt.title(f"train:{round(in_frac_train,2)*100}%, test:{round(in_frac,2)*100}%")