"""
This module aims to standardize the training and evaluation procedure.
"""
import numpy as np
import pandas as pd
import xarray as xr

from os.path import join, exists
from os import listdir

from ninolearn.utils import print_header, small_print_header
from ninolearn.pathes import modeldir, processeddir

# evaluation decades
# decades = [1963, 1972, 1982, 1992, 2002, 2012, 2018]
decades = [1953, 1962, 1972, 1982, 1992] # boundaries for ZC decades 
decades_elninolike = []

n_decades = len(decades)

# lead times for the evaluation
lead_times = [0, 3, 6, 9, 12, 15, 18, 21]
n_lead = len(lead_times)
    
decade_color = ['orange', 'violet', 'limegreen' ]#, 'darkgoldenrod', 'red', 'royalblue']
decade_name = ['1953-1962', '1962-1971', '1972-1981', '1982-1991'] #, '1992-2001', '2002-2011', '2012-2017']


def cross_training(model, pipeline, n_iter, modelname, **kwargs):
    """
    Training the model on different training sets in which each time a period\
    corresponding to a decade out of 1962-1971, 1972-1981, ..., 2012-last \
    ovserved date is spared.

    :param model: A model that follows the guidelines how a model object\
    should be set up.py

    :param pipeline: a function that takes lead time as argument and returns\
    the corresponding feature, label, time and persistance.

    :param save_dir: The prefix of the save directory.

    :param **kwargs: Arguments that shall be passed to the .set_parameter() \
    method of the provided model.
    """


    for lead_time in lead_times:
        X, y, timey = pipeline(lead_time) #, return_persistance=False)

        print_header(f'Lead time: {lead_time} month')

        for j in range(n_decades-1):
            m = model(**kwargs)
            m.hyperparameters['name'] = modelname
            dir_name = f"{m.hyperparameters['name']}_decade{decades[j]}_lead{lead_time}"
            path = join(modeldir, dir_name)

            n_files=0
            if exists(path):
                n_files = len(listdir(path))

            if not exists(path) or n_files==0:
                small_print_header(f'Test period: {decades[j]}-01-01 till {decades[j+1]-1}-12-01')

                test_indeces = (timey>=f'{decades[j]}-01-01') & (timey<=f'{decades[j+1]-1}-12-01')
                train_indeces = np.invert(test_indeces)
                trainX, trainy, traintime = X[train_indeces,:], y[train_indeces], timey[train_indeces]
                # print(trainX)
                m.fit_RandomizedSearch(trainX, trainy, traintime, n_iter=n_iter)
                m.save(location=modeldir, dir_name=dir_name)

            else:
                print(f'{dir_name} already exists')
            del m

def cross_hindcast(model, pipeline, model_name, **kwargs):
    """
    Generate a hindcast from 1962 till today using the models which were
    trained by the .cross_training() method.

    :param model: The considered model.
http://localhost:8888/notebooks/Documents/GitHub/ninolearn/docs-sphinx/source/jupyter_notebook_tutorials/StandardizedResearch.ipynb#Cross-train-the-model
    :param pipeline: The data pipeline that already was used before in \
    .cross_training().
    """

    first_lead_loop = True
    print(f'decades: {decades}')
    for i in range(n_lead):
        lead_time = lead_times[i]
        print_header(f'Lead time: {lead_time} months')

        X, y, timey = pipeline(lead_time)

        ytrue = np.array([])
        timeytrue = pd.DatetimeIndex([])

        first_dec_loop = True
        for j in range(n_decades-1):
            small_print_header(f'Predict: {decades[j]}-01-01 till {decades[j+1]-1}-12-01')

            # test indices
            test_indeces = (timey>=f'{decades[j]}-01-01') & (timey<=f'{decades[j+1]-1}-12-01')
            testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

            m = model(**kwargs)
            m.load(location=modeldir, dir_name=f'{model_name}_decade{decades[j]}_lead{lead_time}')

            # allocate arrays and variables for which the model must be loaded
            if first_dec_loop:
                n_outputs = m.n_outputs

                output_names = m.output_names
                pred_full = np.zeros((n_outputs, 0))
                first_dec_loop=False

            # make prediction
            pred = np.zeros((m.n_outputs, testX.shape[0]))
            pred[:,:] = m.predict(testX)


            # make the full time series
            pred_full = np.append(pred_full, pred, axis=1)
            ytrue = np.append(ytrue, testy)
            timeytrue = timeytrue.append(testtimey)
            del m
        ### IG: following code only relevant for real measurement timeseries
        
        # if timeytrue[0]!=pd.to_datetime('1953-01-01'): # IG: changed 1963 to 1953
        #     expected_first_date = '1953-01-01'
        #     got_first_date = timeytrue[0].isoformat()[:10]

        #     raise Exception(f"The first predicted date for lead time {lead_time} \
        #                     is {got_first_date} but expected {expected_first_date}")

        # allocate arrays and variables for which the full length of the time
        # series must be known
        if first_lead_loop:
            n_time = len(timeytrue)
            pred_save =  np.zeros((n_outputs, n_time, n_lead))
            first_lead_loop=False
        pred_save[:,:,i] =  pred_full

    # Save data to a netcdf file
    save_dict = {}
    for i in range(n_outputs):
        save_dict[output_names[i]] = (['target_season', 'lead'],  pred_save[i,:,:])

    ds = xr.Dataset(save_dict, coords={'target_season': timeytrue,
                                       'lead': lead_times} )
    ds.to_netcdf(join(processeddir, f'{model_name}_forecasts.nc'), mode = 'w')
    # dsmean = ds['mean']
    # print(f'shape of forecasts.nc file is: {dsmean.shape}')
    ds.close() 


def cross_hindcast_dem(model, pipeline, model_name):
    """
    Generate a hindcast from 1962 till today using the models which were
    trained by the .cross_training() method. ONLY works for the DEM.
    This routine returns an std estimate that is only based on the corrlation
    skill of the DEM predicted mean.

    :param model: The considered model.

    :param pipeline: The data pipeline that already was used before in \
    .cross_training().
    """
    cross_hindcast(model, pipeline, model_name)
    
    ### TODO: unpack the std part from dem_forecasts.nc because this code is deprecated
    std_estimate = xr.open_dataset(join(processeddir, f'{model_name}_forecasts.nc') )['std']
    # std_estimate = xr.open_dataarray(join(processeddir, f'{model_name}_std_estimate.nc'))

    first_lead_loop = True
    print(f'decades: {decades}')
    for i in range(n_lead):
        lead_time = lead_times[i]
        print_header(f'Lead time: {lead_time} months')

        X, y, timey = pipeline(lead_time)

        ytrue = np.array([])
        timeytrue = pd.DatetimeIndex([])

        first_dec_loop = True
        for j in range(n_decades-1):
            small_print_header(f'Predict: {decades[j]}-01-01 till {decades[j+1]-1}-12-01')

            # test indices
            test_indeces = (timey>=f'{decades[j]}-01-01') & (timey<=f'{decades[j+1]-1}-12-01')
            testX, testy, testtimey = X[test_indeces,:], y[test_indeces], timey[test_indeces]

            m = model()
            m.load(location=modeldir, dir_name=f'{model_name}_decade{decades[j]}_lead{lead_time}')

            # allocate arrays and variables for which the model must be loaded
            if first_dec_loop:
                n_outputs = m.n_outputs
                output_names = m.output_names
                pred_full = np.zeros((n_outputs+1, 0))
                first_dec_loop=False

            # make prediction
            pred = np.zeros((m.n_outputs+1, testX.shape[0]))
            pred[:2,:] = m.predict(testX)

            for k in range(len(testtimey)):
                month = testtimey[k].date().month
                pred[-1, k] = std_estimate[month-1, i] #IG: swapped indices 'month-1' and i for use with forecasts.nc

            # make the full time series
            pred_full = np.append(pred_full, pred, axis=1)
            ytrue = np.append(ytrue, testy)
            timeytrue = timeytrue.append(testtimey)
            del m

        # if timeytrue[0]!=pd.to_datetime('1963-01-01'):
        #     expected_first_date = '1963-01-01'
        #     got_first_date = timeytrue[0].isoformat()[:10]

        #     raise Exception(f"The first predicted date for lead time {lead_time} \
        #                     is {got_first_date} but expected {expected_first_date}")

        # allocate arrays and variables for which the full length of the time
        # series must be known
        if first_lead_loop:
            n_time = len(timeytrue)
            pred_save =  np.zeros((n_outputs+1, n_time, n_lead))
            first_lead_loop=False

        pred_save[:,:,i] =  pred_full

    # Save data to a netcdf file
    save_dict = {}
    for i in range(n_outputs + 1):
        if i<n_outputs:
            save_dict[output_names[i]] = (['target_season', 'lead'],  pred_save[i,:,:])
        else:
            save_dict['std_estimate'] = (['target_season', 'lead'],  pred_save[i,:,:])

    ds = xr.Dataset(save_dict, coords={'target_season': timeytrue,
                                       'lead': lead_times} )
    ds.to_netcdf(join(processeddir, f'{model_name}_forecasts_with_std_estimated.nc'))
    ds.close()
