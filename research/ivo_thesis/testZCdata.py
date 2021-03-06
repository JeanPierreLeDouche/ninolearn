#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 16:45:01 2021

@author: ivo
"""

from ninolearn.IO.read_raw import ZC_raw, ZC_h, ZC_oni, ZC_simple_read
from ninolearn.plot.ZC_dem_plots import nms_plots, oni_plots
from os.path import join
from ninolearn.pathes import processeddir, plotdir

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time 

### user inputs here 
# versions= ['mu25v2', 'mu22v2' , 'mu19v2', 'mu28v2', 'mu31v2', 'mu15v2', 'mu11v2', 'mu50v2', 'mu04v2', 'mu70v2', 'mu01', 'de12case1', 'de08case1'] #, 'mu20t2']
# versions = ['mu25v2','mu25de10', 'mu24v3', 'mu27v3'] #  'de12hires', 'de08hires', 

# versions = ['mu24v3', 'mu16v3', 'mu19v3', 'mu21v3', 'mu27v3']
# versions = ['mu11v3', 'mu34v3', 'mu39v3']
# versions = ['mu255v3', 'mu54v3', 'mu49v3', 'mu44v3']
# versions = ['mu29v4', 'mu28v4', 'mu27v4', 'mu26v4', 'mu25v4', 'mu24v4', 'mu23v4', 'mu22v4']
# versions = ['mu28v4', 'mu27v4', 'mu26v4', 'mu25v4', 'mu24v4', 'mu23v4', 'mu22v4', 'mu29v4', 'mu30v4', 'mu31v4', 'mu32v4']
# versions = ['de08v4', 'de09v4', 'de11v4', 'de12v4'] ### wavespeed
# versions = [ 'ds01v4', 'ds02v4', 'ds04v4', 'ds05v4', 'ds06v4'] ### upwelling feedback  'ds00v4',
# versions = [ 'dt07v4',  'dt09v4', 'dt11v4', 'dt12v4', 'dt13v4', 'dt14v4', 'dt15v4'] ### sst damping  'dt06v4', 'dt08v4',
# versions = [ 'ds01v4', 'ds02v4', 'ds04v4'] # new case2 #'ds00v4',
# versions = ['de12v4', 'ds00v4', 'ds05v4', 'ds06v4', 'dt05v4', 'dt06v4', 'dt08v4']
# versions = ['de12v4', 'ds05v4', 'ds06v4']#, 'dt05v4', 'dt06v4', 'dt08v4']
# versions = ['de08ns', 'de08dsc', 'de12ns', 'de12dsc', 'ds01ns', 'ds01dsc', 'dt13ns', 'dt13dsc']

# versions = ['de08nosc', 'ds01nosc', 'dt13nosc', 'de12nosc']
# versions = ['de05', 'de20']
# versions = ['ta11', 'ta13', 'ta17', 'ta20', 'mu28v4']
# versions = [   'mu26ta50', 'mu28v4'] # 'mu26ta30', 'mu27ta40', 'mu28ta50',
# versions = ['mu26ta30ltr', 'mu27ta40ltr', 'mu28ta50ltr', 'mu26ta50ltr', 'mu28v4']
# versions = ['mu26ta70short', 'mu23ta40short', 'mu28v4'] # 'mu24ta50short', 'mu25ta60short',  # only mu23ta40 works
# versions = ['mu23ta40short', 'mu28v4']
# versions = [ 'mu25ta60v2', 'mu25ta90v2', 'mu22ta99v2', 'mu23ta90v2']# 'mu26ta70v2',

# versions = ['mu23ta20', 'mu23ta15v2', 'mu24ta15v2', 'mu25ta15v2', 'mu26ta15v2', 'mu24ta20','mu25ta20', 'mu26ta20']
versions = ['mu23lsr', 'mu23lsr2']

versions.sort()


individual_plots = False
compare_versions = True
today = str(datetime.date.today()).replace("-", "_")
plotname = 'amptesting4'
plotname += '_' + today 
###

# test data lengths
versions_filtered = []

for version in versions:
    data = ZC_simple_read(version)
    if all(data.dtypes != 'O'):
        versions_filtered.append(version)
        print('found other datatypes in data !')
    data_time = np.unique(data['time'])
    runtime = pd.to_datetime(data_time[-1]).year -  pd.to_datetime(data_time[0]).year
    print(f'Dataset version: {version}, time has length: {runtime} years')
    
versions = versions_filtered
print('filtered versions are:', versions)
         #%%

def ZConi_evaluate(versions, individual_plots = False, compare_versions = True): 
    
    """
    function for evaluating ZC87 outputs 
    
    versions: list of version names suffixed to the fort.149 output of the ZC87
    model so fort.149_{version}
    
    individual_plots: plots comparing the ONI of the ZC to ERSSTv5 ONI to give 
    an impression (to a human) of qualitatively similar behaviour
    
    comparison_plots: calculate the ONI for different versions of the ZC87 model
    and plot these together for comparison
    """
    
    
    font = {'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 20,
            }    
    
    def plot_name(ver_name):
        string = ver_name[0:2] + ' = ' + ver_name[2] + '.' + ver_name[3] + ' (' + ver_name[4:] + ')'
        if ver_name[0:2] == 'ta':
            string = 'tau = ' + ver_name[2] + '.' + ver_name[3] 
        if ver_name[0:2] == 'mu' and ver_name[4:6] == 'ta':     
            string = 'mu = ' + ver_name[2] + '.' + ver_name[3] + ' ' + ver_name[4:6] + ' = ' + ver_name[6] + '.' + ver_name[7]
        if ver_name[0:2] == 'mu' and ver_name[4:6] == 'ta' and ver_name[8:11] == 'ltr':
            string = 'tau = ' + ver_name[2] + '.' + ver_name[3] + ' ' + ver_name[4:6] + ' = ' + ver_name[6] + '.' + ver_name[7] + '('+ \
                ver_name[8:] + ')'
        return string
    
    ### calculate values
    
    for version in versions:
        data = ZC_simple_read(version)
        time = np.unique(data['time'])
        runtime = pd.to_datetime(time[-1]).year -  pd.to_datetime(time[0]).year
        print('-----------------------------------------------------------------')
        print(f'Dataset version: {version}, time has length: {runtime} years')
        print('-----------------------------------------------------------------')
                    
        ZC_raw(version)       
        ZC_oni(version)
    
    ### make individual plots 
    
    #defaults as follows:
    ls = 'solid'
    lw = 3
    
    
    if individual_plots == True:
        for version in versions:
            oni_plots(version)
    
    ### make plot comparing ZC runs with different mu
        
    if compare_versions == True:
        for version in versions:
                
            ONI_full = pd.read_csv(join(processeddir, ('oni_ZC_' + version + '.csv')))
            ONI = ONI_full['anom']
            ONI.index = pd.to_datetime(ONI_full['time'])
            
            if version[0:2] == 'mu':
                ls = 'dotted'
                lw = 2
            elif version[0:2] == 'de':
                # ls = 'dotted'
                ls = 'solid'
                lw = 3
            elif version[0:2] == 'ta': ls = 'solid'
            else:
                print('Warning: No linestyle selected from version name !')
            
            plt.plot(ONI.index, ONI, label = plot_name(version), ls = ls, lw = lw)
            plt.legend(fontsize = 20)
            
            plt.xlabel('time', fontdict = font)
            plt.ylabel('ONI', fontdict = font)
            plt.grid()
            
            plt.yticks(fontsize = 20)
            plt.xticks(fontsize = 20)
            plt.title(r'Several runs of ZC87 with different parameter values', fontdict = font)
        print('saving parameter comparison')
        plt.savefig(join(plotdir, f'parameter_comparison_{plotname}'))
            
diagnostics = ZConi_evaluate(versions, individual_plots, compare_versions)

        
        