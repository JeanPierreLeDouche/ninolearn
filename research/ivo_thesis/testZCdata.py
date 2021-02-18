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


### user inputs here 
# versions= ['mu25v2', 'mu22v2' , 'mu19v2', 'mu28v2', 'mu31v2', 'mu15v2', 'mu11v2', 'mu50v2', 'mu04v2', 'mu70v2', 'mu01', 'de12case1', 'de08case1'] #, 'mu20t2']
versions = ['mu25v2', 'de12hires', 'de08hires']

versions.sort()
individual_plots = False
compare_versions = True
###

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
                ls = 'solid'
                lw = 2
            elif version[0:2] == 'de':
                ls = 'dotted'
                lw = 3
            else:
                print('ERROR: No linestyle selected from version name')
            
            plt.plot(ONI.index, ONI, label = plot_name(version), ls = ls, lw = lw)
            plt.legend()
            
            plt.xlabel('time', fontdict = font)
            plt.ylabel('ONI', fontdict = font)
            plt.title(r'Several runs of ZC87 with different parameter values', fontdict = font)
        print('saving parameter comparison')
        plt.savefig(join(plotdir, 'parameter_comparison'))
            
diagnostics = ZConi_evaluate(versions, individual_plots, compare_versions)

        
        