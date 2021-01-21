#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivo
"""

from ninolearn.IO.read_raw import ZC_raw, ZC_h, ZC_oni, ZC_simple_read
from ninolearn.preprocess.prepare import prep_nms
from ninolearn.plot.ZC_dem_plots import nms_plots, oni_plots
import pandas as pd
import numpy as np

""" specify version of ZC data here which will be used in naming the produced
datasets:
"""
version = 'undistorted'

times = np.unique(ZC_simple_read(version)['time'])

t_start = times[0] + pd.Timedelta((2*365 + 90),'D')
t_end = times[-1] - pd.Timedelta(90,'D')

print(f'tstart = {t_start} and tend = {t_end}')

#%%
### read raw ZC data and save to 1x1 grid file in processeddir
### also makes field of h and sst
ZC_raw(version)

## calculates monthly averaged (?) fields of thermocline height within region 
## of interest
ZC_h(version)

## cacluate ONI in region of interest:
ZC_oni(version)

## calculate network metrics from sst (Henk's suggestion) or thermocline height (like Paul)
prep_nms(version, 0.99, t_start, t_end)

################## POST PROCESSING ############################################
nms_plots(version)

#%%

oni_plots(version)

