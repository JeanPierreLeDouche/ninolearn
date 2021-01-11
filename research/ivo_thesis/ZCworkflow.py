#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:02:10 2020

@author: ivo
"""

from ninolearn.IO.read_raw import ZC_raw, ZC_h, ZC_oni
from ninolearn.preprocess.prepare import prep_nms
from ninolearn.plot.ZC_dem_plots import nms_plots


""" specify version of ZC data here which will be used in naming the produced
datasets:
"""
version = 'mu28'
# version = 'undistorted'


### read raw ZC data and save to 1x1 grid file in processeddir
### also makes field of h and sst
ZC_raw(version)

## calculates monthly averaged (?) fields of thermocline height within region 
## of interest
ZC_h(version)

## cacluate ONI in region of interest:
ZC_oni(version)

## calculate network metrics from sst (Henk's suggestion) or thermocline height (like Paul)
prep_nms(version)


################## POST PROCESSING ############################################

nms_plots(version)


