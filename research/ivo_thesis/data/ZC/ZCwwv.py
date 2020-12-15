#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:52:51 2020

@author: ivo
"""

"""
TODO:
    1. figure out how to calculate wwv from thermocline depth as proxy
    2. calculate wwv from h 


"""

from os.path import join
from ninolearn.pathes import processeddir, rawdir

from os.path import join, exists
import xarray as xr
import pandas as pd

# from ninolearn.pathes import processeddir
# from ninolearn.utils import generateFileName, small_print_header
# from ninolearn.preprocess.anomaly import _get_period, computeMeanClimatology
# from ninolearn.preprocess.prepare import
from ninolearn.IO.read_processed import data_reader
# from ninolearn.IO.read_raw import oni
# from ninolearn.IO.read_raw import nino34detrend_anom
from ninolearn.download import download, sources
from ninolearn.pathes import processeddir, rawdir

# import pandas as pd
import numpy as np
# import xarray as xr
import xesmf as xe
import math


