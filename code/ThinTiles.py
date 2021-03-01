#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:41:24 2021

@author: patrice
"""

import os
import glob
import numpy as np

DataFolder = '/media/patrice/DataDrive/SEE_ICE/JointTrain/' 
RemainingTiles=5000

for f in range(1,8):
    ilist=glob.glob(DataFolder+'C'+str(f)+'/*.tif')
    if len(ilist)>RemainingTiles:
        idx = np.random.choice(np.arange(len(ilist)), len(ilist)-RemainingTiles, replace=False)
        for i in range(len(idx)):
            os.remove(ilist[idx[i]])