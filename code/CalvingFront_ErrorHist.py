#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:06:36 2021

@author: patrice
"""
import numpy as np
import glob
import seaborn as sns



ScorePath1 = '/media/patrice/DataDrive/SEE_ICE/TestOutputDenseNet121/'

DatList=glob.glob(ScorePath1+'*.npy')

MasterData=np.load(DatList[0])
for d in range(1, len(DatList)):
    data=np.load(DatList[0])
    MasterData=np.concatenate((MasterData,data), axis=0)
    
    
sns.histplot(data=MasterData, binwidth=(10))

e95=np.percentile(MasterData,95)