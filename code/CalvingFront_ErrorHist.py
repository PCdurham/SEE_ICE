#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 10:06:36 2021

@author: patrice
"""
import numpy as np
import glob
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
import os



ScorePath1 = '/media/patrice/DataDrive/SEE_ICE/VGG16_50_RGBNIR_fp16_kernel3/'

DatList=glob.glob(ScorePath1+'*.npy')

MasterData=np.zeros(1)
for d in range(0, len(DatList)):
    data=np.load(DatList[d])
    if np.max(data)>1000:
        print('Warning: catastrophic error in '+ os.path.basename(DatList[d]))
        plt.figure()
        sns.histplot(data=data, binwidth=(10))
        plt.title('Catastrophic error for '+os.path.basename(DatList[d]))
    else:
    
        MasterData=np.concatenate((MasterData,data), axis=0)
        #plt.figure()
        #sns.histplot(data=MasterData, binwidth=(10))

MasterData=MasterData[1:]
plt.figure()   
sns.histplot(data=MasterData, binwidth=(10))
plt.title('master')

print('Modal error= '+str(int(statistics.mode(MasterData))))
print('median error '+str(int(np.median(MasterData))))
print('mean error '+str(int(np.median(MasterData))))
print('stdev error '+str(int(np.std(MasterData))))