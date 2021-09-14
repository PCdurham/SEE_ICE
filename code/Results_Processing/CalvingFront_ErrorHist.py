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



ScorePath1 = '/media/patrice/DataDrive/SEE_ICE/FullData_Revision/Jak_VGG16_joint_50_RGBNIR_fp16_patch7/'

DatList=glob.glob(ScorePath1+'*.npy')

MasterData=np.zeros(1)
for d in range(0, len(DatList)):
    data=np.load(DatList[d])
    if np.max(data)>1000:
        print('Warning: catastrophic error in '+ os.path.basename(DatList[d]))
        plt.figure()
        sns.histplot(data=data, binwidth=(10))
        plt.title('Catastrophic error for '+os.path.basename(DatList[d]))
        MasterData=np.concatenate((MasterData,data), axis=0)
    else:
    
        MasterData=np.concatenate((MasterData,data), axis=0)
        #plt.figure()
        #sns.histplot(data=MasterData, binwidth=(10))

MasterData3=MasterData[1:]
plt.figure()   
#n_bins=np.asarray(range(0,510,10))

#sns.histplot(data=np.clip(MasterData, n_bins[0], n_bins[-1]), bins=n_bins)
sns.histplot(MasterData, binwidth=10)
plt.ylabel('Helheim')
plt.xlabel('Error[m]')

print('Modal error= '+str(int(statistics.mode(MasterData))))
print('median error '+str(int(np.median(MasterData))))
print('mean error '+str((np.mean(MasterData))))
print('stdev error '+str((np.std(MasterData))))
betterthan100=int(100*np.sum(1*MasterData<100)/len(MasterData))
print(str(betterthan100),'% of data below 100m error')
