#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:39:06 2021

@author: patrice

goes through a folder of results and gets the F1 from saved npz files
"""
import glob
import numpy as np
from sklearn import metrics

folder='/media/patrice/DataDrive/SEE_ICE/Jak_VGG16_100_RGBNIR_fp16_patch7/'
subsample=1000000 #use 0 for all points, else give number of points


def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(int(time.time() - startTime_for_tictoc)) + " seconds.")
    else:
        print ("Toc: start time not set")

tic()

results_files=glob.glob(folder+'*.npz')

MasterResults=np.zeros((1,3))
for r in range(len(results_files)):
    datanpz=np.load(results_files[r])
    data=datanpz['arr_0']
    MasterResults=np.concatenate((MasterResults, data), axis=0)
    
MasterResults=MasterResults[1:]

Truth=MasterResults[:,0]
CNN=MasterResults[:,1]
CSC=MasterResults[:,2]

CNN=CNN[Truth!=0]#clear class 0 in truth data
CSC=CSC[Truth!=0]
Truth=Truth[Truth!=0]

Truth=Truth[CNN!=0]#clear border 0 classes
CSC=CSC[CNN!=0]
CNN=CNN[CNN!=0]

Truth=Truth[CSC!=0]
CNN=CNN[CSC!=0]
CSC=CSC[CSC!=0]

if subsample>0:
    idx = np.random.choice(np.arange(len(Truth)), subsample, replace=False)
    Truth=Truth[idx]
    CNN=CNN[idx]
    CSC=CSC[idx]
    


reportCSC = metrics.classification_report(Truth, CSC, digits = 3)
reportCNN = metrics.classification_report(Truth, CNN, digits = 3)

print('CNN tiled classification results for ' + folder)
print(reportCNN)
print('\n')
print('CNN-Supervised classification results for ' + folder)
print(reportCSC)
print('\n')

toc()