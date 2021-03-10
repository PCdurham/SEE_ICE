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
import pandas as pd
import statistics


MasterFolder='/media/patrice/DataDrive/SEE_ICE/FullData_Revision/' #save all run outputs here and nothing else.
OutputName = '/media/patrice/DataDrive/SEE_ICE/FinalDataFrame_calving.csv' #where data will be saved


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
        
def GetGlacier(folder):
    if 'Hel' in folder:
        return 'Helheim'
    elif 'Jak' in folder:
        return 'Jakobshavn'
    elif 'Sto' in folder:
        return 'Store'
    
def GetValidation(folder):
    if 'Hel' in folder:
        return 'Seen'
    elif 'Jak' in folder:
        return 'Unseen'
    elif 'Sto' in folder:
        return 'Unseen'
    
def GetPatch(folder):
    if 'patch1/' in folder:
        return 1
    elif 'patch3/' in folder:
        return 3
    elif 'patch5/' in folder:
        return 5
    elif 'patch7/' in folder:
        return 7
    elif 'patch15/' in folder:
        return 15
    
def GetTileSize(folder):
    if '_100_' in folder:
        return 100
    elif '_50_' in folder:
        return 50
    elif '_75_' in folder:
        return 75
    
def GetBands(folder):
    if 'NIR' in folder:
        return 'NIR_RGB'
    else:
        return 'RGB'

def GetTraining(folder):
    if 'joint' in folder:
        return 'Joint'
    else:
        return 'Single'

tic()
RunList=glob.glob(MasterFolder+'*/')
DataArray=np.zeros((4*len(RunList),8))# columns: 1:F1 2:patch 3:tile size 4:bands 5:joint/single training 6:CSC phase 7:Glacier
DataFrame=pd.DataFrame(DataArray, columns=['Error', 'Error_Type','Patch_Size','Tile_Size', 'Bands', 'CNN_Training', 'Glacier', 'Validation'])
DataPoint=0
for j in range(len(RunList)):
    folder=RunList[j]
    results_files=glob.glob(folder+'*.npy')
    
    MasterResults=np.zeros(1)
    for r in range(len(results_files)):
        data=np.load(results_files[r])
        MasterResults=np.concatenate((MasterResults,data), axis=0)
        
    MasterResults=MasterResults[1:]
    
    
    
    #start filling out the data array

    DataFrame['Error'][DataPoint]=statistics.mode(MasterResults)
    DataFrame['Patch_Size'][DataPoint]=GetPatch(folder)
    DataFrame['Tile_Size'][DataPoint]=GetTileSize(folder)
    DataFrame['Bands'][DataPoint]=GetBands(folder)
    DataFrame['CNN_Training'][DataPoint]=GetTraining(folder)
    DataFrame['Glacier'][DataPoint]=GetGlacier(folder)
    DataFrame['Validation'][DataPoint]=GetValidation(folder)
    DataFrame['Error_Type'][DataPoint]='Mode'
    DataPoint+=1
    DataFrame['Error'][DataPoint]=np.median(MasterResults)
    DataFrame['Patch_Size'][DataPoint]=GetPatch(folder)
    DataFrame['Tile_Size'][DataPoint]=GetTileSize(folder)
    DataFrame['Bands'][DataPoint]=GetBands(folder)
    DataFrame['CNN_Training'][DataPoint]=GetTraining(folder)
    DataFrame['Glacier'][DataPoint]=GetGlacier(folder)
    DataFrame['Validation'][DataPoint]=GetValidation(folder)
    DataFrame['Error_Type'][DataPoint]='Median'
    DataPoint+=1
    DataFrame['Error'][DataPoint]=np.mean(MasterResults)
    DataFrame['Patch_Size'][DataPoint]=GetPatch(folder)
    DataFrame['Tile_Size'][DataPoint]=GetTileSize(folder)
    DataFrame['Bands'][DataPoint]=GetBands(folder)
    DataFrame['CNN_Training'][DataPoint]=GetTraining(folder)
    DataFrame['Glacier'][DataPoint]=GetGlacier(folder)
    DataFrame['Validation'][DataPoint]=GetValidation(folder)
    DataFrame['Error_Type'][DataPoint]='Mean'
    DataPoint+=1
    DataFrame['Error'][DataPoint]=np.std(MasterResults)
    DataFrame['Patch_Size'][DataPoint]=GetPatch(folder)
    DataFrame['Tile_Size'][DataPoint]=GetTileSize(folder)
    DataFrame['Bands'][DataPoint]=GetBands(folder)
    DataFrame['CNN_Training'][DataPoint]=GetTraining(folder)
    DataFrame['Glacier'][DataPoint]=GetGlacier(folder)
    DataFrame['Validation'][DataPoint]=GetValidation(folder)
    DataFrame['Error_Type'][DataPoint]='St.Dev.'
    DataPoint+=1
    
DataFrame.to_csv(OutputName)

    
    

    

        

        


toc()