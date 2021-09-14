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
import skimage.measure as measure
import skimage.io as io
import scipy
import os

MasterFolder='/media/patrice/DataDrive/SEE_ICE/FullData_Revision/' #save all run outputs here and nothing else.
VMfolder='/media/patrice/DataDrive/SEE_ICE/Validate/VMrasters/'
OutputName = '/media/patrice/DataDrive/SEE_ICE/FinalDataFrame_valley.csv' #where data will be saved


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
    
def GetVMerror(VMimage, ClassImage):
       

    PredictedRocks=np.logical_or(ClassImage==6, ClassImage==7)
    RockContours=measure.find_contours(1*PredictedRocks, level=0.5)
    GlacierContour=np.zeros((VMimage.shape))
    for c in range(len(RockContours)):
        Contour=np.int16(RockContours[c])
        if Contour.shape[0]>100:
            for p in range(Contour.shape[0]):
                GlacierContour[Contour[p,0], Contour[p,1]]=1
           
    #start the comparison
    Xp,Yp=np.where(GlacierContour==1)
    Xm,Ym=np.where(VMimage==1)
    Xp=Xp.reshape(-1,1)
    Yp=Yp.reshape(-1,1)
    Xm=Xm.reshape(-1,1)
    Ym=Ym.reshape(-1,1)
   
    XM=np.concatenate((Xm,Ym), axis=1)
    XP=np.concatenate((Xp,Yp), axis=1)
    D=scipy.spatial.distance.cdist(XP, XM, metric='euclidean')
    
    return 10*np.min(D, axis=0)
    

tic()
RunList=glob.glob(MasterFolder+'*/')
DataArray=np.zeros((4*len(RunList),8))# columns: 1:F1 2:patch 3:tile size 4:bands 5:joint/single training 6:CSC phase 7:Glacier
DataFrame=pd.DataFrame(DataArray, columns=['Error', 'Error_Type','Patch_Size','Tile_Size', 'Bands', 'CNN_Training', 'Glacier', 'Validation'])
DataPoint=0
for j in range(len(RunList)):
    folder=RunList[j]
    results_files=glob.glob(folder+'*_classified.tif')
    
    MasterResults=np.zeros(1)
    for r in range(len(results_files)):
        ClassImage=io.imread(results_files[r])
        VM_name=VMfolder+'VM_'+os.path.basename(results_files[r])[0:-15]+'.tif'
        VMimage=io.imread(VM_name)
        data=GetVMerror(VMimage, ClassImage)
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
