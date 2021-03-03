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

MasterFolder='C:\\Users\\Melanie Marochov\\Documents\\2_MScR\\1_Revised_Results\\Revised_Results\\' #save all run outputs here and nothing else.
OutputName = 'C:\\Users\\Melanie Marochov\\Documents\\2_MScR\\1_Revised_Results\\Revised_Results_Data\\' #where data will be saved

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
        
def GetF1(report):
    lines = report.split('\n')
    for line in lines[0:-1]:
        if 'weighted' in line:
            dat = line.split(' ')
    
    return dat[17]

def GetGlacierF1(report):
    lines = report.split('\n')
    for line in lines[0:-1]:
        if '4.0' in lines:
            dat = line.split(' ')
    
    return dat[14]

def classification_report_csv(report, filename):
    report_data = []
    report = report.replace('avg', "")
    report = report.replace('accuracy', "Accuracy")
    report = report.replace('macro', "Macro_avg")
    report = report.replace('weighted', "Weighted_avg")
    
    lines = report.split("\n")
    no_empty_lines = [line for line in lines if line.strip()]
        
    for line in no_empty_lines[1:]:
        row = {}
        row_data = line.split(' ')
        row_data = list(filter(None, row_data))
        if 'Accuracy' in line:
            row_data.insert(1, 'NaN')
            row_data.insert(2, 'NaN')
            
        row['Class'] = row_data[0]
        row['Precision'] = (row_data[1])
        row['Recall'] = (row_data[2])
        row['F1_score'] = float(row_data[3])
        row['Support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index = False)

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
    if 'patch1' in folder:
        return 1
    elif 'patch3' in folder:
        return 3
    elif 'patch5' in folder:
        return 5
    elif 'patch7' in folder:
        return 7
    elif 'patch15' in folder:
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
DataArray=np.zeros((4*len(RunList),9))# columns: 1:F1 2:patch 3:tile size 4:bands 5:joint/single training 6:CSC phase 7:Glacier
DataFrame=pd.DataFrame(DataArray, columns=['F1', 'F1_Type','Patch_Size','Tile_Size', 'Bands', 'CNN_Training','CSC_Phase', 'Glacier', 'Validation'])
DataPoint=0
for j in range(len(RunList)):
    folder=RunList[j]
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
    
    #start filling out the data array

    DataFrame['F1'][DataPoint]=GetF1(reportCNN)
    DataFrame['Patch_Size'][DataPoint]=GetPatch(folder)
    DataFrame['Tile_Size'][DataPoint]=GetTileSize(folder)
    DataFrame['Bands'][DataPoint]=GetBands(folder)
    DataFrame['CNN_Training'][DataPoint]=GetTraining(folder)
    DataFrame['Glacier'][DataPoint]=GetGlacier(folder)
    DataFrame['Validation'][DataPoint]=GetValidation(folder)
    DataFrame['CSC_Phase'][DataPoint]=1
    DataFrame['F1_Type'][DataPoint]='All'
    DataPoint+=1
    DataFrame['F1'][DataPoint]=GetGlacierF1(reportCNN)
    DataFrame['Patch_Size'][DataPoint]=GetPatch(folder)
    DataFrame['Tile_Size'][DataPoint]=GetTileSize(folder)
    DataFrame['Bands'][DataPoint]=GetBands(folder)
    DataFrame['CNN_Training'][DataPoint]=GetTraining(folder)
    DataFrame['Glacier'][DataPoint]=GetGlacier(folder)
    DataFrame['Validation'][DataPoint]=GetValidation(folder)
    DataFrame['CSC_Phase'][DataPoint]=1
    DataFrame['F1_Type'][DataPoint]='Glacier'
    DataPoint+=1
    DataFrame['F1'][DataPoint]=GetF1(reportCSC)
    DataFrame['Patch_Size'][DataPoint]=GetPatch(folder)
    DataFrame['Tile_Size'][DataPoint]=GetTileSize(folder)
    DataFrame['Bands'][DataPoint]=GetBands(folder)
    DataFrame['CNN_Training'][DataPoint]=GetTraining(folder)
    DataFrame['Glacier'][DataPoint]=GetGlacier(folder)
    DataFrame['Validation'][DataPoint]=GetValidation(folder)
    DataFrame['CSC_Phase'][DataPoint]=2
    DataFrame['F1_Type'][DataPoint]='All'
    DataPoint+=1
    DataFrame['F1'][DataPoint]=GetGlacierF1(reportCSC)
    DataFrame['Patch_Size'][DataPoint]=GetPatch(folder)
    DataFrame['Tile_Size'][DataPoint]=GetTileSize(folder)
    DataFrame['Bands'][DataPoint]=GetBands(folder)
    DataFrame['CNN_Training'][DataPoint]=GetTraining(folder)
    DataFrame['Glacier'][DataPoint]=GetGlacier(folder)
    DataFrame['Validation'][DataPoint]=GetValidation(folder)
    DataFrame['CSC_Phase'][DataPoint]=2
    DataFrame['F1_Type'][DataPoint]='Glacier'
    DataPoint+=1
    
DataFrame.to_csv(OutputName)

    
    

    

        

        


toc()