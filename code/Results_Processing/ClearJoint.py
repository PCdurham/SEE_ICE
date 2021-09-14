#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 16:39:06 2021

@author: patrice

goes through a folder of results and removes tiles used in joint fine training
to keep the validation quality estimates free of data used in CNN training
"""
import glob
import numpy as np
from sklearn import metrics
import pandas as pd
import os

MasterFolder='/media/patrice/DataDrive/SEE_ICE/FullData_Revision/' #save all run outputs here and nothing else.
JointTileFolder='/media/patrice/DataDrive/SEE_ICE/JointTrainData/'
JointTileList=glob.glob(JointTileFolder+'*.tif')
Jointlist=[]
for f in range(len(JointTileList)):
    ImageName=os.path.basename(JointTileList[f])
    if not(JointTileList[f].__contains__('SCLS')):
        Jointlist.append(JointTileList[f])



RunList=glob.glob(MasterFolder+'*/')

for j in range(len(RunList)):
    if 'joint' in RunList[j]:
        content=glob.glob(RunList[j]+'*.*')
        for i in range(len(Jointlist)):
            for k in range(len(content)):
                if os.path.basename(Jointlist[i])[:-4] in content[k]:
                    print('Deleting '+os.path.basename(content[k]))
                    os.remove(content[k])
                    
            
    

    
    

    

        

        

