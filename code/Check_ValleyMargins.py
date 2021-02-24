 #-*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:37:24 2021

@author: Patrice
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import glob
import skimage.io as io
import skimage.measure as measure
import os
import seaborn as sns
import statistics

#location of the class tifs
OutputFolder='/media/patrice/DataDrive/SEE_ICE/VGG16_50_RGBNIR_fp16_kernel3/' 

#locatin of the valley margin tifs
InputFolder='/media/patrice/DataDrive/SEE_ICE/Validate/Seen_Validation_Helheim/'

VMList=glob.glob(InputFolder+'VM_*.tif')

TotalDistances=np.zeros(1)
for i in range(len(VMList)):
    VMimage=io.imread(VMList[i])
    
    ClassimageName=OutputFolder+os.path.basename(VMList[i])[3:-4]+'_classified.tif'
    ClassImage=io.imread(ClassimageName)
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
    mindist=10*np.min(D, axis=0)
    TotalDistances=np.concatenate((TotalDistances, mindist))

TotalDistances=TotalDistances[1:]
sns.histplot(TotalDistances, binwidth=10)
print('Modal error= '+str(int(statistics.mode(TotalDistances))))
print('median error '+str(int(np.median(TotalDistances))))
print('mean error '+str(int(np.mean(TotalDistances))))
print('stdev error '+str(int(np.std(TotalDistances))))
       
    # ImageRoot=os.path.basename(im)[:-4]
    # distname=ScorePath+ImageRoot+'_vmdistances'
    # np.save(distname, mindist)
   

