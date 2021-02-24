# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:37:24 2021

@author: Patrice
"""
import numpy as np
import scipy
import matplotlib.pyplot as plt
import glob
import skimage


Folder=''


ClassList=glob.glob(Folder+'/_classified.tif')


for i in range(len(ClassList)):
    ClassImage=skimage.io.imread(ClassList[i])
    VMimageName=
    VMimage=skimage.io.imread(VMimageName)
    PredictedRocks==np.logical_and(ClassImage==6, ClassImage==7)
    RockContours=skimage.measure.find_contours(1*PredictedRocks, level=0.5)
    GlacierContour=np.zeros((VMimage.shape))
    for c in range(len(RockContours)):
        Contour=np.int16(RockContours[c])
        for p in range(len(Contour)):
            GlacierContour[Contour[c,0], Contour[c,1]]=1
            
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
    mindist=10*np.min(D, axis=1)
        
    ImageRoot=os.path.basename(im)[:-4]
    distname=ScorePath+ImageRoot+'_vmdistances'
    np.save(distname, mindist)
    