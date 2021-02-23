# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 14:52:32 2021

@author: Patrice
Simple utility script to read tiles from drive and compile a large tensor saved as an npy file.  
Use only if you have enough ram to contain all your samples at once
"""
import numpy as np
import glob
import skimage.io as io

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print ("Toc: start time not set")

tic()

folder='/media/patrice/DataDrive/SEE_ICE/Train/'
tilesize=100
bands=4
classes=7
subsample=0.40#percentage subsample in each class
NormFactor=8192 #will save a normalised tensor ready for the CNN, better for memory to normalise now

Itot=0
for c in range(1,classes+1):
    class_folder=folder+'C'+str(c)+'/'
    clist=glob.glob(class_folder+'*.tif')
    Itot=Itot+len(clist)

print ('found '+str(Itot)+' tile samples')





MasterTensor=np.zeros((int(subsample*Itot),tilesize,tilesize,bands),dtype='float16')
MasterLabel=np.zeros((int(subsample*Itot)),dtype='float16')

tile=0
for c in range(1,classes+1):
    class_folder=folder+'C'+str(c)+'/'
    clist=glob.glob(class_folder+'*.tif')
    idx = np.random.choice(np.arange(len(clist)), int(len(clist)*subsample), replace=False)
       
    for i in range(len(idx)):
        I=io.imread(clist[idx[i]]).reshape((1,tilesize,tilesize,bands))
        Label=c
        MasterTensor[tile,:,:,:] = np.float16(I/NormFactor)
        MasterLabel[tile] = np.float16(Label)
        tile+=1
    print('Class '+str(c)+' compiled')
        

#Output as npy arrays for both the tensor and the label
np.save(folder+'Med30kTensor'+str(tilesize),MasterTensor)
np.save(folder+'Med30kLabel'+str(tilesize),MasterLabel)
toc()
    