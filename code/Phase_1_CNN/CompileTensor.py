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

folder='/media/patrice/DataDrive/SEE_ICE/JointTrain/'
OutputName='JointTensor5k'
tilesize=50
bands=4
classes=7
subsample=1#percentage subsample in each class
NormFactor=8192 #will save a normalised tensor ready for the CNN, better for memory to normalise now
UINT8=False #if true this will overide NormFactor and reduce the radiometry to 8-bit via normalisation by 16384
FP16=True #cast final tensor in float 16 for mixed precision training

Itot=0
for c in range(1,classes+1):
    class_folder=folder+'C'+str(c)+'/'
    clist=glob.glob(class_folder+'*.tif')
    Itot=Itot+len(clist)

print ('found '+str(Itot)+' tile samples')





MasterTensor=np.zeros((int(subsample*Itot),tilesize,tilesize,bands), dtype='float16')
MasterLabel=np.zeros((int(subsample*Itot)), dtype='float16')

tile=0
for c in range(1,classes+1):
    class_folder=folder+'C'+str(c)+'/'
    clist=glob.glob(class_folder+'*.tif')
    idx = np.random.choice(np.arange(len(clist)), int(len(clist)*subsample), replace=False)
       
    for i in range(len(idx)):
        I=io.imread(clist[idx[i]]).reshape((1,tilesize,tilesize,bands))
        Label=c
        MasterLabel[tile] = Label
        if UINT8 and not(FP16):
            MasterTensor=np.uint8(MasterTensor)
            MasterTensor[tile,:,:,:] = np.uint8(255*I/16384)
        elif FP16 and UINT8:
            MasterTensor=np.float16(MasterTensor)
            I= np.uint8(255*I/16384)
            MasterTensor[tile,:,:,:]=np.float16(I/255)
        elif not(UINT8) and FP16:
            MasterTensor=np.float16(MasterTensor)
            MasterTensor[tile,:,:,:]=np.float16(I/NormFactor)
        else:
            MasterTensor=np.int16(MasterTensor)
            MasterTensor[tile,:,:,:]=np.int16(I)
        tile+=1
    print('Class '+str(c)+' compiled')
        
if UINT8 and not(FP16):#downsample radiometry and save as uint8

    np.save(folder+OutputName+'_T_uint8',MasterTensor)
    np.save(folder+OutputName+'_L_uint8',MasterLabel)
    
elif FP16 and UINT8:#data will be float 16, but first they have been downsampled to 8bit before normalisation

    np.save(folder+OutputName+'_T_uint8float16',MasterTensor)
    np.save(folder+OutputName+'_L_uint8float16',MasterLabel)
    
elif not(UINT8) and FP16:
    
    np.save(folder+OutputName+'_T_float16',MasterTensor)
    np.save(folder+OutputName+'_L_float16',MasterLabel) 
    
else:

    np.save(folder+OutputName+'_T_int16',MasterTensor)
    np.save(folder+OutputName+'_L_int16',MasterLabel)

#Output as npy arrays for both the tensor and the label



toc()
    
