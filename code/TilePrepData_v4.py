# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:02:30 2020
@author: Melanie Marochov


    TILE PREPARATION

"""
# =============================================================================

"""Import Libraries"""

import numpy as np
import pandas as pd
import skimage.io as IO
import sys
# =============================================================================

"""User Input"""

ImName = 'H13_09_19RGBN.tif' #name of image to be tiled
ClassName = 'H13_09_19V.tif' #name of class raster used to assign classes to tiles
ImFolder = 'E:\\Masters\\Helheim19\\g_13_09\\' #location of image to be tiled
DataFolder = 'D:\\S2_Images\\' #folder location for output tiles
RootName = 'S2A'

size = 1120 #size (in pixels) of output tiles
LastTile =0 #last tile number from previous image

# =============================================================================
###############################################################################

"""Functions"""

#### CHECKLABEL FUNCTION #### 
#Checks class percentage of each tile and creates the label vector.
#Tiles which contain less than 10% of pure class, or a mixture of classes are rejected
#Label vectors are produced from tiles which contain >=90% pure class.

###############################################################################
    
# Helper function to crop images to have an integer number of tiles. No padding is used.
def CropToTile (Im, size):
    if len(Im.shape) == 2:#handle greyscale
        Im = Im.reshape(Im.shape[0], Im.shape[1],1)

    crop_dim0 = size * (Im.shape[0]//size)
    crop_dim1 = size * (Im.shape[1]//size)
    return Im[0:crop_dim0, 0:crop_dim1, :]
    

###############################################################################
    #Save image tiles to disk based on their associated class 
def save_tile(RasterTile, CurrentTile, DataFolder, RootName):
    TileName = DataFolder+RootName+str(CurrentTile) + '.tif'
    IO.imsave(TileName, RasterTile)


####################################################################################
# =============================================================================

"""Processing"""
#Tile sliding- extraction with a stride of 1

#Load image
Im3D = IO.imread(ImFolder+ImName)
ClassRaster = IO.imread(ImFolder+ClassName)


#Tile Processing
#im is image ready to be tiled
im = CropToTile (Im3D, size)
CroppedClassRaster = CropToTile (ClassRaster, size)


if len(im.shape) ==2:
    h, w = im.shape
    d = 1
else:
    h, w, d = im.shape

CurrentTile = LastTile+1
for y in range(0, h,size): #from first pixel to last pixel where 224 will fit, in steps of stride
    for x in range(0, w,size):
        LabelTile = CroppedClassRaster[y:y+size,x:x+size] #Cropped class raster to tile size
        Tile = im[y:y+size,x:x+size,:].reshape(size,size,d) # image tile
        Tile = np.uint8(255*Tile/16384) # used to be 16384 but replaced with max of image used.
        save_tile(Tile, CurrentTile, DataFolder, RootName) #Save the tile to disk
        save_tile(LabelTile, CurrentTile, DataFolder, 'SCLS_'+RootName) #Save the tile to disk
        CurrentTile+=1 #saves rotated tile and does not overwrite previously saved files
        #save_tile(Tile, Label, CurrentTile, DataFolder, size, stride) #Save the tile to disk

print('Biggest valid tile was '+str(CurrentTile))
#TileName = 'T'+str(CurrentTile) + '.jpg'
