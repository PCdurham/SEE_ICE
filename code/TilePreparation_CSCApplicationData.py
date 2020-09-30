# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:02:30 2020
@author: Melanie Marochov and Patrice Carbonneau


Name:           Tile Preparation - for CSC Application
Compatibility:  Python 3.6
Description:    Tiles images into manageable 4D stacks of image bands (R,G,B,NIR)
                for applying the CNN-Supervised Classification (CSC) workflow.

"""
# =============================================================================

""" Import Libraries """

import numpy as np
import skimage.io as IO

# =============================================================================

""" User Input - Fill in the info below before running """

ImName = 'H13_09_19RGBN.tif' #name of image to be tiled
ClassName = 'H13_09_19V.tif' #name of class raster used to assess accuracy
ImFolder = 'E:\\Masters\\Helheim19\\g_13_09\\' # folder location of image to be tiled
DataFolder = 'D:\\S2_Images\\' #folder location for output tiles
RootName = 'S2A'

size = 3000 #size (in pixels) of output tiles
LastTile = 0 #last tile number from previous image (prevents overwriting files)

# =============================================================================

""" Helper Functions """

    # CropToTile Function
# Helper function to crop images to have an integer number of tiles. No padding is used.
def CropToTile (Im, size):
    if len(Im.shape) == 2:#handle greyscale
        Im = Im.reshape(Im.shape[0], Im.shape[1],1)

    crop_dim0 = size * (Im.shape[0]//size)
    crop_dim1 = size * (Im.shape[1]//size)
    return Im[0:crop_dim0, 0:crop_dim1, :]
    

# =============================================================================

    #Save image tiles to disk  

def save_tile(RasterTile, CurrentTile, DataFolder, RootName):
    TileName = DataFolder+RootName+str(CurrentTile) + '.png' #saves tile as .png
    IO.imsave(TileName, RasterTile)


# =============================================================================
# =============================================================================

"""Processing"""

#Load image
Im3D = IO.imread(ImFolder+ImName)
ClassRaster = IO.imread(ImFolder+ClassName)


#Tile Processing

im = CropToTile (Im3D, size) #im is image ready to be tiled
CroppedClassRaster = CropToTile (ClassRaster, size)


if len(im.shape) ==2:
    h, w = im.shape
    d = 1
else:
    h, w, d = im.shape

CurrentTile = LastTile+1
for y in range(0, h,size): #from first pixel to last pixel where tile size will fit
    for x in range(0, w,size):
        LabelTile = CroppedClassRaster[y:y+size,x:x+size] #Cropped class raster to tile size
        Tile = im[y:y+size,x:x+size,:].reshape(size,size,d) # image tile
        Tile = np.uint8(255*Tile/16384) # Normalises tile.
        save_tile(Tile, CurrentTile, DataFolder, RootName) #Save the tile to disk
        save_tile(LabelTile, CurrentTile, DataFolder, 'SCLS_'+RootName) #Save the tile to disk
        CurrentTile+=1 #to prevent overwriting previously saved files

print('Biggest valid tile was '+str(CurrentTile))
