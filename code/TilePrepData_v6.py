# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:02:30 2020
@author: Melanie Marochov and Patrice Carbonneau


    TILE PREPARATION

"""
# =============================================================================

"""Import Libraries"""

import numpy as np
import skimage.io as IO

import glob
# =============================================================================

"""User Input"""


ImFolder = 'E:\\See_Ice\\TrainData\\' #location of image to be tiled
DataFolder = 'E:\\See_Ice\\Tiles50\\' #folder location for output tiles
size = 50 #size (in pixels) of output tiles
stride = 40 #number of pixels the tiler slides before extracting another tile


# =============================================================================
###############################################################################

"""Functions"""

#### CHECKLABEL FUNCTION #### 
#Checks class percentage of each tile and creates the label vector.
#Tiles which contain less than 10% of pure class, or a mixture of classes are rejected
#Label vectors are produced from tiles which contain >=90% pure class.

def CheckLabel(ClassTile):
    
    size=ClassTile.shape[0] #gets the size of the tile
    vals, counts = np.unique(ClassTile, return_counts = True)
    if (vals[0] == 0) and (counts[0] > 0.1 * size**2): #if only 10% of the tile has a class
        Valid = False #This identifies the tile as non-classified (so it will reject it)
    elif counts[np.argmax(counts)] >= 0.95 * size**2: #if biggest class is over 90% of area valid is True
        Valid = True #This identifies the class of the tile 
    else:
        Valid = False #mix of classes that add up to 90% area
   
    return Valid #Given a classification tile, runs the check for class label

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
def save_tile(I, LabelVector, CurrentTile, DataFolder, size, stride):
    PickFolder = np.random.uniform() #Picks a random number to allocate isolated tiles to folder (uniform between 0 and 1)
    TileName = 'T'+str(CurrentTile) + '.png'
    if PickFolder <= 0.8: #For distributing in train and test folders
        if LabelVector== 1:
            IO.imsave(DataFolder+'Train'+'\\C1\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C1\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C1\\'+'L'+TileName, I)  
        elif LabelVector== 2:
            IO.imsave(DataFolder+'Train'+'\\C2\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C2\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C2\\'+'L'+TileName, I)   
        elif LabelVector  == 3:
            IO.imsave(DataFolder+'Train'+'\\C3\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C3\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C3\\'+'L'+TileName, I)  
        elif LabelVector  == 4:
            IO.imsave(DataFolder+'Train'+'\\C4\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C4\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C4\\'+'L'+TileName, I)  
        elif LabelVector  == 5:
            IO.imsave(DataFolder+'Train'+'\\C5\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C5\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C5\\'+'L'+TileName, I)  
        elif LabelVector  == 6:
            IO.imsave(DataFolder+'Train'+'\\C6\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C6\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C6\\'+'L'+TileName, I)  
        elif LabelVector  == 7:
            IO.imsave(DataFolder+'Train'+'\\C7\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C7\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Train'+'\\C7\\'+'L'+TileName, I)  
    elif (PickFolder > 0.8):
        if LabelVector  == 1:
            IO.imsave(DataFolder+'Valid'+'\\C1\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C1\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C1\\'+'L'+TileName, I)  
        elif LabelVector  == 2:
            IO.imsave(DataFolder+'Valid'+'\\C2\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C2\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C2\\'+'L'+TileName, I)  
        elif LabelVector  == 3:
            IO.imsave(DataFolder+'Valid'+'\\C3\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C3\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C3\\'+'L'+TileName, I)  
        elif LabelVector  == 4:
            IO.imsave(DataFolder+'Valid'+'\\C4\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C4\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C4\\'+'L'+TileName, I)  
        elif LabelVector  == 5:
            IO.imsave(DataFolder+'Valid'+'\\C5\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C5\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C5\\'+'L'+TileName, I)  
        elif LabelVector  == 6:
            IO.imsave(DataFolder+'Valid'+'\\C6\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C6\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C6\\'+'L'+TileName, I)  
        elif LabelVector  == 7:
            IO.imsave(DataFolder+'Valid'+'\\C7\\'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C7\\'+'U'+TileName, I)
            I=np.rot90(I)+np.uint8(3*np.random.random(size=I.shape))
            IO.imsave(DataFolder+'Valid'+'\\C7\\'+'L'+TileName, I)  


####################################################################################
# =============================================================================

"""Processing"""

img = glob.glob(ImFolder + "clip*.*")
#Tile sliding
CurrentTile = 0
for i in range(len(img)):
    #Load image

    ImName=img[i]
    TrainName=ImFolder +'Train_'+ImName[-12:]
    Im3D = IO.imread(ImName)
    ClassRaster = IO.imread(TrainName)
    im = CropToTile (Im3D, size)
    CroppedClassRaster = CropToTile (ClassRaster, size)
 
    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape
    
    
    for y in range(0, h-size, stride): #from first pixel to last pixel where 224 will fit, in steps of stride
        for x in range(0, w-size, stride):
            LabelTile = CroppedClassRaster[y:y+size,x:x+size] #Cropped class raster to tile size
            Label = np.median(CroppedClassRaster[y:y+size,x:x+size].reshape(1,-1)) #class expressed as a single number for an individual tile
            Valid = CheckLabel(LabelTile)
            Tile = im[y:y+size,x:x+size,:].reshape(size,size,d) # image tile
            Tile = np.uint8(255*Tile/16384)
            if Valid:#==true i.e. if the tile has a dominant class assigned to it
                save_tile(Tile, Label, CurrentTile, DataFolder, size, stride) #Save the tile to disk
                CurrentTile+=1 #current tile plus 1 - so won't overwrite previously saved file
                Tile=np.rot90(Tile) #rotates tile 90 degrees
                save_tile(Tile, Label, CurrentTile, DataFolder, size, stride) #Save the tile to disk
                CurrentTile+=1 #saves rotated tile and does not overwrite previously saved files
                #save_tile(Tile, Label, CurrentTile, DataFolder, size, stride) #Save the tile to disk


