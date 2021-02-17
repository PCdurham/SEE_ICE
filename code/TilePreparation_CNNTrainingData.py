# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 12:02:30 2020
@author: Melanie Marochov and Patrice Carbonneau


Name:           Tile Preparation
Compatibility:  Python 3.7
Description:    Tiles images into 4D stacks of image bands (R,G,B,NIR)
                for training and validating the Convolutional neural network
                (CNN) in phase one of CNN-Supervised Classification (CSC).


"""
# =============================================================================

""" Import Libraries """

import numpy as np
import skimage.io as IO
import glob

# =============================================================================

""" User Input - Fill in the info below before running """

ImFolder = '/media/patrice/DataDrive/SEE_ICE/RawData/'    #location of image to be tiled e.g. 'E:\\See_Ice\\TrainData\\'.
DataFolder = '/media/patrice/DataDrive/SEE_ICE/'  #folder location for output tiles.
size = 100           #size (in pixels) of output tiles.
stride = 10         #number of pixels the tiler slides before extracting another tile.


# =============================================================================

""" Helper Functions """

    # CheckLabel Function 
#Checks class percentage of each tile and creates the label vector.
#Tiles which contain less than 10% of pure class, or a mixture of classes are rejected.
#Label vectors are produced from tiles which contain >=90% pure class.

def CheckLabel(ClassTile):
    
    size=ClassTile.shape[0] #gets the size of the tile
    vals, counts = np.unique(ClassTile, return_counts = True)
    if (vals[0] == 0) and (counts[0] > 0.1 * size**2): #If only 10% of the tile has a class.
        Valid = False #This identifies the tile as non-classified (so it will reject it).
    elif counts[np.argmax(counts)] >= 0.95 * size**2: #If biggest class is over 95% of area valid is True.
        Valid = True #This identifies the class of the tile. 
    else:
        Valid = False #mix of classes that add up to 90% area.
   
    return Valid #Given a classification tile, runs the check for class label.

# =============================================================================
    
    # CropToTile Function
# Crops images to have an integer number of tiles. No padding is used.
    
def CropToTile (Im, size):
    if len(Im.shape) == 2: #handle greyscale.
        Im = Im.reshape(Im.shape[0], Im.shape[1],1)

    crop_dim0 = size * (Im.shape[0]//size)
    crop_dim1 = size * (Im.shape[1]//size)
    return Im[0:crop_dim0, 0:crop_dim1, :]
    

# =============================================================================

    # SaveTile Function
#Save image tiles to disk based on their associated class 
    
def save_tile(I, LabelVector, CurrentTile, DataFolder, size, stride):
    PickFolder = 0# np.random.uniform() #Picks a random number to allocate isolated tiles to folder (uniform between 0 and 1).
    TileName = 'T'+str(CurrentTile) + '.tif'
    if PickFolder <= 0.95: #For distributing in train and test folders.
        if LabelVector== 1:
            IO.imsave(DataFolder+'Train'+'/C1/'+TileName, I)

        elif LabelVector== 2:
            IO.imsave(DataFolder+'Train'+'/C2/'+TileName, I)
  
        elif LabelVector  == 3:
            IO.imsave(DataFolder+'Train'+'/C3/'+TileName, I)

        elif LabelVector  == 4:
            IO.imsave(DataFolder+'Train'+'/C4/'+TileName, I)

        elif LabelVector  == 5:
            IO.imsave(DataFolder+'Train'+'/C5/'+TileName, I)
 
        elif LabelVector  == 6:
            IO.imsave(DataFolder+'Train'+'/C6/'+TileName, I)

        elif LabelVector  == 7:
            IO.imsave(DataFolder+'Train'+'/C7/'+TileName, I)

    elif (PickFolder > 0.95):
        if LabelVector  == 1:
            IO.imsave(DataFolder+'Valid'+'\\C1\\'+TileName, I)
 
        elif LabelVector  == 2:
            IO.imsave(DataFolder+'Valid'+'\\C2\\'+TileName, I)
   
        elif LabelVector  == 3:
            IO.imsave(DataFolder+'Valid'+'\\C3\\'+TileName, I)

        elif LabelVector  == 4:
            IO.imsave(DataFolder+'Valid'+'\\C4\\'+TileName, I)
 
        elif LabelVector  == 5:
            IO.imsave(DataFolder+'Valid'+'\\C5\\'+TileName, I)
 
        elif LabelVector  == 6:
            IO.imsave(DataFolder+'Valid'+'\\C6\\'+TileName, I)
 
        elif LabelVector  == 7:
            IO.imsave(DataFolder+'Valid'+'\\C7\\'+TileName, I)

 
# =============================================================================
# =============================================================================

""" Processing """

img = glob.glob(ImFolder + "clip*.*")
#Tile sliding
CurrentTile = 0
for i in range(len(img)):
    #Load image

    ImName=img[i]
    TrainName=ImFolder +'Train_'+ImName[-13:]
    Im3D = IO.imread(ImName)
    ClassRaster = IO.imread(TrainName)
    im = CropToTile (Im3D, size)
    CroppedClassRaster = CropToTile (ClassRaster, size)
 
    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape
    
    
    for y in range(0, h-size, stride): #From first pixel to last pixel where tile size will fit, in steps of stride.
        for x in range(0, w-size, stride):
            LabelTile = CroppedClassRaster[y:y+size,x:x+size] #Cropped class raster to tile size.
            Label = np.median(CroppedClassRaster[y:y+size,x:x+size].reshape(1,-1)) #Class expressed as a single number for an individual tile.
            Valid = CheckLabel(LabelTile)
            Tile = np.int16(im[y:y+size,x:x+size,:].reshape(size,size,d)) #Image tile.
            #Tile = np.uint8(255*Tile/16384)
            if Valid: #==true i.e. if the tile has a dominant class assigned to it.
                #raw tile
                I=Tile
                save_tile(I, Label, CurrentTile, DataFolder, size, stride) #Save the tile to disk.
                CurrentTile+=1 #Current tile plus 1 - so won't overwrite previously saved file.
                #90 rotation + noise.
                Tile=np.rot90(Tile)
                I=Tile+np.int16(10*np.random.uniform(size=Tile.shape)) #Rotates tile 90 degrees + noise from 0-2.
                save_tile(I, Label, CurrentTile, DataFolder, size, stride) #Save the tile to disk.
                CurrentTile+=1 #Saves rotated tile and does not overwrite previously saved files.
                #180 rotation + noise.
                Tile=np.rot90(Tile)
                I=Tile+np.int16(10*np.random.uniform(size=Tile.shape)) #Rotates tile 90 degrees+ noise from 0-2.
                save_tile(I, Label, CurrentTile, DataFolder, size, stride) #Save the tile to disk.
                CurrentTile+=1 #Saves rotated tile and does not overwrite previously saved files.
               #270 rotation + noise.
                Tile=np.rot90(Tile)
                I=Tile+np.int16(10*np.random.uniform(size=Tile.shape)) #Rotates tile 90 degrees+ noise from 0-2.
                save_tile(I, Label, CurrentTile, DataFolder, size, stride) #Save the tile to disk.
                CurrentTile+=1 #Saves rotated tile and does not overwrite previously saved files.
                


