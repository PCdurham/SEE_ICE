# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:25:48 2020

@author: Melanie Marochov
"""

""" IMPORTS """

import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import itertools
from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import MultiLabelBinarizer
import skimage.io as io
from keras.models import load_model

# =============================================================================
'''
This script runs an existing CNN model on a single image and outputs the results as a figure
'''

""" USER INPUT """

train_path = 'D:\\CNN_Data\\'
ModelName = 'Train22432VGG16_32stride13ims5eps' 
#Image_name = 'E:\\Masters\\Helheim19\\zb_18_06\\clip\\clip_18_06RGB.tif\\' #put entire path name with tiff of image used to show classification
Image_name = 'E:\\Masters\\Helheim19\\zb_18_06\\clip\\clip_18_06RGB.tif\\' #put entire path name with tiff of image used to show classification
size = 224
stride = 224
NormFactor = 255 #Factor to scale the images to roughly 0-1

# =============================================================================
""" FUNCTIONS """

def class_prediction_to_image(im, predictions, size):

    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
        im=im[:,:,0]
    else:
        h, w, d = im.shape

     
    nTiles_height = h//size
    nTiles_width = w//size
    #TileTensor = np.zeros((nTiles_height*nTiles_width, size,size,d))
    TileImage = np.zeros((h,w))
    B=0
    for y in range(0, nTiles_height):
        for x in range(0, nTiles_width):
            x1 = np.int32(x * size)
            y1 = np.int32(y * size)
            x2 = np.int32(x1 + size)
            y2 = np.int32(y1 + size)
            TileImage[y1:y2,x1:x2] = np.argmax(predictions[B,:])
            B+=1

    return TileImage
# =============================================================================
# Helper function to crop images to have an integer number of tiles. No padding is used.
def CropToTile (Im, size):
    if len(Im.shape) == 2:#handle greyscale
        Im = Im.reshape(Im.shape[0], Im.shape[1],1)

    crop_dim0 = size * (Im.shape[0]//size)
    crop_dim1 = size * (Im.shape[1]//size)
    return Im[0:crop_dim0, 0:crop_dim1, :]


# =============================================================================
# Makes a bucket of zeros in the shape of a tensor and then puts each tile into its own slot in the tensor
    
im = io.imread(Image_name)#reads in image and stored as im
im = CropToTile (im, size)

nTiles_height = im.shape[0]//size
nTiles_width = im.shape[1]//size
Tiles = np.zeros((nTiles_height*nTiles_width, size, size, im.shape[2]))#creates a tensor of zeros based on the number of bands you are using


if len(im.shape) ==2:
    h, w = im.shape
    d = 1
else:
    h, w, d = im.shape

S=0 #sample index
for y in range(0, nTiles_height):
    for x in range(0, nTiles_width):
        x1 = np.int32(x * size)
        y1 = np.int32(y * size)
        x2 = np.int32(x1 + size)
        y2 = np.int32(y1 + size)
        Tiles[S,:,:,:] = im[y1:y2,x1:x2].reshape(size,size,d)
        S+=1        
#del(im)
Tiles = Tiles/NormFactor       
#S=0 #sample index
#for y in range(0, h-size, stride): #from first pixel to last pixel where 224 will fit, in steps of stride
#    for x in range(0, w-size, stride): #for each tile in the image 
#        Tiles[S,:,:,:] = im[y:y+size,x:x+size,:].reshape(size,size,d) # image tile
#        S+=1
# =============================================================================
        
""" LOAD CONVNET """
print('Loading ' + ModelName + '.h5')
FullModelPath = train_path + ModelName + '.h5'
ConvNetmodel = load_model(FullModelPath)
ConvNetmodel.summary()

""" RUN CNN """
predict = ConvNetmodel.predict(Tiles, verbose=1) #runs model on tiles - the dimension of tiles has to be on the same dimensions i.e. 3 band depth 
#predict is a label vector - one number per tile 
class_raster = (class_prediction_to_image(im, predict, size))
plt.figure()
plt.imshow(class_raster)




##Load image
#Image = io.imread(Image_name)
#Image = Image[:,:,0:3]
#for b in range(0,3):
#    Image[:,:,b] = 255 * Image[:,:,b]/10000#np.max(Im3D[:,:,b].reshape(1,-1)) #normalise the image
#
#
##Image = io.imread(Image_name)
#cmapCHM = colors.ListedColormap(['black','lightblue','orange','green','yellow','red'])
#plt.figure(figsize = (12, 9.5))
#plt.imshow(Image, cmap= cmapCHM)

#
#observed = test_labels
#predicted = np.argmax(predict, axis=1) #+1 to make the classes
#report = metrics.classification_report(observed, predicted, digits=3)
#print('VGG16 results')
#print('\n')
#print(report)