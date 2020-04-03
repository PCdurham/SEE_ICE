# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:25:48 2020

@author: Melanie Marochov
"""

""" IMPORTS """

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers.convolutional import *
from tensorflow.keras.layers import Conv2D

import skimage.transform as T
import os.path
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import itertools
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.preprocessing import MultiLabelBinarizer
import skimage.io as io
from tensorflow.keras.models import load_model
import matplotlib.colors as colors
import matplotlib.patches as mpatches

# =============================================================================
'''
This script runs an existing CNN model on a single image and outputs the results 
as a figure showing both the original RGB input, the classified output image, 
and a classification report. 
'''

""" USER INPUT """

train_path = 'D:\\CNN_Data\\'
Output_figure_path = 'D:\\VGG16_outputs\\'
ModelName = 'Train10035VGG16_10035stride13ims8eps' 
#Image_name = 'E:\\Masters\\Helheim19\\zb_18_06\\clip\\clip_18_06RGB.tif\\' #put entire path name with tiff of image used to show classification
Image_name = 'E:\\Masters\\Helheim19\\zb_18_06\\clip\\clip_18_06RGB.tif\\' #put entire path name with tiff of image used to show classification
Image_date = '18_06_19'
Image_validation_raster = 'E:\\Masters\\Helheim19\\zb_18_06\\clip\\Train_18_06RGB.tif\\'
size = 100
stride = 100
NormFactor = 255 #Factor to scale the images to roughly 0-1


# =============================================================================
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
            TileImage[y1:y2,x1:x2] = np.argmax(predictions[B,:])+1
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
im = CropToTile (im[:,:,0:3], size) #indexed to remove NIR band im[:,:,0:3]
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
#Save classification reports to csv with Pandas
def classification_report_csv(report, filename):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = {}
        row_data = line.split(' ') 
        row_data = list(filter(None, row_data))
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index = False) 

# =============================================================================
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


""" DISPLAY AND/OR OUTPUT FIGURE RESULTS """

plt.figure(figsize = (20, 6)) #reduce these values if you have a small screen

Im3D = np.int16(io.imread(Image_name))
Im3D = np.int16(Im3D *0.0255) #change to maximum value in images - normalised between 0-255
plt.subplot(1,2,1)
plt.imshow(Im3D)
plt.xlabel('Input RGB Image ('+str(Image_date) +')', fontweight='bold')

plt.subplot(1,2,2)
cmapCHM = colors.ListedColormap(['orange','gold','mediumturquoise','teal','darkslategrey','lightgrey', 'darkgrey'])
plt.imshow(class_raster, cmap=cmapCHM)
plt.xlabel('Output VGG16 Classification', fontweight='bold')

class0_box = mpatches.Patch(color='darkgrey', label='Bedrock')
class1_box = mpatches.Patch(color='lightgrey', label='Snow on Rock')
class2_box = mpatches.Patch(color='darkslategrey', label='Snow on Ice')
class3_box = mpatches.Patch(color='mediumturquoise', label='Melange')
class4_box = mpatches.Patch(color='gold', label='Ice-berg Water')
class5_box = mpatches.Patch(color='orange', label='Open Water')
class6_box = mpatches.Patch(color='teal', label='Glacier Ice')

ax=plt.gca()
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])  #chartBox.width*0.6 changed to just width
ax.legend(loc='upper center', bbox_to_anchor=(1.13, 0.8), shadow=True, ncol=1, handles=[class0_box, class1_box,class2_box,class6_box,class3_box,class4_box,class5_box])


""" SAVE FIGURE TO FOLDER """

print('Saving output figure...')
FigName = Output_figure_path + Image_date + ModelName + '.png'
plt.savefig(FigName)


""" PRODUCE AND SAVE CLASSIFICATION REPORT """

Class = io.imread(Image_validation_raster) #Raster of validation classes
if (Class.shape[0] != im.shape[0]) or (Class.shape[1] != im.shape[1]): #Reshapes rasters so they are the same
    print('WARNING: inconsistent image and class mask sizes')
    Class = T.resize(Class, (im.shape[0], im.shape[1]), preserve_range = True) #bug handling for vector

Class = Class.reshape(-1,1)  #vectorised validation raster
PredictedClassVECT = class_raster.reshape(-1,1) # This is the CNN tiles prediction
PredictedClassVECT = PredictedClassVECT[Class != 0] #Removes areas with no validation class data from the predicted class vector 
Class = Class[Class != 0] #Removes areas with no class data from validation data 
Class = np.int32(Class)
PredictedClassVECT = np.int16(PredictedClassVECT) # both changed to int16
reportCNN = metrics.classification_report(Class, PredictedClassVECT, digits = 3)
print(reportCNN)

#saves classification report to results folder as a csv. file 
CNNname = Output_figure_path + 'CNN_Report_'+ str(Image_date)+ '.csv'   
classification_report_csv(reportCNN, CNNname)

# =============================================================================
# =============================================================================


#predicted = np.argmax(predict, axis=1) #+1 to make the classes
