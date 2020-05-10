# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 15:25:48 2020

@author: Melanie Marochov
"""

""" IMPORTS """

import numpy as np
import skimage.transform as T
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
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
Output_figure_path = 'D:\\NewVGG16_outputsTest\\Scoresby\\'
ModelName = 'VGG16_noise_RGBNIR_50' 
Image_name = 'D:\\S2_Images\\S2A1.png\\' #put entire path name with tiff of image used to show classification
#example Image_name = 'E:\\Masters\\Helheim19\\zb_18_06\\clip\\clip_18_06RGB.tif\\' #put entire path name with tiff of image used to show classification
Image_validation_raster = 'D:\\S2_Images\\SCLS_S2A1.png\\'
Image_date = '01-08-2019'
size = 50
stride = size
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
#Save classification reports to csv with Pandas
def classification_report_csv(report, filename):
    report_data = []
    report = report.replace('avg', "")
    report = report.replace('accuracy', "Accuracy")
    report = report.replace('macro', "Macro_avg")
    report = report.replace('weighted', "Weighted_avg")
    
    lines = report.split("\n")
    no_empty_lines = [line for line in lines if line.strip()]
        
    for line in no_empty_lines[1:]:
        row = {}
        row_data = line.split(' ')
        row_data = list(filter(None, row_data))
        if 'Accuracy' in line:
            row_data.insert(1, 'NaN')
            row_data.insert(2, 'NaN')
            
        row['Class'] = row_data[0]
        row['Precision'] = (row_data[1])
        row['Recall'] = (row_data[2])
        row['F1_score'] = float(row_data[3])
        row['Support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index = False) 
# =============================================================================
#fetches the overall avg F1 score from a classification report
def GetF1(report):
    lines = report.split('\n')
    for line in lines[0:-1]:
        if 'weighted' in line:
            dat = line.split(' ')
    
    return dat[17]

# =============================================================================

# =============================================================================
# Makes a bucket of zeros in the shape of a tensor and then puts each tile into its own slot in the tensor
    
im = io.imread(Image_name)#reads in image and stored as im
im = CropToTile (im, size) #indexed to remove NIR band im[:,:,0:3]  im[:,:,0:3]
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

        
""" LOAD CONVNET """

print('Loading ' + ModelName + '.h5')
FullModelPath = train_path + ModelName + '.h5'
ConvNetmodel = load_model(FullModelPath)
ConvNetmodel.summary()


""" RUN CNN """

predict = ConvNetmodel.predict(Tiles, verbose=1) #runs model on tiles - the dimension of tiles has to be on the same dimensions i.e. 3 band depth 
#predict is a label vector - one number per tile 
class_raster = (class_prediction_to_image(im, predict, size))


# =============================================================================


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
CNNname = Output_figure_path + 'CNN_Report_'+ str(Image_date)+str(ModelName)+ '.csv'   
classification_report_csv(reportCNN, CNNname)

# =============================================================================

""" DISPLAY AND/OR OUTPUT FIGURE RESULTS """

plt.figure(figsize = (20, 6)) #reduce these values if you have a small screen
cmapCHM = colors.ListedColormap(['orange','gold','mediumturquoise','lightgrey', 'darkgrey','teal','darkslategrey'])

Im3D = np.int16(io.imread(Image_name))
#Im3D = np.int16(Im3D *0.0255) #change to maximum value in images - normalised between 0-255
plt.subplot(1,3,1)
plt.imshow(Im3D[:,:,0:3])
plt.xlabel('Input RGB Image ('+str(Image_date) +')', fontweight='bold')

plt.subplot(1,3,2)
#cmapCHM = colors.ListedColormap(['black','orange','gold','mediumturquoise','lightgrey', 'darkgrey','teal','darkslategrey'])
plt.imshow(class_raster, cmap=cmapCHM)
plt.xlabel('Output VGG16 Classification - F1: ' + GetF1(reportCNN), fontweight='bold')


plt.subplot(1,3,3)
cmapCHM = colors.ListedColormap(['black','orange','gold','teal','mediumturquoise','darkslategrey','lightgrey', 'darkgrey'])
Validation_Raster = np.int16(io.imread(Image_validation_raster))
plt.imshow(Validation_Raster, cmap=cmapCHM)
plt.xlabel('Validation Labels', fontweight='bold')


class0_box = mpatches.Patch(color='black', label='Unclassified')
class1_box = mpatches.Patch(color='darkgrey', label='Snow on Ice')
class2_box = mpatches.Patch(color='lightgrey', label='Glacier Ice')
class3_box = mpatches.Patch(color='darkslategrey', label='Bedrock')
class4_box = mpatches.Patch(color='teal', label='Snow on Bedrock')
class5_box = mpatches.Patch(color='mediumturquoise', label='Mélange')
class6_box = mpatches.Patch(color='gold', label='Ice-berg Water')
class7_box = mpatches.Patch(color='orange', label='Open Water')


ax=plt.gca()
chartBox = ax.get_position()
ax.set_position([chartBox.x0, chartBox.y0, chartBox.width, chartBox.height])  #chartBox.width*0.6 changed to just width
ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1), shadow=True, ncol=1, handles=[class0_box,class1_box,class2_box,class3_box,class4_box,class5_box,class6_box,class7_box])


""" SAVE FIGURE TO FOLDER """

print('Saving output figure...')
FigName = Output_figure_path + Image_date + ModelName + '.png'
plt.savefig(FigName, bbox_inches='tight')


# =============================================================================


#predicted = np.argmax(predict, axis=1) #+1 to make the classes
