#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'
__date__ = '15 APR 2019'
__version__ = '1.1'
__status__ = "initial release"
__url__ = "https://github.com/geojames/Self-Supervised-Classification"


"""
Name:           CNNSupervisedClassification.py
Compatibility:  Python 3.6
Description:    Performs CNN-Supervised Image CLassification with a 
                pre-trained Convolutional Neural Network model.
                User options are in the first section of code.

Requires:       keras, numpy, pandas, matplotlib, skimage, sklearn

Dev Revisions:  JTD - 19/4/19 - Updated file paths, added auto detection of
                    river names from input images, improved image reading loops

Licence:        MIT
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to 
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
of the Software, and to permit persons to whom the Software is furnished to do 
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.
"""

###############################################################################

""" Libraries"""
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from skimage import io
import skimage.transform as T
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.filters.rank import modal
import os.path
from sklearn import metrics
from skimage.morphology import disk
#from imblearn.combine import SMOTEENN
import copy
import sys
from IPython import get_ipython #this can be removed if not using Spyder
import glob
###############################################################################


"""User data input. Fill in the info below before running"""

ModelName = 'VGG16_noise_RGBNIR_50'     #should be the model name from previous run of TrainCNN.py
TrainPath = 'D:\\SEE_ICE\\'  #location of the model
PredictPath = 'D:\\SEE_ICE\\First60HelValidationTiles\\'   #Location of the images
ScorePath = 'D:\\SEE_ICE\SI_test\\'      #location of the output files and the model
Experiment = 'debug11May'    #ID to append to output performance files
ModelTuning=True
TuningDataName='Test' #no extension

'''BASIC PARAMETER CHOICES'''
UseSmote = False #Turn SMOTE-ENN resampling on and off
TrainingEpochs = 100 #Typically this can be reduced
Ndims = 4 # Feature Dimensions for the pre-trained CNN.
NClasses = 7  #The number of classes in the data. This MUST be the same as the classes used to retrain the model
Filters = 32
Kernel_size = 3 
size = 50 #size of the prediction tiles
SaveClassRaster = False #If true this will save each class image to disk.  Outputs are not geocoded in this script. For GIS integration, see CnnSupervisedClassification_PyQGIS.py
DisplayHoldout =  True #Display the results figure which is saved to disk.  
OutDPI = 900 #Recommended 150 for inspection 1200 for papers.  

'''FILTERING OPTIONS'''
#These parameters offer extra options to smooth the classification outputs.  By default they are set
SmallestElement = 1 # Despeckle the classification to the smallest length in pixels of element remaining, just enter linear units (e.g. 3 for 3X3 pixels)


'''MODEL PARAMETERS''' #These would usually not be edited
DropRate = 0.5
ModelChoice = 2 # 2 for deep model and 3 for very deep model 
LearningRate = 0.001
Chatty = 1 # set the verbosity of the model training.  Use 1 at first, 0 when confident that model is well tuned
MinSample = 250000 #minimum sample size per class before warning



# Path checks- checks for folder ending slash, adds if nessesary

if ('/' or "'\'") not in PredictPath[-1]:
    PredictPath = PredictPath + '/'   

if ('/' or "'\'") not in ScorePath[-1]:
    ScorePath = ScorePath +'/'
    
# create Score Directory if not present
if os.path.exists(ScorePath) == False:
    os.mkdir(ScorePath)


###############################################################################


""" HELPER FUNCTIONS SECTION """

# Helper function to crop images to have an integer number of tiles. No padding is used.
def CropToTile (Im, size):
    if len(Im.shape) == 2:#handle greyscale
        Im = Im.reshape(Im.shape[0], Im.shape[1],1)

    crop_dim0 = size * (Im.shape[0]//size)
    crop_dim1 = size * (Im.shape[1]//size)
    return Im[0:crop_dim0, 0:crop_dim1, :]
    
# =============================================================================
#Helper functions to move images in and out of tensor format
def split_image_to_tiles(im, size):
    
    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape

     
    nTiles_height = h//size
    nTiles_width = w//size
    TileTensor = np.zeros((nTiles_height*nTiles_width, size,size,d))
    B=0
    for y in range(0, nTiles_height):
        for x in range(0, nTiles_width):
            x1 = np.int32(x * size)
            y1 = np.int32(y * size)
            x2 = np.int32(x1 + size)
            y2 = np.int32(y1 + size)
            TileTensor[B,:,:,:] = im[y1:y2,x1:x2].reshape(size,size,d)
            B+=1

    return TileTensor

# =============================================================================
def slide_rasters_to_tiles(im, CLS, size):
    
    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape


    TileTensor = np.zeros(((h-size)*(w-size), size,size,d))
    Label = np.zeros(((h-size)*(w-size),1))
    B=0
    for y in range(0, h-size):
        for x in range(0, w-size):
            Label[B] = np.median(CLS[y:y+size,x:x+size].reshape(1,-1)) 

            TileTensor[B,:,:,:] = im[y:y+size,x:x+size,:].reshape(size,size,d)
            B+=1

    return TileTensor, Label

# =============================================================================
#Create the label vector
def PrepareTensorData(ImageTile, ClassTile, size):
    #this takes the image tile tensor and the class tile tensor
    #It produces a label vector from the tiles which have 90% of a pure class
    #It then extracts the image tiles that have a classification value in the labels
    LabelVector = np.zeros(ClassTile.shape[0])
    
    for v in range(0,ClassTile.shape[0]):
        Tile = ClassTile[v,:,:,0]
        vals, counts = np.unique(Tile, return_counts = True)
        if (vals[0] == 0) and (counts[0] > 0.1 * size**2):
            LabelVector[v] = 0
        elif counts[np.argmax(counts)] >= 0.9 * size**2:
            LabelVector[v] = vals[np.argmax(counts)] 
    
    LabelVector = LabelVector[LabelVector > 0]
    ClassifiedTiles = np.zeros((np.count_nonzero(LabelVector), size,size,3))
    C = 0
    for t in range(0,np.count_nonzero(LabelVector)):
        if LabelVector[t] > 0:
            ClassifiedTiles[C,:,:,:] = ImageTile[t,:,:,:]
            C += 1
    return LabelVector, ClassifiedTiles


# =============================================================================
def class_prediction_to_image(im, PredictedTiles, size):#size is size of tiny patches (in pixels)

    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape

     
    nTiles_height = h//size
    nTiles_width = w//size
    #TileTensor = np.zeros((nTiles_height*nTiles_width, size,size,d))
    TileImage = np.zeros(im.shape)
    B=0
    for y in range(0, nTiles_height):
        for x in range(0, nTiles_width):
            x1 = np.int32(x * size)
            y1 = np.int32(y * size)
            x2 = np.int32(x1 + size)
            y2 = np.int32(y1 + size)
            #TileTensor[B,:,:,:] = im[y1:y2,x1:x2].reshape(size,size,d)
            TileImage[y1:y2,x1:x2] = np.argmax(PredictedTiles[B,:])#+1
            B+=1

    return TileImage

# =============================================================================
# This is a helper function to repeat a filter on 3 colour bands.  Avoids an extra loop in the big loops below
def ColourFilter(Image):
    med = np.zeros(np.shape(Image))
    for b in range (0,3):
        img = Image[:,:,b]
        med[:,:,b] = median(img, disk(5))
    return med
 

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

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #CHANGED from original to save overall accuracy
        #Original code:
        
#def classification_report_csv(report, filename):
#    report_data = []
#    lines = report.split('\n')
#    for line in lines[2:-5]:
#        row = {}
#        row_data = line.split(' ') 
#        row_data = list(filter(None, row_data))
#        row['class'] = row_data[0]
#        row['precision'] = float(row_data[1])
#        row['recall'] = float(row_data[2])
#        row['f1_score'] = float(row_data[3])
#        row['support'] = float(row_data[4])
#        report_data.append(row)
#    dataframe = pd.DataFrame.from_dict(report_data)
#    dataframe.to_csv(filename, index = False) 

 
# =============================================================================
# Return a class prediction to the 1-Nclasses hierarchical classes
def SimplifyClass(ClassImage, ClassKey):
    Iclasses = np.unique(ClassImage)
    for c in range(0, len(Iclasses)):
        KeyIndex = ClassKey.loc[ClassKey['LocalClass'] == Iclasses[c]]
        Hclass = KeyIndex.iloc[0]['HierarchClass']
        ClassImage[ClassImage == Iclasses[c]] = Hclass
    return ClassImage



# =============================================================================
#fetches the overall avg F1 score from a classification report
def GetF1(report):
    lines = report.split('\n')
    for line in lines[0:-1]:
        if 'weighted' in line:
            dat = line.split(' ')
    
    return dat[17]
#=================================================================================
def TuneModelEpochs(Tiles,Labels, model,TuningDataName,Path):
        #Split the data for tuning. Use a double pass of train_test_split to shave off some data
    (trainX, testX, trainY, testY) = train_test_split(Tiles, Labels, test_size=0.2)
    

    history = model.fit(trainX, trainY, epochs = 500, batch_size = 5000, validation_data = (testX, testY))
    #Plot the test results
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    
    epochs = range(1, len(loss_values) + 1)
    get_ipython().run_line_magic('matplotlib', 'qt')
    plt.figure(figsize = (12, 9.5))
    plt.subplot(1,2,1)
    plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
    plt.plot(epochs,val_loss_values, 'b', label = 'Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    plt.subplot(1,2,2)
    plt.plot(epochs, acc_values, 'go', label = 'Training acc')
    plt.plot(epochs, val_acc_values, 'g', label = 'Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    DF=pd.DataFrame(history_dict)
    DF.to_csv(Path+TuningDataName+'.csv')
    sys.exit("Tuning Finished, adjust parameters and re-train the model")

    


###############################################################################
    
""" INSTANTIATE THE cCNN or MLP CLASSIFIER """ 
   


if Kernel_size>1:
    # create cCNN model
    model = Sequential()
    model.add(Conv2D(Filters,Kernel_size, data_format='channels_last', input_shape=(Kernel_size, Kernel_size, Ndims))) #model.add(Conv2D(16,5, data_format='channels_last', input_shape=(5,5,4)))
    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(NClasses+1, kernel_initializer='normal', activation='softmax')) 
    
        # Tune an optimiser
    Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
        # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    
    model.summary()
else:
   

    # create  MLP model
    model = Sequential()
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001),input_dim=Ndims, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(NClasses+1, kernel_initializer='normal', activation='softmax')) 
    
        # Tune an optimiser
    Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
        # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    
    model.summary() 

#EstimatorNN = KerasClassifier(build_fn=patchCNN_model_L2D, epochs=TrainingEpochs, batch_size=50000, verbose=Chatty)
  

# =============================================================================

""" LOAD THE CONVNET MODEL """

#print('Loading re-trained convnet model produced by a run of TrainCNN.py')
print('Loading ' + ModelName + '.h5')
FullModelPath = TrainPath + ModelName + '.h5'
ConvNetmodel = load_model(FullModelPath)

#ClassKeyPath = TrainPath + ModelName + '.csv'
#ClassKey = pd.read_csv(ClassKeyPath)

# =============================================================================

""" CLASSIFY THE HOLDOUT IMAGES WITH THE CNN-SUPERVISED CLASSIFICATION """ 
# Getting Names from the files
# Glob list fo all jpg images, get unique names form the total list
img = glob.glob(PredictPath+"S2A*.png")



# Get training class images (covers tif and tiff file types)
class_img = glob.glob(PredictPath + "SCLS_S2A*.png")


for i,im in enumerate(img): 
    
    Im3D = np.int16(io.imread(im))
    #Im3D = io.imread(im)#
    #print(isinstance(Im3D,uint8))
    if len(Im3D) == 2:
        Im3D = Im3D[0]
    Class = io.imread(class_img[i])
    PresentClasses=np.unique(Class)
    if np.max(PresentClasses)==0:
        print('No truth label data for ' + os.path.basename(im))
    else:
        print('CNN-supervised classification of ' + os.path.basename(im))
        if (Class.shape[0] != Im3D.shape[0]) or (Class.shape[1] != Im3D.shape[1]):
            print('WARNING: inconsistent image and class mask sizes for ' + im)
            Class = T.resize(Class, (Im3D.shape[0], Im3D.shape[1]), preserve_range = True) #bug handling for vector
        ClassIm = copy.deepcopy(Class)
        
        #Tile the images to run the convnet
        ImCrop = CropToTile (Im3D[:,:,0:Ndims], size) #pass RGB or RGBNIR to the convnet, as needed
        I_tiles = split_image_to_tiles(ImCrop, size)
        I_tiles=(I_tiles)/255
        
        ImCrop = None
        
# =============================================================================
        
        """ APPLY THE CONVNET """
        
        #Apply the initial VGG model to detect training areas
        print('Detecting CNN-supervised training areas')
        
        PredictedTiles = ConvNetmodel.predict(I_tiles, batch_size = 32, verbose = Chatty)
        #Convert the convnet one-hot predictions to a new class label image
        I_tiles = None

        PredictedClass = class_prediction_to_image(Class, PredictedTiles, size)
        #So now we have a class image of the VGG output with classes corresponding to indices


        
        
        """ APPLY THE PATCH CNN USING SAME CLASS SYSTEM """

        #needed so the CSC CNN uses the same class system as VGG  
        PredictedTiles = None

        #Prep the pixel data into a tensor of patches
        I_Stride1Tiles, Labels = slide_rasters_to_tiles(Im3D[:,:,0:Ndims], PredictedClass, Kernel_size) 
        I_Stride1Tiles = np.int16(I_Stride1Tiles) / 255 #already normalised
        I_Stride1Tiles = np.squeeze(I_Stride1Tiles)
        Labels[0,0]=NClasses #force at least 1 pixel to have class 7 and control 1 hot encoding. means that argument of maximum predition is the class.
        Labels1Hot = to_categorical(Labels)
        if ModelTuning:
            TuneModelEpochs(I_Stride1Tiles,Labels1Hot, model, TuningDataName,TrainPath)
        elif Kernel_size>1:
            print('Fitting compact CNN Classifier on ' + str(I_Stride1Tiles.shape[0]) + ' tiles')
        else:
            print('Fitting MLP Classifier on ' + str(I_Stride1Tiles.shape[0]) + ' tiles')

        model.fit(x=I_Stride1Tiles, y=Labels1Hot, epochs=TrainingEpochs, batch_size=5000, verbose=Chatty)
                
        #Fit the predictor to all patches
        Predicted = model.predict(x=I_Stride1Tiles, batch_size=50000, verbose=Chatty)
        I_Stride1Tiles = None
        Labels1Hot = None

        Predicted = np.argmax(Predicted, axis=1) 

        
        #Reshape the predictions to image format and display
        PredictedImage = Predicted.reshape(Im3D.shape[0]-Kernel_size, Im3D.shape[1]-Kernel_size) #why the -5? (Im3D.shape[0]-5, Im3D.shape[1]-5)
        if SmallestElement > 0:
            PredictedImage = modal(np.uint8(PredictedImage), disk(2*SmallestElement)) #clean up the class with a mode filter

        Predicted = None
# =============================================================================


        """ PRODUCE CLASSIFICATION REPORTS """

        Class = Class[0:Class.shape[0]-Kernel_size, 0:Class.shape[1]-Kernel_size] 
        #makes sure Class is same shape as PredictedImage and PredictedClass
        Class = Class.reshape(-1,1) 
        #reshapes to a 1d vector

        
        PredictedClass = PredictedClass[0:PredictedClass.shape[0]-Kernel_size, 0:PredictedClass.shape[1]-Kernel_size] 
        #makes the same shape as other output rasters

        PredictedImageVECT = PredictedImage.reshape(-1,1) #This is the pixel-based prediction
        PredictedClassVECT = PredictedClass.reshape(-1,1) # This is the CNN tiles prediction
        PredictedImageVECT = PredictedImageVECT[Class != 0] # removes the zeros
        PredictedClassVECT = PredictedClassVECT[Class != 0]
        
        Class = Class[Class != 0] #removes zeros from classes
        Class = np.int32(Class)

        PredictedImageVECT = np.int32(PredictedImageVECT)
        PredictedClassVECT = np.int32(PredictedClassVECT)

        reportSSC = metrics.classification_report(Class, PredictedImageVECT, digits = 3)
        reportCNN = metrics.classification_report(Class, PredictedClassVECT, digits = 3)
        
        print('CNN tiled classification results for ' + os.path.basename(im))
        print(reportCNN)
        print('\n')
        print('CNN-Supervised classification results for ' + os.path.basename(im))
        print(reportSSC)
        print('\n')
        
        CSCname = ScorePath + 'CSC_' + os.path.basename(im)[:-4] + '_' + Experiment + '.csv'    
        classification_report_csv(reportSSC, CSCname)

        CNNname = ScorePath + 'CNN_' + os.path.basename(im)[:-4] + '_' + Experiment + '.csv'    
        classification_report_csv(reportCNN, CNNname)
        DATname = ScorePath + 'OvPdat_' + os.path.basename(im)[:-4] + '_' + Experiment + '.npz'     
        DAT=np.concatenate((Class.reshape(-1,1), PredictedClassVECT.reshape(-1,1),PredictedImageVECT.reshape(-1,1)), axis=1)
        np.savez_compressed(DATname,DAT)            
        
# =============================================================================
        
        """ SAVE AND OUTPUT FIGURE RESULTS """

        #Display and/or oputput figure results

        for c in range(0,8): #this sets 1 pixel to each class to standardise colour display - max num add one to num of classes
            ClassIm[c,0] = c
            PredictedClass[c,0] = c
            PredictedImage[c,0] = c
        #get_ipython().run_line_magic('matplotlib', 'qt')
        plt.figure(figsize = (12, 9.5)) #reduce these values if you have a small screen
        plt.subplot(2,2,1)
        plt.imshow(Im3D[:,:,0:3])
        plt.title('Classification Results for ' + os.path.basename(im), fontweight='bold')
        plt.xlabel('Input RGB Image', fontweight='bold')
        plt.subplot(2,2,2)
        cmapCHM = colors.ListedColormap(['black','orange','gold','mediumturquoise','lightgrey', 'darkgrey','teal','darkslategrey'])
        plt.imshow(np.squeeze(ClassIm), cmap=cmapCHM)
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
        ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.02), shadow=True, handles=[class0_box, class1_box,class2_box,class3_box,class4_box,class5_box,class6_box,class7_box])
    
        plt.subplot(2,2,3)
        plt.imshow(np.squeeze(PredictedClass), cmap=cmapCHM)
        plt.xlabel('CNN Tile Classification. F1: ' + GetF1(reportCNN), fontweight='bold')
        plt.subplot(2,2,4)
        plt.imshow(PredictedImage, cmap=cmapCHM)
        
        plt.xlabel('CNN-Supervised Classification. F1: ' + GetF1(reportSSC), fontweight='bold' )

        FigName = ScorePath + 'CSC_'+  Experiment + '_'+ os.path.basename(im)[:-4] +'.png'
        plt.savefig(FigName, dpi=OutDPI, bbox_inches='tight')
        if not DisplayHoldout:
            plt.close()
        
            

                
    


