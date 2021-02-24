#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__license__ = 'MIT'
__date__ = 'FEB 2020'
__version__ = '1.1'
__status__ = "initial release"


"""
Name:           CNNSupervisedClassification.py
Compatibility:  Python 3.8
Description:    Performs CNN-Supervised Image CLassification with a 
                pre-trained Convolutional Neural Network model.
                User options are in the first section of code.

Requires:       Tensorflow 2.4, numpy, pandas, matplotlib, skimage, sklearn
                gdal

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
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as mpatches
import skimage
from skimage import io
import skimage.transform as T
from skimage.filters import median
from skimage.measure import label
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,Conv2D, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from skimage.filters.rank import modal
import skimage.measure as measure
from skimage.morphology import binary_dilation, binary_closing, binary_opening, binary_erosion
import os.path
from sklearn import metrics
from skimage.morphology import disk
#from imblearn.combine import SMOTEENN
import copy
import sys
from IPython import get_ipython #this can be removed if not using Spyder
import glob
import gdal
from skimage.segmentation import morphological_geodesic_active_contour
import statistics                                  
###############################################################################


"""User data input. Fill in the info below before running"""

ModelName = 'VGG16_50_RGBNIRfloat16_98acc'     #should be the model name from previous run of TrainCNN.py
ModelPath = '/media/patrice/DataDrive/SEE_ICE/Models/'  #location of the model
PredictPath = '/media/patrice/DataDrive/SEE_ICE/Validate/Seen_Validation_Helheim/'#'Validate/Seen_Validation_Helheim/'#H13_09_19_3000px\\#Sc01_08_19_3000px\\'   #Location of the images
ScorePath = '/media/patrice/DataDrive/SEE_ICE/Hel_VGG16_50_RGBNIR_fp16_kernel15/' #Results_Sc01_08_19_3000px\\Tiles100_results\\RGBNIR\\Patch_1\\'      #location of the output files and the model
Experiment = 'VGGIR_50'    #ID with model, bands and tilesize, kernel size will be added further down


'''BASIC PARAMETER CHOICES'''
TrainingEpochs = 300 #will use early stopping
Ndims = 4 # Feature Dimensions for the pre-trained CNN.
NClasses = 7  #The number of classes in the data. This MUST be the same as the classes used to retrain the model
Filters = 32
Kernel_size = 15 #1,3,5,7,15 are valid
size = 50#size of the prediction tiles
CNNsamples= 500000 #number of subsamples PER CLASS to extract and train cCNN or MLP
SizeThresh=3#number of image megapixels your ram can handle, 
GACglacier=True #if true the geodetic contour will be used to refine the glacier class
DisplayHoldout =  False #Display the results figure which is saved to disk.  
OutDPI = 400 #Recommended 150 for inspection 1200 for papers.  

'''FILTERING OPTIONS'''
#These parameters offer extra options to smooth the classification outputs.  By default they are set
ImFilterSize=1 #median filtering, use 1 for no filter
SmallestElement = 2 # Despeckle the classification to the smallest length in pixels of element remaining, just enter linear units (e.g. 3 for 3X3 pixels)


'''MODEL TRAINING PARAMETERS''' #These would usually not be edited
LearningRate = 0.001
Chatty = 1 # set the verbosity of the model training.  Use 1 at first, 0 when confident that model is well tuned
Patience=5 #smaller cCNN require more patience, as much as 15, bigger can be 5. Use 20 for the MLP (kernel size 1)
minimumdelta=0.005
###############################################################################
Experiment=Experiment+'k'+str(Kernel_size)


'''setup for RTX use of mixed precision'''
#Needs Tensorflow 2.4 and an RTX GPU
if ('RTX' in os.popen('nvidia-smi -L').read()) and ('2.4' in tf.__version__):
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

""" HELPER FUNCTIONS SECTION """
def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print ("Elapsed time is " + str(int(time.time() - startTime_for_tictoc)) + " seconds.")
    else:
        print ("Toc: start time not set")

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
def slide_rasters_to_tiles(im, size):
    
    if len(im.shape) ==2:
        h, w = im.shape
        d = 1
    else:
        h, w, d = im.shape

    TileTensor = np.zeros(((h-size+1)*(w-size+1), size,size,d))

    B=0
    for y in range(0, h-size+1):
        for x in range(0, w-size+1):

            TileTensor[B,:,:,:] = im[y:y+size,x:x+size,:].reshape(size,size,d)
            B+=1

    return TileTensor


def Sample_Raster_Tiles(im, CLS, size, Ndims, samples, NClasses):
    MasterTileTensor = np.zeros((1,size,size,Ndims))
    MasterLabel = np.zeros((1,1))
    for C in range(NClasses+1):
        X, Y =np.where(CLS==C)
        actual_samples=samples
        if len(X)>samples:
            actual_samples=len(X)
        sample_idx = np.int32(len(X)*np.random.uniform(size=(actual_samples,1)))
        TileTensor = np.zeros((actual_samples, size,size,Ndims))
        Label = np.zeros((samples,1))
        for s in range(samples+1):
            try:
                TileTensor[s,:,:,:] = im[Y[sample_idx[s,0]]:Y[sample_idx[s,0]]+size,X[sample_idx[s,0]]:X[sample_idx[s,0]]+size,0:Ndims]
                #Label[s] = np.median(CLS[Y[sample_idx[s,0]]:Y[sample_idx[s,0]]+size,X[sample_idx[s,0]]:X[sample_idx[s,0]]+size].reshape(1,-1))
                Label[s]=CLS[Y[sample_idx[s,0]]+size//2,X[sample_idx[s,0]]+size//2]
            except:
                edge=1
                    
        data = Label.ravel() !=0
        TileTensor = np.compress(data, TileTensor, axis=0)
        Label = np.compress(data, Label, axis=0)
        MasterTileTensor = np.concatenate((MasterTileTensor,TileTensor), axis=0)
        MasterLabel = np.concatenate((MasterLabel,Label), axis=0)
    return MasterTileTensor[1:,:,:,:], MasterLabel[1:,:]
        
    
    

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


# =============================================================================
# Return a class prediction to the 1-Nclasses hierarchical classes
def SimplifyClass(ClassImage, ClassKey):
    Iclasses = np.unique(ClassImage)
    for c in range(0, len(Iclasses)):
        KeyIndex = ClassKey.loc[ClassKey['LocalClass'] == Iclasses[c]]
        Hclass = KeyIndex.iloc[0]['HierarchClass']
        ClassImage[ClassImage == Iclasses[c]] = Hclass
    return ClassImage

def BlankFrame(Image):
    Image[0:10, :]=np.zeros((10, Image.shape[1]))
    Image[:,0:10]=np.zeros((Image.shape[0], 10))
    Image[Image.shape[0]-10:Image.shape[0], :]=np.zeros((10, Image.shape[1]))
    Image[:,Image.shape[1]-10:Image.shape[1] ]=np.zeros((Image.shape[0], 10))
    
    return Image


# =============================================================================
#fetches the overall avg F1 score from a classification report
def GetF1(report):
    lines = report.split('\n')
    for line in lines[0:-1]:
        if 'weighted' in line:
            dat = line.split(' ')
    
    return dat[17]
#=================================================================================
tic()
###############################################################################
    
""" INSTANTIATE THE cCNN or MLP CLASSIFIER """ 
if Kernel_size==3:
    # create cCNN model
    model = Sequential()
    model.add(Conv2D(Filters,3, data_format='channels_last', input_shape=(Kernel_size, Kernel_size, Ndims))) #model.add(Conv2D(16,5, data_format='channels_last', input_shape=(5,5,4)))
    model.add(Flatten())
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(NClasses+1, kernel_initializer='normal', activation='softmax', dtype='float32')) 
    
        # Tune an optimiser
    Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
        # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    
    model.summary()  


elif Kernel_size==5:
    # create cCNN model
    model = Sequential()
    model.add(Conv2D(Filters,3, data_format='channels_last', input_shape=(Kernel_size, Kernel_size, Ndims))) #model.add(Conv2D(16,5, data_format='channels_last', input_shape=(5,5,4)))
    model.add(Conv2D(Filters,3))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(NClasses+1, kernel_initializer='normal', activation='softmax', dtype='float32')) 
    
        # Tune an optimiser
    Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
        # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    
    model.summary()
    
elif Kernel_size==7:
        # create cCNN model
    model = Sequential()
    model.add(Conv2D(Filters,3, data_format='channels_last', input_shape=(Kernel_size, Kernel_size, Ndims))) #model.add(Conv2D(16,5, data_format='channels_last', input_shape=(5,5,4)))
    model.add(Conv2D(Filters,3))
    model.add(Conv2D(Filters,3))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(NClasses+1, kernel_initializer='normal', activation='softmax', dtype='float32')) 
    
        # Tune an optimiser
    Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
        # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    
    model.summary()
   
elif Kernel_size==15:
        # create cCNN model
    model = Sequential()
    model.add(Conv2D(Filters,3, data_format='channels_last', input_shape=(Kernel_size, Kernel_size, Ndims))) #model.add(Conv2D(16,5, data_format='channels_last', input_shape=(5,5,4)))
    model.add(Conv2D(Filters,3))
    model.add(Conv2D(Filters,3))
    model.add(Conv2D(Filters,3))
    model.add(Conv2D(Filters,3))
    model.add(Conv2D(Filters,3))
    model.add(Flatten())
    model.add(Dense(512, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(NClasses+1, kernel_initializer='normal', activation='softmax', dtype='float32')) 
    
        # Tune an optimiser
    Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
        # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    
    model.summary()
    
elif Kernel_size==1:
   

    # create  MLP model
    model = Sequential()
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001),input_dim=Ndims, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    
    model.add(Dense(NClasses+1, kernel_initializer='normal', activation='softmax', dtype='float32')) 
    
        # Tune an optimiser
    Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    
        # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=Optim, metrics = ['accuracy'])
    
    model.summary() 
    
else:
    print('Warning: invalid kernel size')

#EstimatorNN = KerasClassifier(build_fn=patchCNN_model_L2D, epochs=TrainingEpochs, batch_size=50000, verbose=Chatty)
  

# =============================================================================

""" LOAD THE CONVNET MODEL """

#print('Loading re-trained convnet model produced by a run of TrainCNN.py')
print('Loading ' + ModelName + '.h5')
FullModelPath = ModelPath + ModelName + '.h5'
ConvNetmodel = load_model(FullModelPath)



# =============================================================================

""" CLASSIFY THE HOLDOUT IMAGES WITH THE CNN-SUPERVISED CLASSIFICATION """ 
# Getting Names from the files

extensions = ("*.png","*.jpg","*.jpeg","*.tif")
Folderlist = []
for extension in extensions:
    Folderlist.extend(glob.glob(PredictPath+"/"+extension))

Imagelist=[]
for f in range(len(Folderlist)):
    ImageName=os.path.basename(Folderlist[f])
    if not(Folderlist[f].__contains__('SCLS')):
        if not(Folderlist[f].__contains__('EDGE')):
            if not(Folderlist[f].__contains__('VM')):
                Imagelist.append(Folderlist[f])
    



for i,im in enumerate(Imagelist): 
    
    Im3D = np.int16(io.imread(im))
    Im3D = Im3D[:,:,0:Ndims]
    #Im3D = io.imread(im)#
    #print(isinstance(Im3D,uint8))
    if len(Im3D) == 2:
        Im3D = Im3D[0]
    
    try:
        Class = io.imread(os.path.dirname(im)+'/SCLS_'+os.path.basename(im))
        print('Found validation label data for ' + os.path.basename(im))
        validate_output=True
        if (Class.shape[0] != Im3D.shape[0]) or (Class.shape[1] != Im3D.shape[1]):
            print('WARNING: inconsistent image and class mask sizes for ' + im)
            Class = T.resize(Class, (Im3D.shape[0], Im3D.shape[1]), preserve_range = True) #bug handling for vector
        ClassIm = copy.deepcopy(Class)
    except:
        print('No truth label data for ' + os.path.basename(im))
        validate_output=False
    try:
        CalvingFront = io.imread(os.path.dirname(im)+'/EDGE_'+os.path.basename(im))
        print('Found validation data for calving front' + os.path.basename(im))
        validate_edge=True
        CalvingFront_display=binary_dilation(CalvingFront, selem=np.ones((10,10)))
        ClassIm[CalvingFront_display==1]=8
    except:
        print('No data for calving front' + os.path.basename(im))
        validate_edge=False
            
            


    print('CNN-supervised classification of ' + os.path.basename(im))

    
    #Tile the images to run the convnet
    ImCrop = CropToTile (Im3D[:,:,0:Ndims], size) #pass RGB or RGBNIR to the convnet, as needed
    I_tiles = split_image_to_tiles(ImCrop, size)
    ImMax=np.max(np.unique(Im3D))
    if ImMax<256:
        NormFactor=255 #8bit imagery
    else:
        NormFactor=16384/2 #14 S2 bit imagery
    
    I_tiles=(I_tiles)/NormFactor
    
    ImCrop = None
        
# =============================================================================
        
    """ APPLY THE CONVNET """
    
    #Apply the initial VGG model to detect training areas
    print('Detecting CNN-supervised training areas')
    
    PredictedTiles = ConvNetmodel.predict(I_tiles, batch_size = 32, verbose = Chatty)
    #Convert the convnet one-hot predictions to a new class label image
    I_tiles = None

    PredictedClass = class_prediction_to_image(np.squeeze(Im3D[:,:,0]), PredictedTiles, size)



    #needed so the CSC CNN uses the same class system as VGG  
    PredictedTiles = None
    
    #Prep the pixel data into a tensor of patches
    I_Stride1Tiles, Labels = Sample_Raster_Tiles(Im3D, PredictedClass, Kernel_size, Ndims,CNNsamples, NClasses) 
    I_Stride1Tiles = np.float16((I_Stride1Tiles) / NormFactor) #already normalised
    I_Stride1Tiles = np.squeeze(I_Stride1Tiles)
    Labels[0,0]=NClasses #force at least 1 pixel to have class 7 and control 1 hot encoding. means that argument of maximum predition is the class.
    Labels1Hot = to_categorical(Labels)

    if Kernel_size>1:
        print('Fitting compact CNN Classifier on ' + str(I_Stride1Tiles.shape[0]) + ' tiles')
    else:
        print('Fitting MLP Classifier on ' + str(I_Stride1Tiles.shape[0]) + ' pixels')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=Patience, min_delta=minimumdelta, restore_best_weights=(True))
    (trainX, testX, trainY, testY) = train_test_split(I_Stride1Tiles, Labels1Hot, test_size=0.2)
    history=model.fit(x=trainX, y=trainY, epochs=TrainingEpochs, batch_size=int(500000/Kernel_size**2), verbose=Chatty, validation_data = (testX, testY),callbacks=[callback])
    I_stride1Tiles=None       
    '''Predict all pixels with line by line option to manage memory'''

    if SizeThresh*1000000>=(Im3D.shape[0]*Im3D.shape[1]):
        print('Classifying whole image...')
        Tiles = np.squeeze(slide_rasters_to_tiles(Im3D[:,:,0:Ndims], Kernel_size))/NormFactor
        PredictedTiles = np.argmax(model.predict(Tiles, batch_size=25000), axis=1)
        PredictedSubImage = PredictedTiles.reshape(Im3D.shape[0]-Kernel_size+1, Im3D.shape[1]-Kernel_size+1)
    else:
        PredictedSubImage = np.zeros((Im3D.shape[0]-Kernel_size+1, Im3D.shape[1]-Kernel_size+1))
        print('Classifying rows...')
        for r in range(0,Im3D.shape[0]-Kernel_size):
            
            if r % 1000 == 0:
                print("%d: %s" % (r,'of 3000'))
            #print('row '+str(r)+' of '+str(Im3D.shape[0]))
            Irow = Im3D[r:r+Kernel_size,:,:]
            #tile the image
            Tiles = np.squeeze(slide_rasters_to_tiles(Irow, Kernel_size))/NormFactor
            Predicted = model.predict(Tiles, batch_size=25000)
        
            PredictedSubImage[r,:]=np.argmax(Predicted, axis=1)
        
    if SmallestElement > 0:
        PredictedSubImage = modal(np.uint8(PredictedSubImage), disk(2*SmallestElement)) #clean up the class with a mode filter
    
    PredictedImage = np.zeros((Im3D.shape[0], Im3D.shape[1]))
    PredictedImage[Kernel_size//2:Im3D.shape[0]-Kernel_size//2, Kernel_size//2:Im3D.shape[1]-Kernel_size//2]=PredictedSubImage
    Predicted = None
    PredictedSubImage = None
# # =============================================================================
#'''Detect Calving front from binary morphology operations and active contours'''
    print('Detecting Calving Front')
    try:
        G2=1*(PredictedImage==4)
        OceanPixels=np.logical_or(PredictedImage==2, PredictedImage==1)
        RealOcean=np.logical_or(OceanPixels, PredictedImage==3)
        # O2label=label(O2, connectivity=1)
        # [c, count]=np.unique(O2label, return_counts=True)
        # c=c[1:]
        # count=count[1:]
        # RealOcean=1*(O2label==np.argmax(count)+1)
        G2label=label(G2, connectivity=1)
        [c, count]=np.unique(G2label, return_counts=True)
        c=c[1:]
        count=count[1:]
        RealGlacier=1*(G2label==np.argmax(count)+1)
        # RealGlacier=binary_closing(RealGlacier, selem=np.ones((3,3)))
        # RealOcean=binary_closing(RealOcean, selem=np.ones((3,3)))
        # GO2=BlankFrame(scipy.ndimage.binary_fill_holes(RealGlacier))
        RO2=1*scipy.ndimage.binary_fill_holes(RealOcean)
        #deploy the geodetic active contours
        RealGlacier=scipy.ndimage.binary_fill_holes(RealGlacier)
        init_ls=1*binary_dilation(1*RealGlacier,selem=np.ones((5,5)))
        init_ls=1*binary_closing(init_ls, selem=np.ones((30,30)))
        init_ls=1*scipy.ndimage.binary_fill_holes(init_ls)
        image = 1-(np.float32(1*RealGlacier))
        GO2 = morphological_geodesic_active_contour(image, 30, init_ls,
                                                smoothing=1, balloon=-0.5,
                                                threshold=0.6)
        GO2=scipy.ndimage.morphology.binary_erosion(GO2, iterations=2)
    
            
        RO2=np.logical_and(RO2, np.logical_not(GO2))
        GO2=BlankFrame(GO2)
        C=measure.find_contours(GO2, level=0.5)
        csize=np.zeros(len(C))
        for i in range(len(C)):
            csize=C[i].shape[0]
        MainContour=np.int16(C[np.argmax(csize)])
        GlacierContour=np.zeros((GO2.shape))
        for c in range(len(MainContour)):
            GlacierContour[MainContour[c,0], MainContour[c,1]]=1
        
    
        
        # Overlap_b4dilate=np.logical_and(RealGlacier,RealOcean)
        # RealGlacier=np.logical_and(RealGlacier, np.logical_not(Overlap_b4dilate))
        # RealOcean=np.logical_and(RealOcean, np.logical_not(Overlap_b4dilate))
        GlacierContour=binary_closing(GlacierContour, selem=np.ones((3,3)))
        RO2=binary_dilation(RO2, selem=np.ones((30,30)))
        cfront=np.logical_and(RO2,GlacierContour)
        #cfront=np.zeros((10,10), dtype='bool')
        if cfront.any():
            cfrontlabel=label(cfront, connectivity=2)
            [c, count]=np.unique(cfrontlabel, return_counts=True)
            c=c[1:]
            count=count[1:]
            cfront=(cfrontlabel==np.argmax(count)+1)
    
        if GACglacier:#use the GAC to enforce the ice class of the main glacier, will in effect remove ocean-type pixels from the ice.
            C6=PredictedImage==6
            C7=PredictedImage==7
            PredictedImage[GO2==1]=4
            PredictedImage[C6]=6#restitute rock classes
            PredictedImage[C7]=7
            C6=None
            C7=None
        #Set the front in the predicted image as class 8
        if cfront.any() and (np.sum(1*cfront.reshape(1,-1))<np.sqrt(cfront.shape[0]**2+cfront.shape[1]**2)):#front has to be shorter than the diagonal of the image or else it's an error
            PredictedImage_display=copy.deepcopy(PredictedImage)
            PredictedImage_class=copy.deepcopy(PredictedImage)
            PredictedImage_class[cfront]=8
            cfront=binary_dilation(cfront, selem=np.ones((10,10)))
            PredictedImage_display[cfront]=8
        else:
            PredictedImage_display=copy.deepcopy(PredictedImage)
            PredictedImage_class=copy.deepcopy(PredictedImage)
            print('WARNING: calving front detection failed for '+os.path.basename(im))
            validate_edge=False
    except:
        PredictedImage_display=copy.deepcopy(PredictedImage)
        PredictedImage_class=copy.deepcopy(PredictedImage)
        print('WARNING: calving front detection failed for '+os.path.basename(im))
        validate_edge=False

    
    if validate_edge:
        Xp,Yp=np.where(PredictedImage_class==8)
        Xm,Ym=np.where(CalvingFront==1)
        Xp=Xp.reshape(-1,1)
        Yp=Yp.reshape(-1,1)
        Xm=Xm.reshape(-1,1)
        Ym=Ym.reshape(-1,1)
        
        XM=np.concatenate((Xm,Ym), axis=1)
        XP=np.concatenate((Xp,Yp), axis=1)

        D=scipy.spatial.distance.cdist(XP, XM, metric='euclidean')
        mindist=10*np.min(D, axis=0)
        print('Modal error= '+str(int(statistics.mode(mindist))))
        print('median error '+str(int(np.median(mindist))))
        print('mean error '+str((np.mean(mindist))))
        print('stdev error '+str((np.std(mindist))))    
        ImageRoot=os.path.basename(im)[:-4]
        distname=ScorePath+ImageRoot+'_cfdistances'
        np.save(distname, mindist)
    # fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    # ax[0,0].imshow(cfront)
    # ax[0,1].imshow(RO2)
    # ax[1,0].imshow(GlacierContour)
    # ax[1,1].imshow(GO2)

#         """ PRODUCE CLASSIFICATION REPORTS """
    if validate_output:
          #reshapes to a 1d vector
        Class = Class.reshape(-1,1) 
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
    else:
        print('No validation data to estimate quality is available')
        


            
        
# # =============================================================================
        
#         """ SAVE AND OUTPUT FIGURE RESULTS """

    #Display and/or oputput figure results

    RGBdisplayImage=skimage.img_as_ubyte(Im3D[:,:,0:3])
    p2, p98 = np.percentile(RGBdisplayImage, (2, 98))
    RGBdisplayImage= skimage.exposure.rescale_intensity(RGBdisplayImage, in_range=(p2, p98))
    for c in range(0,9): #this sets 1 pixel to each class to standardise colour display - max num add one to num of classes
        ClassIm[c,0] = c
        PredictedClass[c,0] = c
        PredictedImage[c,0] = c
        PredictedImage_display[c,0] = c
    #get_ipython().run_line_magic('matplotlib', 'qt')
    plt.figure(figsize = (12, 9.5)) #reduce these values if you have a small screen
    if validate_output:
        plt.subplot(2,2,1)
        plt.imshow(RGBdisplayImage)
        plt.title('Classification Results for ' + os.path.basename(im), fontweight='bold')
        plt.xlabel('Input RGB Image', fontweight='bold')
        plt.subplot(2,2,2)
        #cmapCHM = colors.ListedColormap(['black','orange','gold','mediumturquoise','lightgrey', 'darkgrey','teal','darkslategrey'])
        cmapCHM = colors.ListedColormap(['black','#03719c','#06b1c4','#6fe1e5','#b9ebee','mintcream','#c9c9c9','#775b5a','red'])
        plt.imshow(np.squeeze(ClassIm), cmap=cmapCHM)
        plt.xlabel('Validation Labels', fontweight='bold')
        
        class0_box = mpatches.Patch(color='black', label='Unclassified')
        class1_box = mpatches.Patch(color='#775b5a', label='Bedrock')
        class2_box = mpatches.Patch(color='#c9c9c9', label='Snow on Bedrock')
        class3_box = mpatches.Patch(color='mintcream', label='Snow on Ice')
        class4_box = mpatches.Patch(color='#b9ebee', label='Glacier Ice')
        class5_box = mpatches.Patch(color='#6fe1e5', label='Mélange')
        class6_box = mpatches.Patch(color='#06b1c4', label='Ice-berg Water')
        class7_box = mpatches.Patch(color='#03719c', label='Open Water')
        class8_box = mpatches.Patch(color='red', label='Calving Front')
    
        
    #        class0_box = mpatches.Patch(color='black', label='Unclassified')
    #        class1_box = mpatches.Patch(color='darkgrey', label='Snow on Ice')
    #        class2_box = mpatches.Patch(color='lightgrey', label='Glacier Ice')
    #        class3_box = mpatches.Patch(color='darkslategrey', label='Bedrock')
    #        class4_box = mpatches.Patch(color='teal', label='Snow on Bedrock')
    #        class5_box = mpatches.Patch(color='mediumturquoise', label='Mélange')
    #        class6_box = mpatches.Patch(color='gold', label='Ice-berg Water')
    #        class7_box = mpatches.Patch(color='orange', label='Open Water')
        
        ax=plt.gca()
        ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.02), shadow=True, handles=[class0_box, class1_box,class2_box,class3_box,class4_box,class5_box,class6_box,class7_box, class8_box])
    
        plt.subplot(2,2,3)
        plt.imshow(np.squeeze(PredictedClass), cmap=cmapCHM)
        plt.xlabel('CNN Tile Classification. F1: ' + GetF1(reportCNN), fontweight='bold')
        plt.subplot(2,2,4)
        plt.imshow(PredictedImage_display, cmap=cmapCHM)
        if validate_edge:
            plt.xlabel('CSC F1: ' + GetF1(reportSSC)+'   Calving Front error [m]: '+("{0:.2f}".format((np.median(mindist)))), fontweight='bold' )
        else:
            plt.xlabel('CSC F1: ' + GetF1(reportSSC), fontweight='bold' )
        FigName = ScorePath + 'CSC_'+  Experiment + '_'+ os.path.basename(im)[:-4] +'.png'
        plt.savefig(FigName, dpi=OutDPI, bbox_inches='tight')
        # plt.figure()
        # plt.hist(mindist,bins=range(int(min(mindist)), int(max(mindist)) + 10, 10))
    else:
        plt.subplot(2,2,1)

        plt.imshow(RGBdisplayImage)
        plt.title('Classification Results for ' + os.path.basename(im), fontweight='bold')
        plt.xlabel('Input RGB Image', fontweight='bold')
        
        cmapCHM = colors.ListedColormap(['black','#03719c','#06b1c4','#6fe1e5','#b9ebee','mintcream','#c9c9c9','#775b5a', 'red'])
               
        class0_box = mpatches.Patch(color='black', label='Unclassified')
        class1_box = mpatches.Patch(color='#775b5a', label='Bedrock')
        class2_box = mpatches.Patch(color='#c9c9c9', label='Snow on Bedrock')
        class3_box = mpatches.Patch(color='mintcream', label='Snow on Ice')
        class4_box = mpatches.Patch(color='#b9ebee', label='Glacier Ice')
        class5_box = mpatches.Patch(color='#6fe1e5', label='Mélange')
        class6_box = mpatches.Patch(color='#06b1c4', label='Ice-berg Water')
        class7_box = mpatches.Patch(color='#03719c', label='Open Water')
        class8_box = mpatches.Patch(color='red', label='Calving Front')
        
    #        class0_box = mpatches.Patch(color='black', label='Unclassified')
    #        class1_box = mpatches.Patch(color='darkgrey', label='Snow on Ice')
    #        class2_box = mpatches.Patch(color='lightgrey', label='Glacier Ice')
    #        class3_box = mpatches.Patch(color='darkslategrey', label='Bedrock')
    #        class4_box = mpatches.Patch(color='teal', label='Snow on Bedrock')
    #        class5_box = mpatches.Patch(color='mediumturquoise', label='Mélange')
    #        class6_box = mpatches.Patch(color='gold', label='Ice-berg Water')
    #        class7_box = mpatches.Patch(color='orange', label='Open Water')
         
        ax=plt.gca()
        ax.legend(loc='upper center', bbox_to_anchor=(1.3, 1.02), shadow=True, handles=[class0_box, class1_box,class2_box,class3_box,class4_box,class5_box,class6_box,class7_box, class8_box])
    
        plt.subplot(2,2,3)
        plt.imshow(np.squeeze(PredictedClass), cmap=cmapCHM)
        plt.xlabel('CNN Tile Classification. F1: ' + GetF1(reportCNN), fontweight='bold')
        plt.subplot(2,2,4)
        plt.imshow(PredictedImage_display, cmap=cmapCHM)
        
        plt.xlabel('CNN-Supervised Classification. F1: ' + GetF1(reportSSC), fontweight='bold' )
        
        FigName = ScorePath + 'CSC_'+  Experiment + '_'+ os.path.basename(im)[:-4] +'.png'
        plt.savefig(FigName, dpi=OutDPI, bbox_inches='tight')
    if not DisplayHoldout:
        plt.close()
        
    #if it's a tif, assume it's a geotif and save the georeferenced image, otherwise just save a png
    if 'tif' in im:
        
        print('Saving geotif to disk')
        ImageRoot=os.path.basename(im)[:-4]
        SavedClass=ScorePath+ImageRoot+'_classified'+'.tif'
        ImageSize=PredictedImage.shape
        ImageFile = gdal.Open(im)
        driver = gdal.GetDriverByName("GTiff")
        outdata = driver.Create(SavedClass, ImageSize[1], ImageSize[0], gdal.GDT_Byte)
        outdata.SetGeoTransform(ImageFile.GetGeoTransform())
        outdata.SetProjection(ImageFile.GetProjection())
        outdata.GetRasterBand(1).WriteArray(PredictedImage_class)
        outdata.FlushCache() ##saves to disk!!
        outdata = None
        ImageFile = None
        
    else:
        print('Saving non-georeferenced class raster to disk in png format')
        ImageRoot=os.path.basename(im)[:-4]
        outname=ScorePath+ImageRoot+'_classified'+'.png'
        io.imsave(outname,PredictedImage_class)
     #Memory cleanup   
    PredictedClass=None
    PredictedClassVECT=None
    PredictedImage=None
    PredictedImage_class=None
    PredictedImage_display=None
    PredictedImageVect=None
    RealGlacier=None
    RealOcean=None
    RGBdisplayImage=None
    Tiles=None
    xpixel=None
    ypixel=None
    G2=None
    G2label=None
    GlacierPixels=None
    image=None
    CalvingFront=None
    CalvingFront_display=None
    cfront=None
    cfrontlabel=None
    ClassIm=None
    Class=None
    GlacierContour=None
    init_ls=None
        
toc()   
            

                
    


