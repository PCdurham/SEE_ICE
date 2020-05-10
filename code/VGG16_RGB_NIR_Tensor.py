# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:20:26 2020

@authors: Melanie Marochov and Patrice Carbonneau


VGG16 model with 4 input bands, meant to be used with RGB+NIR imagery.  Note that transfer learning is no longer an option.

"""

# =============================================================================
""" INITIAL SET-UP """

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.layers.convolutional import *
import glob
import random
from skimage import io
#from matplotlib import pyplot as plt
import sys
# =============================================================================
# =============================================================================

"""USER INPUTS"""

#Trained model and class key will also be written out to the training folder
train_path = 'E:\\See_Ice\\Tiles100\\Train'
valid_path = 'E:\\See_Ice\\Tiles100\\Valid'
#test_path = 'G:\\SEE_ICE\\TileSize_50\\TileSize_50\\\Test'
TileSize = 100
Nbands=3
Nclasses=7
BaseFilters=32
training_epochs = 15
ImType='.png' #jpg or tif
ModelTuning = False #set to True if you need to tune the training epochs. Remember to lengthen the epochs
TuningFigureName = 'Tune_VGG16_noise_RGB_75'#name of the tuning figure, no need to add the path
learning_rate = 0.0001
verbosity = 1
ModelOutputName = 'VGG16_noise_RGB_100'  #where the model will be saved

#plots images with labels


def CompileTensor(path, size, Nbands, ImType):
    MasterTensor = np.zeros((1,size,size,Nbands))
    MasterLabels = np.zeros((1,1))
    for c in range(1,8):
        fullpath=path+'\\C'+str(c)
        img = glob.glob(fullpath+"\\*"+ImType)
        if len(img)>10000:#maxes out total samples to 10k per class.  Improves balance.
            random.seed(a=int(np.random.random((1))*42), version=2)
            img=random.sample(img, 10000)
            
        tensor=np.zeros(((len(img),size,size,Nbands)))
        labels=np.zeros((len(img),1))
        for i in range(len(img)):
            I=io.imread(img[i])/255
            tensor[i,:,:,:]=I[:,:,0:Nbands]
            labels[i]=c
        MasterTensor=np.concatenate((MasterTensor,tensor), axis=0)
        MasterLabels=np.concatenate((MasterLabels,labels), axis=0)
        print('Processed class '+str(c))
    return MasterTensor[1:,:,:,:], MasterLabels[1:,:]

    
# =============================================================================
# =============================================================================

""" BUILD FINE-TUNED VGG16 MODEL """  

##########################################################
"""Convnet section"""
##########################################################
#Setup the convnet and add dense layers for the big tile model
model = Sequential()
model.add(Convolution2D(2*BaseFilters, 3, input_shape=(TileSize, TileSize, Nbands), data_format='channels_last', activation='relu', padding='same'))
model.add(Convolution2D(2*BaseFilters, 3, activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Convolution2D(4*BaseFilters, 3, activation='relu', padding='same'))
model.add(Convolution2D(4*BaseFilters, 3, activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(8*BaseFilters, 3, activation='relu', padding='same'))
model.add(Convolution2D(8*BaseFilters, 3, activation='relu', padding='same'))
model.add(Convolution2D(8*BaseFilters, 3, activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Convolution2D(16*BaseFilters, 3, activation='relu', padding='same'))
model.add(Convolution2D(16*BaseFilters, 3, activation='relu', padding='same'))
model.add(Convolution2D(16*BaseFilters, 3, activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))


model.add(Convolution2D(16*BaseFilters, 3, activation='relu', padding='same'))
model.add(Convolution2D(16*BaseFilters, 3, activation='relu', padding='same'))
model.add(Convolution2D(16*BaseFilters, 3, activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512, activation='relu',kernel_regularizer= regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu',kernel_regularizer= regularizers.l2(0.001)))
model.add(layers.Dense(Nclasses+1, activation='softmax'))



        

#Tune an optimiser
Optim = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(Adam(lr=learning_rate), loss= 'categorical_crossentropy', metrics=['accuracy'])
model.summary()  
# =============================================================================

""" Compile Tensors """
TrainTensor, TrainLabels_sparse = CompileTensor(train_path, TileSize, Nbands, ImType)
Trainlabels=to_categorical(TrainLabels_sparse)
if ModelTuning:
    ValidTensor, ValidLabels_sparse = CompileTensor(valid_path, TileSize, Nbands, ImType)
    ValidLabels=to_categorical(ValidLabels_sparse)
# =============================================================================

""" TRAIN  VGG16 MODEL """
if ModelTuning:
    #Split the data for tuning. Use a double pass of train_test_split to shave off some data

    history = model.fit(TrainTensor, Trainlabels, epochs = training_epochs, batch_size = 75, validation_data = (ValidTensor, ValidLabels))
    #Plot the test results
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    
    epochs = range(1, len(loss_values) + 1)
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
    FigName = train_path + TuningFigureName
    plt.savefig(FigName, dpi=900)
    
    sys.exit("Tuning Finished, adjust parameters and re-train the model") # stop the code if still in tuning phase.



#To train the model - fits the model to our batches. Epochs should be number of images in training batches divided by number of batches
model.fit(TrainTensor, Trainlabels,  batch_size=50, epochs=training_epochs, verbose=1)

# =============================================================================


# =============================================================================

""" SAVE THE MODEL """

FullModelPath = train_path + ModelOutputName +'.h5'
model.save(FullModelPath)

#============================================================================

