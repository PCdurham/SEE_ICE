# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:20:26 2020

@author: Melanie Marochov and Patrice Carbonneau



Name:           Phase 1 VGG16 - Transfer Learning (RGB)
Compatibility:  Python 3.7
Description:    VGG16 model using transfer learning with 3 input bands, meant
                to be used with RGB imagery.


"""

# =============================================================================

""" Import Libraries """

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import sys
import glob
from skimage import io
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
import random

# =============================================================================

""" User Inputs - Fill in the info below before running """

#Trained model will also be written out to the training folder
train_path = 'path'     #where training tiles are located e.g. 'E:\\See_Ice\\Train'.
valid_path = 'path'     #where validation tiles are located - use \\ instead of \ in path names.
TileSize = 100          #size of tiles created using TilePreparation_CNNTrainingData. 
training_epochs = 6     #number of epochs model will iterate over training data.
ModelTuning = False     #set to True if you need to tune the training epochs. Remember to lengthen the epochs.
TuningFigureName = 'empty'   #name of the tuning figure, no need to add the path.
learning_rate = 0.0001
verbosity = 1
ModelOutputName = 'empty'  #name of saved model.
ImType='.png' #png, jpg, or tif
Nbands=3 #can only be 3 if using this script with imagenet weights.

# =============================================================================

""" Helper Functions """
   
    
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

# =============================================================================

"""Convnet section"""

#Setup the convnet and add dense layers for the big tile model
conv_base = VGG16(weights='imagenet', include_top = False, input_shape = (TileSize,TileSize,Nbands)) 
conv_base.summary()
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer= regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Dense(8, activation='softmax'))

#LabelTensor.shape[1]
#Freeze all or part of the convolutional base to keep imagenet weigths intact
set_trainable = False
for layer in conv_base.layers:
    if (layer.name == 'block5_conv3') or (layer.name == 'block5_conv2') or (layer.name == 'block5_conv1'):# or (layer.name == 'block4_conv3'):
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
        

#Tune an optimiser
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

""" Train or tune VGG16 model """

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

""" Save the trained model """

FullModelPath = train_path + ModelOutputName +'.h5'
model.save(FullModelPath)
