# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:20:26 2020

@author: Melanie Marochov and Patrice Carbonneau


VGG16 ATTEMPT 2

"""

# =============================================================================
""" INITIAL SET-UP """

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
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.layers.convolutional import *
import sys
from tensorflow.keras.layers import Conv2D
import glob
from skimage import io
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
# =============================================================================
# =============================================================================

"""USER INPUTS"""

#Trained model and class key will also be written out to the training folder
train_path = 'G:\\SEE_ICE\\TileSize_50\\TileSize_50\\Train5030'
valid_path = 'G:\\SEE_ICE\\TileSize_50\\TileSize_50\\Valid5030'
test_path = 'G:\\SEE_ICE\\TileSize_50\\TileSize_50\\\Test5030'
TileSize = 50
training_epochs = 7
ModelTuning = False #set to True if you need to tune the training epochs. Remember to lengthen the epochs
TuningFigureName = 'Tune_VGG16_3Bands_TL'#name of the tuning figure, no need to add the path
learning_rate = 0.0001
verbosity = 1
ModelOutputName = 'VGG16_3Bands_TL'  #where the model will be saved
ImType='.jpg' #jpg or tif
Nbands=3 #can only be 3 if using this script with imagenet weights
#plots images with labels
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')


#plots confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix. 
    Normalization can be applied by setting 'Normalization=True'.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def CompileTensor(path, size, Nbands, ImType):
    MasterTensor = np.zeros((1,size,size,Nbands))
    MasterLabels = np.zeros((1,1))
    for c in range(1,8):
        fullpath=path+'\\C'+str(c)
        img = glob.glob(fullpath+"\\*"+ImType)
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
conv_base = VGG16(weights='imagenet', include_top = False, input_shape = (TileSize,TileSize,Nbands)) #used to be input_shape = (224,224,3)
conv_base.summary()
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer= regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
#Using a sigmoid for this tree shadow  / non-tree shadow application. Switch to softmax if more objects
model.add(layers.Dense(8, activation='softmax'))

#LabelTensor.shape[1]
#Freeze all or part of the convolutional base to keep imagenet weigths intact
#conv_base.trainable = False
set_trainable = False
for layer in conv_base.layers:
    #print(layer.name)
    if (layer.name == 'block5_conv3') or (layer.name == 'block5_conv2') or (layer.name == 'block5_conv1'):# or (layer.name == 'block4_conv3'):
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

         

#Tune an optimiser
Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
model.compile(Adam(lr=learning_rate), loss= 'categorical_crossentropy', metrics=['accuracy'])
model.summary() 

# =============================================================================

""" Compile Tensors """
TrainTensor, TrainLabels_sparse = CompileTensor(train_path, TileSize, Nbands, ImType)
Trainlabels=to_categorical(TrainLabels_sparse)
if ModelTuning:
    ValidTensor, ValidLabels_sparse = CompileTensor(valid_path, Nbands, ImType)
    ValidLabels=to_categorical(ValidLabels_sparse)

# =============================================================================

""" TRAIN OR TUNE VGG16 MODEL """



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


#PredictedClass = class_prediction_to_image(im, predictions, size)
#plt.imshow(predicted)
#plt.imshow(image) #image is the name of function


