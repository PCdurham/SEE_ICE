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
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten, ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical
#from tensorflow.keras.layers.convolutional import *
import glob
from tensorflow.keras.layers import Conv2D
import random
from skimage import io
from matplotlib import pyplot as plt
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.preprocessing import MultiLabelBinarizer
import sys
# =============================================================================
# =============================================================================

"""USER INPUTS"""

#Trained model and class key will also be written out to the training folder
train_path = 'E:\\See_Ice\\Tiles50\\Train'
valid_path = 'E:\\See_Ice\\Tiles50\\Valid'
test_path = 'G:\\SEE_ICE\\TileSize_50\\TileSize_50\\\Test'
TileSize = 50
Nbands=4
Nclasses=7
BaseFilters=32
training_epochs = 10
ImType='.png' #jpg or tif
ModelTuning = False #set to True if you need to tune the training epochs. Remember to lengthen the epochs
TuningFigureName = 'Tune_VGG16_RGBNIR_75'#name of the tuning figure, no need to add the path
learning_rate = 0.0001
verbosity = 1
ModelOutputName = 'VGG16_noise_RGBNIR_50'  #where the model will be saved

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

