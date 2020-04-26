# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 15:20:26 2020

@author: Melanie Marochov


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
#from tensorflow.keras.layers.convolutional import *

from tensorflow.keras.layers import Conv2D


from matplotlib import pyplot as plt
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.preprocessing import MultiLabelBinarizer
# =============================================================================
# =============================================================================

"""USER INPUTS"""

#Trained model and class key will also be written out to the training folder
train_path = 'D:\CNN_Data\Train10030'
valid_path = 'D:\CNN_Data\Valid10030'
test_path = 'D:\CNN_Data\Test10030'
training_epochs = 8
learning_rate = 0.0001
verbosity = 1
ModelOutputName = 'VGG16_13ims'  #where the model will be saved

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
    
# =============================================================================
# =============================================================================

""" BUILD FINE-TUNED VGG16 MODEL """  

##########################################################
"""Convnet section"""
##########################################################
#Setup the convnet and add dense layers for the big tile model
conv_base = VGG16(weights='imagenet', include_top = False, input_shape = (100,100,3)) #used to be input_shape = (224,224,3)
conv_base.summary()
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer= regularizers.l2(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
#Using a sigmoid for this tree shadow  / non-tree shadow application. Switch to softmax if more objects
model.add(layers.Dense(7, activation='softmax'))

#LabelTensor.shape[1]
#Freeze all or part of the convolutional base to keep imagenet weigths intact
conv_base.trainable = True
set_trainable = True
#for layer in conv_base.layers:
#    if (layer.name == 'block5_conv3') or (layer.name == 'block5_conv2') or (layer.name == 'block5_conv1'):# or (layer.name == 'block5_conv3'):
#        set_trainable = True
#    if set_trainable:
#        layer.trainable = True
#    else:
#        layer.trainable = False

model.summary()          

#Tune an optimiser
Optim = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
# =============================================================================

""" SET DATA PATHS """

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(100,100), classes=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'], batch_size=5) #all used to be target_size=(224,224)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(100,100), classes=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'], batch_size=5)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(100,100), classes=['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'], batch_size=5, shuffle=False)

# =============================================================================

""" TRAIN FINE-TUNED VGG16 MODEL """

model.compile(Adam(lr=learning_rate), loss= 'categorical_crossentropy', metrics=['accuracy'])

#To train the model - fits the model to our batches. Epochs should be number of images in training batches divided by number of batches
model.fit_generator(train_batches, steps_per_epoch=39560,
                    validation_data=valid_batches, validation_steps=5606, epochs=training_epochs, verbose=verbosity)

# =============================================================================

""" PREDICT USING FINE-TUNED VGG16 MODEL """

#test_imgs, test_labels = next(test_batches)
test_imgs, test_labels = next(test_batches)
test_labels = test_batches.classes
#plots(test_imgs)

#mlb = MultiLabelBinarizer()
#mlb_label_train = mlb.fit_transform(test_labels)

#plots(test_imgs, titles=test_labels)

predictions = model.predict_generator(test_batches, steps=11184, verbose=0)

#cm = confusion_matrix(test_labels,np.round(predictions[:,6]))
#
#cm_plot_labels = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
#plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')

#Classification report

observed = test_labels
predicted = np.argmax(predictions, axis=1) #+1 to make the classes
report = metrics.classification_report(observed, predicted, digits=3)
print('VGG16 results')
print('\n')
print(report)


# =============================================================================

""" SAVE THE MODEL """

FullModelPath = train_path + ModelOutputName + str(training_epochs) + 'eps' +'.h5'
model.save(FullModelPath)


#PredictedClass = class_prediction_to_image(im, predictions, size)
#plt.imshow(predicted)
#plt.imshow(image) #image is the name of function


