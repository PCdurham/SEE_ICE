# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:28:26 2021

@author: Patrice
"""
""" Libraries"""
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys
import os
from IPython import get_ipython #this can be removed if not using Spyder




'''User Inputs'''
TensorPath = '/media/patrice/DataDrive/SEE_ICE/Train/Tensor30k_T_float16.npy'  #location of the tile tensor
LabelPath = '/media/patrice/DataDrive/SEE_ICE/Train/Tensor30k_L_float16.npy' #location of the labels
OutPath='/media/patrice/DataDrive/SEE_ICE/Train/'  
ModelTuning=False
TuningDataName='VGG16_75_RGBNIRfloat16_99acc'
NClasses=7
NIR=True
Tilesize=75
CNNname='VGG' # choose NAS for NASNET large, VGG for VGG16 or Dense for Densenet 121
batches=50 #try 200 for 50x50 tiles
targetacc=0.99


'''custom callback for accuracy target training'''
class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["val_accuracy"]
        if accuracy >= self.threshold:
            print('')
            print('Validation accuracy target reached, stopping training')
            print('')
            self.model.stop_training = True

        



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

'''Load Tensor Data'''
Tensor=np.load(TensorPath)
Label=np.load(LabelPath)

if NIR:
    Tensor=Tensor[:,0:Tilesize, 0:Tilesize,:]
else:
    Tensor=Tensor[:,0:Tilesize, 0:Tilesize,0:3]





'''Instantiate CNNs'''
if 'VGG' in CNNname:
    # create cCNN model
    inShape = Tensor.shape[1:]
    from tensorflow.keras.applications import VGG16
    CNN_base = VGG16(include_top=False,weights=None,input_shape=inShape)
    

    
    model = Sequential()
    model.add((CNN_base))
    model.add(Flatten())
    #Estimator.add(Dense(32,kernel_regularizer=regularizers.l2(0.001),kernel_initializer='normal',activation='relu'))
    model.add(Dense(512, activation='relu',kernel_regularizer= regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu',kernel_regularizer= regularizers.l2(0.001)))    
    model.add(Dense(NClasses+1,kernel_initializer='normal',activation='softmax', dtype='float32'))
    

    #Tune an optimiser
    Optim = optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999,decay=0.0,amsgrad=True)
    
    # Compile model
    model.compile(loss='categorical_crossentropy',optimizer=Optim,metrics=['accuracy'])
    model.summary()
elif 'Dense'in CNNname:
    from tensorflow.keras.applications import DenseNet121
    inShape = Tensor.shape[1:]
    CNN_base = DenseNet121(include_top=False,weights=None,input_shape=inShape)
    

    
    model = Sequential()
    model.add((CNN_base))
    model.add(Flatten())
    #Estimator.add(Dense(32,kernel_regularizer=regularizers.l2(0.001),kernel_initializer='normal',activation='relu'))
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(NClasses+1,kernel_initializer='normal',activation='softmax', dtype='float32'))


    #Tune an optimiser
    Optim = optimizers.Adam(lr=0.00001,beta_1=0.9,beta_2=0.999,decay=0.0,amsgrad=True)
    
    # Compile model
    model.compile(loss='categorical_crossentropy',optimizer=Optim,metrics=['accuracy'])
    model.summary()

elif 'NAS'in CNNname:
    from tensorflow.keras.applications import ResNet50
    inShape = Tensor.shape[1:]
    CNN_base = ResNet50(include_top=False,weights=None,input_shape=inShape)
    

    
    model = Sequential()
    model.add((CNN_base))
    model.add(Flatten())
    #Estimator.add(Dense(32,kernel_regularizer=regularizers.l2(0.001),kernel_initializer='normal',activation='relu'))
    model.add(Dense(64, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
    model.add(Dense(32, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(16, kernel_regularizer= regularizers.l2(0.001), kernel_initializer='normal', activation='relu'))
    model.add(Dense(NClasses+1,kernel_initializer='normal',activation='softmax', dtype='float32'))


    #Tune an optimiser
    Optim = optimizers.Adam(lr=0.00001,beta_1=0.9,beta_2=0.999,decay=0.0,amsgrad=True)
    
    # Compile model
    model.compile(loss='categorical_crossentropy',optimizer=Optim,metrics=['accuracy'])
    model.summary()

'''Tune and Fit CNN'''
Labels1Hot = to_categorical(Label)

def TuneModelEpochs(Tiles,Labels, model,TuningDataName,Path):
        #Split the data for tuning. Use a double pass of train_test_split to shave off some data
    (trainX, testX, trainY, testY) = train_test_split(Tiles, Labels, test_size=0.2)
    
    history = model.fit(trainX, trainY, epochs = 50, batch_size = batches, validation_data = (testX, testY))
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
    plt.title('Training and Validation Loss for '+TuningDataName)
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
    
if ModelTuning:
    TuneModelEpochs(Tensor,Labels1Hot, model,TuningDataName,OutPath)
else:
    (trainX, testX, trainY, testY) = train_test_split(Tensor, Labels1Hot, test_size=0.2)
    when2stop = MyThresholdCallback(threshold=targetacc)
    history = model.fit(trainX, trainY, epochs = 120, batch_size = batches, validation_data = (testX, testY), callbacks=when2stop)
    Predictions=np.argmax(model.predict(testX), axis=1)
    Observations=np.argmax(testY, axis=1)
    report = metrics.classification_report(Observations, Predictions, digits = 3)
    print('Classification report for '+TuningDataName)
    print(report)
    model.save(OutPath+TuningDataName+'.h5')