# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 09:28:26 2021

@author: Patrice
"""
""" Libraries"""
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os





'''User Inputs'''
TensorPath = '/media/patrice/DataDrive/SEE_ICE/JointTrain/JointTensor5k_T_float16.npy'  #location of the tile tensor
LabelPath = '/media/patrice/DataDrive/SEE_ICE/JointTrain/JointTensor5k_L_float16.npy' #location of the labels
InitialModel='/media/patrice/DataDrive/SEE_ICE/Models/VGG16_50_RGBNIRfloat16_99acc.h5'
OutPath='/media/patrice/DataDrive/SEE_ICE/JointTrain/'  
TuningDataName='VGG16_50_Joint_RGBNIRfloat16_995acc'
NClasses=7
NIR=True
Tilesize=50
batches=25 #try 200 for 50x50 tiles
targetacc=0.995
LowLearn=0.00001


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





'''Load initial CNN'''
model=tf.keras.models.load_model(InitialModel)
model.save_weights(os.path.dirname(InitialModel)+'W_'+os.path.basename(InitialModel))

'''Create copy of the CNN with a new learning rate'''
inShape = Tensor.shape[1:]

CNN_base = VGG16(include_top=False,weights=None,input_shape=inShape)



model = Sequential()
model.add((CNN_base))
model.add(Flatten())
#Estimator.add(Dense(32,kernel_regularizer=regularizers.l2(0.001),kernel_initializer='normal',activation='relu'))
model.add(Dense(512, activation='relu',kernel_regularizer= regularizers.l2(0.001)))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu',kernel_regularizer= regularizers.l2(0.001)))    
model.add(Dense(NClasses+1,kernel_initializer='normal',activation='softmax', dtype='float32'))


#Tune an optimiser but with a new LR
Optim = optimizers.Adam(lr=LowLearn,beta_1=0.9,beta_2=0.999,decay=0.0,amsgrad=True)

# Compile model
model.compile(loss='categorical_crossentropy',optimizer=Optim,metrics=['accuracy'])
model.load_weights(os.path.dirname(InitialModel)+'W_'+os.path.basename(InitialModel))
model.summary()




'''Tune and Fit CNN'''
Labels1Hot = to_categorical(Label)

(trainX, testX, trainY, testY) = train_test_split(Tensor, Labels1Hot, test_size=0.2)
when2stop = MyThresholdCallback(threshold=targetacc)
history = model.fit(trainX, trainY, epochs = 120, batch_size = batches, validation_data = (testX, testY), callbacks=when2stop)
Predictions=np.argmax(model.predict(testX), axis=1)
Observations=np.argmax(testY, axis=1)
report = metrics.classification_report(Observations, Predictions, digits = 3)
print('Classification report for '+TuningDataName)
print(report)
model.save(OutPath+TuningDataName+'.h5')

#close the TF session
session.close()