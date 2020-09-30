# SEE_ICE
Classification of Glacial Landscapes using CNN-Supervised Classification (CSC) :snowflake:


## Description

The methods and code provided here allows pixel-level semantic classification of Sentinel-2 imagery, adapted from the CSC workflow originally designed to classify aerial imagery of rivers (original source code: https://github.com/geojames/CNN-Supervised-Classification). It's intended application is the classification of glacial landscapes and was specifically trained on imagery containing marine-terminating outlet glaciers in Greenland using seven classes:
1. Open Water
1. Bergy Water
1. Mélange
1. Glacier Ice
1. Snow on Ice
1. Snow on Rock
1. Rock

## Dependencies
* Keras (We used Tensorflow version 2.1)
* Scikit-Learn
* Scikit-Image
* Pandas
* Numpy
* Matplotlib

We used the Anaconda distribution of Python 3 (written in version 3.7) which installs all the needed libraries except tensorflow.

## Application
### Training VGG16 Convolutional Neural Networks
#### Step 1: Tile Data Preparation

_Sentinel 2 Imagery_


