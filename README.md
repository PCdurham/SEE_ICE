# SEE_ICE
Classification of Glacial Landscapes using CNN-Supervised Classification (CSC) ([Carbonneau et al. 2020](https://www.sciencedirect.com/science/article/pii/S0034425720304806)) :snowflake:


## Description

The methods and code provided here allows pixel-level semantic classification of Sentinel-2 imagery, adapted from the CSC workflow originally designed to classify aerial imagery of rivers (original source code: https://github.com/geojames/CNN-Supervised-Classification). Its intended application is the classification of glacial landscapes and was specifically trained on imagery containing marine-terminating outlet glaciers in Greenland using seven classes:
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
 
We used the [Anaconda distribution](https://www.anaconda.com/products/individual) of Python 3 (written in version 3.7) which installs all the needed libraries except tensorflow.

## Application
### Training VGG16 Convolutional Neural Networks
#### Step 1: Data Preparation

_**Sentinel 2 Imagery**_ The code was designed for application on  Sentinel-2 imagery (Bands 4, 3, 2, and 8) which should be combined into composite four-band images and cropped to the desired training area. Training also requires a class raster (created manually using rasterised polygons) composed of the seven classes detailed in the **Description** section above. This can be done using GIS software (e.g. QGIS/ArcGIS).

_**Image Tiles**_ The scripts for phase one CNN training require input data in the form of tiled 4D tensors. Cropped Sentinel-2 images and associated class rasters can be tiled, labelled, and augmented using the [TilePreparation_CNNTrainingData.py](https://github.com/PCdurham/SEE_ICE/blob/master/code/TilePreparation_CNNTrainingData.py) script which tiles input images according to a specified tile size and stride, and randomly allocates them to training and validation data folders primed for phase one CNN training.

#### Step 2: Phase One CNN Training

Following the organisation of model training data, the [Phase1_VGG16_RGB-RGBNIR.py](https://github.com/PCdurham/SEE_ICE/blob/master/code/Phase1_VGG16_RGB-RGBNIR.py) and [Phase1_VGG16_TransferLearning.py](https://github.com/PCdurham/SEE_ICE/blob/master/code/Phase1_VGG16_TransferLearning.py) scripts can be applied to train the adapted VGG16 architectures with either RGB or RGB+NIR bands and with or without transfer learning using ImageNet weights. User inputs are at the beginning of the script and show a list of variables initially labelled as 'Path' or 'Empty', these should be edited according to the users folder and file names. Following training, the model is saved and can be used in the CSC workflow without further training.

### Executing CSC
#### Step 1: Tiling Imagery for Application
The [TilePreparation_CSCApplicationData.py](https://github.com/MMarochov/SEE_ICE/blob/master/code/TilePreparation_CSCApplicationData.py) script allows users to tile unseen Sentinel-2 tiles and class rasters to a desired size for processing by the CSC workflow. The input images are required to be in composite band (RGBNIR) format as with the tiling script for phase one CNN training data. 

#### Step 2: CSC Application
The pre-trained phase one CNN can be implemented to create image-specific training data using the [CNNSupervisedClassification_SEE_ICE.py](https://github.com/MMarochov/SEE_ICE/blob/master/code/CNNSupervisedClassification_SEE_ICE.py) CSC script to create a pixel-level classification of unseen images of glacial landscapes. The CSC script can be applied using either a pixel- or patch-based approach of classification by altering the `Kernel_size`  variable. A Multilayer Perceptron (MLP) is used if the `Kernel_size` is set to 1, meaning classification is based on the properties of a single pixel. Alternatively, if `Kernel_size` is > 1 a compact CNN (cCNN) is applied and uses a window of pixels to predict the class of the central pixel (therefore the `Kernel_size` must always be an odd number). The script will execute and output performance metrics for each image. csv files with a CNN_ prefix give performance metrics for the phase one CNN model with F1 scores and support (# of pixels) for each class. CSC_ files give the same metrics for the final CSC result after the application of either the MLP or cCNN. A 4-part figure will also be output showing the original image, the existing class labels, the CNN classification and the final CSC classification. Optionally, a saved class raster can also be saved to disk for each processed image.



