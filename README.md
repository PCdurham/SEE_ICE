# SEE_ICE
Classification of Glacial Landscapes using CNN-Supervised Classification (CSC) ([Carbonneau et al. 2020](https://www.sciencedirect.com/science/article/pii/S0034425720304806)) :snowflake:

## March 2021 Update
Added:
- Tensorflow v2.4 (Python 3.8) with support for mixed precision training on RTX GPUs
- Alternative workflow where tensors can be saved to disk as large 4D numpy arrays and used in CNN training.  
- Geotif outputs when input is itself a geotif
- Calving front detection sub-routine based on morphological active contours as implemented in scikit-image
- Additional validation functionality for calving fronts and valley margin edges

## Key Dependencies
- Python 3.8
- Tensorflow 2.4
- gdal
- scikit-image
- scikit-learn

## Description

The methods and code provided here allow pixel-level semantic classification of Sentinel-2 imagery, adapted from the CSC workflow originally designed to classify aerial imagery of rivers (original source code: https://github.com/geojames/CNN-Supervised-Classification). Its intended application is the classification of glacial landscapes and phase one CNNs have been trained specifically on imagery containing marine-terminating outlet glaciers in Greenland, using the following seven classes:
1. Open Water
1. Bergy Water
1. Mélange
1. Glacier Ice
1. Snow on Ice
1. Snow on Rock
1. Rock

 
## Application
### Training VGG16 Convolutional Neural Networks
#### Step 1: Data Preparation

_**Sentinel 2 Imagery**_ The code was designed for application on  Sentinel-2 imagery (Bands 4, 3, 2, and 8) which should be combined into composite four-band images and cropped to the desired training area. Training also requires a class raster (created manually using rasterised polygons) composed of the seven classes detailed in the **Description** section above. This can be done using GIS software (e.g. QGIS/ArcGIS).

_**Image Tiles**_ The scripts for phase one CNN training require input data in the form of tiled 4D tensors. Cropped Sentinel-2 images and associated class rasters can be tiled, labelled, and augmented using the [TilePreparation_CNNTrainingData.py](https://github.com/PCdurham/SEE_ICE/blob/master/code/TilePreparation_CNNTrainingData.py) script which tiles input images according to a specified tile size and stride, and randomly allocates them to training and validation data folders primed for phase one CNN training.

_**Saved Tensors**_ If 32 Gb or more are RAM is available (recommend 64 Gb), a single tensor can be compiled and saved to disk as a large 4D numpy array. Use the script thin_Tiles.py to select a random subset of tiles that will be balanced for each class.  Then run Compile_tensor to create both tensor and label .npy arrays save to disk.

#### Step 2: Phase One CNN Training

Following the organisation of model training data, the [Phase1_VGG16_RGB-RGBNIR.py](https://github.com/PCdurham/SEE_ICE/blob/master/code/Phase1_VGG16_RGB-RGBNIR.py) and [Phase1_VGG16_TransferLearning.py](https://github.com/PCdurham/SEE_ICE/blob/master/code/Phase1_VGG16_TransferLearning.py) scripts can be applied to train the adapted VGG16 architectures with either RGB or RGB+NIR bands and with or without transfer learning using ImageNet weights. User inputs are at the beginning of the script and show a list of variables initially labelled as 'Path' or 'Empty', these should be edited according to the users folder and file names. Following training, the model is saved and can be used in the CSC workflow without further training.

Alterntatively, if you have saved a tensor and label as numpy arrays, you can run train_precompiled_CNN to produce a phase model from this data.

### Executing CSC
#### Step 1: Tiling Imagery for Application
The [TilePreparation_CSCApplicationData.py](https://github.com/MMarochov/SEE_ICE/blob/master/code/TilePreparation_CSCApplicationData.py) script allows users to tile unseen Sentinel-2 tiles and class rasters to a desired size for processing by the CSC workflow. The input images are required to be in composite band (RGBNIR) format as with the tiling script for phase one CNN training data. 

#### Step 2: CSC Application
The pre-trained phase one CNN can be implemented to create image-specific training data using the [CNNSupervisedClassification_SEE_ICE.py](https://github.com/MMarochov/SEE_ICE/blob/master/code/CNNSupervisedClassification_SEE_ICE.py) CSC script to create a pixel-level classification of unseen images of glacial landscapes. The CSC script will classify all the images in the `PredictPath` folder. If users want to apply the workflow to a specific glacier or study area, image datasets for each study area should be organised in separate folders. The CSC script can be applied using either a pixel- or patch-based approach of classification by altering the `Kernel_size`  variable. A Multilayer Perceptron (MLP) is used if the `Kernel_size` is set to 1, meaning classification is based on the properties of a single pixel. Alternatively, if `Kernel_size` is > 1 a compact CNN (cCNN) is applied and uses a window of pixels to predict the class of the central pixel (therefore the `Kernel_size` must always be an odd number). The script will execute and output performance metrics for each image. csv files with a CNN_ prefix give performance metrics for the phase one CNN model with F1 scores and support (# of pixels) for each class. CSC_ files give the same metrics for the final CSC result after the application of either the MLP or cCNN. A 4-part figure will also be output showing the original image, the existing class labels, the CNN classification and the final CSC classification. Optionally, a saved class raster can also be saved to disk for each processed image (e.g. Figure 1) with the geotif script.  

If the parameter space exploration is required for tile size and patch size variables, use the batch script.  this will need a csv as an input where each line has the paramenters for a given job.  required columns as as follows:

- **Model** gives the name of the model. Do not add the h5 extension and don't use a full path.  Folder is is a script input
- **DataSource** gives the folder where the input files are located.  these are assumed to be the 4 band tif image, a validation class raster and an edge raster.
- **OutFolder** gives the partial name of the output folder where results will be written.  Do not add a final / (or \\).  the script will append the patch size to the name and create this folder.
- **EportShortHand** gives a short name for the experiment added into some results file name.
- **Dims** gives the bands (number of channels) for the experiement.  Use 3 for RGB and 4 for NIR+RGB.
- **PatchSize** gives the size of the local phase 2 compact CNN or MLP.  Use 1 for the MLP and 3,5,7 or 15 for the compact CNN 
- **TileSize** gives the XY tile size (eg 50, 75 or 100) expected by the pre-trained CNN model.
-  **WholeImage** is 1 or 0.  Set to 1 if your computer has sufficient RAM to classify a whole image.  Note that for the compact CNN, memory usage increases fast.  EG a 15x15 patch size on a 3000x3000x4 Seninel 2 sub0image will need in excess of 100 Gb of ram to keep the ca. [9000000, 15,15,4] tensor.
-  



![CSC_HelehimS2A5_Patch 7](https://user-images.githubusercontent.com/60142411/94746470-ad173100-0374-11eb-93ec-99b80870c6be.png)
_**Figure 1: CSC output showing a sample image of Helheim Glacier, eastern Greenland.**_ 

## Authors
* Patrice Carbonneau
* Melanie Marochov

## Citation
CSC workflow: 
Carbonneau, P. E., Dugdale, S. J., Breckon, T. P., Dietrich, J. T., Fonstad, M. A., Miyamoto, H. and Woodget, A. S.: Adopting deep learning methods for airborne RGB fluvial scene classification, Remote Sensing of Environment, 251, 112107, doi:https://doi.org/10.1016/j.rse.2020.112107, 2020.

The work for CSC adaptation for glacial landscapes containing marine terminating outlet glaciers in Greenland has not yet been published, but a poster is available [here](https://presentations.copernicus.org/EGU2020/EGU2020-19996_presentation.pdf). 




