import json
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image # This will be used to read/modify images (can be done via OpenCV too)
from numpy import *

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3


# define path to images:

DefectPath = r"H:\Video\VideoAbnormalyDetection\Train\Defects" # This is the path of our positive input dataset
# define the same for negatives
UnDefectPath= r"H:\Video\VideoAbnormalyDetection\Train\Nodefects"

# read the image files:
DefectListing = os.listdir(DefectPath) # it will read all the files in the positive image path (so all the required images)
UnDefectListing = os.listdir(UnDefectPath)
num_pos_samples = size(DefectListing) # simply states the total no. of images
num_neg_samples = size(UnDefectListing)
print(num_pos_samples) # prints the number value of the no.of samples in positive dataset
print(num_neg_samples)
data= []
labels = []

def padarray(A, size):
    t = size - len(A)
    return np.pad(A, pad_width=(0, t), mode='constant')

# Saving Data into the json file
FeaturesDescriptionName = "HOG"
_IsVectorExist = True
_IsModelExist = True
if _IsVectorExist is not True:
    # compute HOG features and label them:
    imageobj = None
    for file in DefectListing: #this loop enables reading the files in the pos_im_listing variable one by one
        # pth = Image.open(DefectPath + '\\' + file) # open the file
        #img = img.resize((64,128))
        try:
            image = imread(DefectPath + '\\' + file) # convert the image into single channel i.e. RGB to grayscale
            # image = image.resize((256,128))
            # calculate HOG for positive features
            fd = hog(image, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True, channel_axis=2)# fd= feature descriptor
            data.append(fd)
            labels.append(1)
        except:
            continue
    # Same for the negative images
    for file in UnDefectListing:
        #img = img.resize((64,128))
        try:
            image = imread(UnDefectPath + '\\' + file)
            # image = image.resize((256,128))
            # Now we calculate the HOG for negative features
            fd = hog(image, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True, channel_axis= 2) 
            data.append(fd)
            labels.append(0)
        except:
            continue

    # Padding All array and fit every element in same size
    templist = []
    [templist.append(i.shape[0]) for i in data]
    maxarraysize = max(templist)
    for i in range(0,len(data)):
        if data[i].shape[0] < maxarraysize:
            data[i] = padarray(data[i], maxarraysize)
    np.savez_compressed(r'H:\Video\PyProject\HogDetect\ModelOutPut\HogVectors', data=data, labels = labels)
else:
    VectorPath = r'H:\Video\PyProject\HogDetect\ModelOutPut\HogVectors.npz'
    loaded = np.load(VectorPath)
    data = loaded['data']
    labels = loaded['labels']


if _IsModelExist is not True:
    # encode the labels, converting them from strings to integers 509436
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    #%%
    # Partitioning the data into training and testing splits, using 80%
    # of the data for training and the remaining 20% for testing
    print(" Constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        np.array(data), labels, test_size=0.20, random_state=42)
    #%% Train the linear SVM
    print(" Training Linear SVM classifier...")
    model = LinearSVC()
    model.fit(trainData, trainLabels)
    #%% Save the Model
    filename = r'H:\Video\PyProject\HogDetect\ModelOutPut\SvmModel.sav'
    pickle.dump(model, open(filename, 'wb'))
else:
    print(" Constructing training/testing split...")
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        np.array(data), labels, test_size=0.20, random_state=42)
    ModelPath = r'H:\Video\PyProject\HogDetect\ModelOutPut\SvmModel.sav'
    model = pickle.load(open(ModelPath, 'rb'))
#%% Evaluate the classifier
print(" Evaluating classifier on test data ...")
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))





