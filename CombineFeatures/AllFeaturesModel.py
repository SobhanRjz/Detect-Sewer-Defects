import pandas as pd

import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import pickle
import seaborn as sns
import pandas as pd
from pandas import concat
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#Resize images to
SIZE = (500, 300)
_IsDataFramExist = True
_IsModelExist = True
#Capture images and labels into arrays.
#Start by creating empty lists.
train_images = []
train_labels = [] 
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3
n_components = 4
#for directory_path in glob.glob("cell_images/train/*"):
for img_path in glob.glob(r"H:\Video\VideoAbnormalyDetection\Train\Defects\*"):
    label = img_path.split("\\")[-1]
    img = cv2.imread(img_path) #Reading color images cv2.imread(img_path, 0)
    fd = hog(img, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True, channel_axis= 2) 
    img = cv2.resize(img, SIZE) #Resize images
    train_images.append(img)
    train_labels.append("Defect")
    pca = PCA(n_components=0.95,whiten=True)
    fd=fd.reshape([-1,orientations]).transpose()
    newX = pca.fit_transform(fd) 
    print(newX.reshape(-1))
    
#for directory_path in glob.glob("cell_images/test/*"): 
for img_path in glob.glob(r"H:\Video\VideoAbnormalyDetection\Train\Nodefects\*"):
    label = img_path.split("\\")[-1]
    img = cv2.imread(img_path, 0) #Reading color images
    img = cv2.resize(img, SIZE) #Resize images
    train_images.append(img)
    train_labels.append("NoDefect")

train_images = np.array(train_images)
train_labels = np.array(train_labels)

(trainData, testData, trainLabels, testLabels) = train_test_split(
        np.array(train_images), train_labels, test_size=0.20, random_state=42)

#Encode labels from text (folder names) to integers.
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(testLabels)
test_labels_encoded = le.transform(testLabels)
le.fit(trainLabels)
train_labels_encoded = le.transform(trainLabels)

#Split data into test and train datasets (already split but assigning to meaningful convention)
#If you only have one dataset then split here
x_train, y_train, x_test, y_test = trainData, train_labels_encoded, testData, test_labels_encoded

# Normalize pixel values to between 0 and 1
#x_train, x_test = x_train / 255.0, x_test / 255.0

###################################################################
# FEATURE EXTRACTOR function
# input shape is (n, x, y, c) - number of images, x, y, and channels
def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  #iterate through each file 
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.
        img = dataset[image, :,:]
    ################################################################
    #START ADDING DATA TO THE DATAFRAME
    #     
        #Full image
        #GLCM = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4]) Radian
        Angle =  [0, np.pi/4, np.pi/2, 3*np.pi/4]
        Distance = [0, 1, 3, 5]
        n = 1
        for dis in Distance:
            for ang in Angle:
                
                GLCM = graycomatrix(img, [dis], [ang])       
                GLCM_Energy = graycoprops(GLCM, 'energy')[0]
                df['Energy' + str(n)] = GLCM_Energy
                GLCM_corr = graycoprops(GLCM, 'correlation')[0]
                df['Corr' + str(n)] = GLCM_corr       
                GLCM_diss = graycoprops(GLCM, 'dissimilarity')[0]
                df['Diss_sim' + str(n)] = GLCM_diss       
                GLCM_hom = graycoprops(GLCM, 'homogeneity')[0]
                df['Homogen' + str(n)] = GLCM_hom       
                GLCM_contr = graycoprops(GLCM, 'contrast')[0]
                df['Contrast' + str(n)] = GLCM_contr

                #Add more filters as needed
                entropy = shannon_entropy(img)
                df['Entropy' + str(n)] = entropy
                n +=1

        #Append features from current image to the dataset
        image_dataset = pd.concat([image_dataset, df])
        
    return image_dataset

#Extract features from training images
OutPutPath = r"H:\Video\PyProject\GLCM\OutPuts"
if _IsDataFramExist is not True:
    image_features = feature_extractor(x_train)
    image_features.to_pickle(OutPutPath + "\\" + "GLCMdataTrain.pkl")

    test_features = feature_extractor(x_test)
    test_features.to_pickle(OutPutPath + "\\" + "GLCMdataTest.pkl")

else:
    image_features = pd.read_pickle(OutPutPath + "\\" + "GLCMdataTrain.pkl")
    test_features = pd.read_pickle(OutPutPath + "\\" + "GLCMdataTest.pkl")
    

X_for_ML =image_features
#Reshape to a vector for Random Forest / SVM training
n_features = image_features.shape[1]
image_features = np.expand_dims(image_features, axis=0)
X_for_ML = np.reshape(image_features, (x_train.shape[0], -1))  #Reshape to #images, features

test_features = np.expand_dims(test_features, axis=0)
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))
#Define the classifier
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

#Can also use SVM but RF is faster and may be more accurate.
#from sklearn import svm
#SVM_model = svm.SVC(decision_function_shape='ovo')  #For multiclass classification
#SVM_model.fit(X_for_ML, y_train)

# Fit the model on training data
if _IsModelExist is not True:
    RF_model.fit(X_for_ML, y_train) #For sklearn no one hot encoding
    pickle.dump(RF_model, open(OutPutPath + "\\" +"RandomForest.sav", 'wb'))
else:
    RF_model = pickle.load(open(OutPutPath + "\\" +"RandomForest.sav", 'rb'))

#Predict on test
test_prediction = RF_model.predict(test_for_RF)
# test_prediction=np.argmax(test_prediction, axis=0)
#Inverse le transform to get original label back. 
test_prediction = le.inverse_transform(test_prediction)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(testLabels, test_prediction))

#Print confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(testLabels, test_prediction)

fig, ax = plt.subplots(figsize=(6,6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)
import random
n=random.randint(0, x_test.shape[0]-1) #Select the index of image to be loaded for testing
img = x_test[n]
plt.imshow(img)

#Extract features and reshape to right dimensions
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_features=feature_extractor(input_img)
input_img_features = np.expand_dims(input_img_features, axis=0)
input_img_for_RF = np.reshape(input_img_features, (input_img.shape[0], -1))
#Predict
img_prediction = RF_model.predict(input_img_for_RF)
img_prediction=np.argmax(img_prediction, axis=0)
img_prediction = le.inverse_transform([img_prediction])  #Reverse the label encoder to original name
print("The prediction for this image is: ", img_prediction)
print("The actual label for this image is: ", testLabels[n])




#Lgb Model
import lightgbm as lgb
 #Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3
d_train = lgb.Dataset(X_for_ML, label=y_train)

# https://lightgbm.readthedocs.io/en/latest/Parameters.html
lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    
              'objective':'multiclass',
              'metric': 'multi_logloss',
              'num_leaves':100,
              'max_depth':10,
              'num_class':4}  #no.of unique values in the target class not inclusive of the end value

if _IsModelExist is not True:
    lgb_model = lgb.train(lgbm_params, d_train, 100) #50 iterations. Increase iterations for small learning rates
    pickle.dump(lgb_model, open(OutPutPath + "\\" +"lgb.sav", 'wb'))
else:
    lgb_model = pickle.load(open(OutPutPath + "\\" +"lgb.sav", 'rb'))

#Predict on test
test_for_RF = np.reshape(test_features, (x_test.shape[0], -1))
test_prediction = lgb_model.predict(test_for_RF)
test_prediction=np.argmax(test_prediction, axis=1)
#Inverse le transform to get original label back. 
test_prediction = le.inverse_transform(test_prediction)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(testLabels, test_prediction))
