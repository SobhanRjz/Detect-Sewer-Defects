##### Search Feature Matching similarity between two pic
import numpy as np 
import matplotlib.pyplot as plt
import glob
from cv2 import imread
import cv2
import os
import pickle
from numpy.core.fromnumeric import shape
import seaborn as sns
import pandas as pd
from pandas import concat
from skimage import feature
from skimage.filters import sobel
from skimage.feature import graycomatrix, graycoprops, hog
from skimage.measure import shannon_entropy
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC  
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
from skimage.feature import local_binary_pattern
from scipy.cluster.vq import kmeans,vq
_IsDataFrameExist = False
_IsModelExist = False
OutPutPath = r"H:\Video\PyProject\CombineFeatures\OutPuts"

def hog_feature(image, n_components = 7, orientations= 12, pixels_per_cell = 16, cells_per_block = 2):
    fd = hog(image, orientations = orientations,
            pixels_per_cell = (pixels_per_cell, pixels_per_cell),
            cells_per_block = (cells_per_block, cells_per_block),
            visualize = False, feature_vector = True, channel_axis = 2, block_norm = "L1") # shape : (x,)

    fd=fd.reshape([-1,orientations]).transpose() # shape : (x, y)
    pca = PCA(n_components=n_components,whiten=True)
    newX = pca.fit_transform(fd)
    #print(pca.explained_variance_ratio_)
    return newX.reshape(-1)# 12 * 7 (0.95 percent show presenting variablity in data) # shape : (x,)

def LBP_feature(image):
    METHOD = 'uniform'  # at most two circular 0-1 or 1-0 transitions
    radius = 1  # distance between central pixels and comparison pixels
    n_points = 8 * radius  # define number of comparison pixels
    eps = 1e-7
    GreyImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(GreyImage, n_points, radius, METHOD)

    (hist, _) = np.histogram(lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2))

    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    
    return hist

def GLCM_features(image):
    Angle =  [0, np.pi/4, np.pi/2, 3*np.pi/4] # Radian
    Distance = [0, 1, 3, 5]
    Properties = ['energy', 'correlation', 'dissimilarity', 'homogeneity', 'contrast', 'ASM']
    FeatureArr = np.empty((0,0))
    for prop in Properties:
        for ChNum in range(3): # channel Number
            ColorChImg = image[:,:,ChNum]
            GLCM = graycomatrix(ColorChImg, Distance, Angle, symmetric=True, normed=True)       
            GLCM_feat = graycoprops(GLCM, prop)
            temparr = GLCM_feat.reshape(-1)
            FeatureArr = np.append(FeatureArr, temparr)

            #Another Filters
            entropy = shannon_entropy(ColorChImg)
            FeatureArr = np.append(FeatureArr, np.array(entropy))
    return FeatureArr

def SIFT_features(image, n_components = 0.95):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    descriptors_float = descriptors.astype(float)
    # pca = PCA(n_components = n_components, whiten = True)
    # newX = pca.fit_transform(descriptors_float)
    k = descriptors_float.shape[0]
    Kmean,variance = kmeans(descriptors_float, k, 1)
    hist, bin_edges=np.histogram(Kmean, bins = 256)

    return hist

def ORB_features(image, n_components = 27):
    ORB = cv2.ORB_create()
    keypoints, descriptors = ORB.detectAndCompute(image, None)
    descriptors_float = descriptors.astype(float)
    # pca = PCA(n_components = n_components, whiten = True)
    # newX = pca.fit_transform(descriptors)
    k = descriptors_float.shape[0]
    Kmean,variance = kmeans(descriptors_float, k, 1)
    hist, bin_edges=np.histogram(Kmean, bins = 256)
    return hist

def KAZE_features(image, n_components = 28):
    fast = cv2.FastFeatureDetector_create() 
    # cv2.FeatureDetector_create("STAR")
    kaze = cv2.KAZE_create()
    (key, descriptors) = kaze.detectAndCompute(image, None)

    descriptors_float = descriptors.astype(float)

    k = descriptors_float.shape[0]
    Kmean,variance = kmeans(descriptors_float, k, 1)
    hist, bin_edges=np.histogram(Kmean, bins = 256)

    return hist

def ReshapeArray(array: np.ndarray, size):
    return array.reshape(size, 1).reshape(-1)

class PythonSwitch:
    def Method(self, Method, image):
        default = "Incorrect Method"
        self.image = image
        return getattr(self, Method, lambda: default)()

    def HOG(self):
        return hog_feature(self.image)

    def GLCM(self):
        return GLCM_features(self.image)

    def SIFT(self):
        return SIFT_features(self.image)

    def ORB(self):
        return ORB_features(self.image)

    def KAZE(self):
        return KAZE_features(self.image)

    def LBP(self):
        return LBP_feature(self.image)

def FeatureExtraction(Images :np.ndarray, OutPutName: str, FeatureList: list):
    OutPutPath = r"H:\Video\PyProject\CombineFeatures\OutPuts"
    if _IsDataFrameExist is not True:
        image_dataset = pd.DataFrame()
        i = -1
        FeatureSwitcher = PythonSwitch()
        # Defect
        for image in Images:
            i += 1
            df = pd.DataFrame()
            for ft in FeatureList:
                DataFeat = FeatureSwitcher.Method(ft, image)
                df1 = pd.DataFrame(columns = [ft + str(j) for j in range(DataFeat.size)]) 
                df1.loc[0] = DataFeat
                df = pd.concat([df, df1], axis = 1)

            image_dataset = pd.concat([image_dataset, df])
            # y = x.index == 0
            # np.where(np.array(y) == True)[0]
            # if i == 30:
            #     break
        image_dataset.to_pickle(OutPutPath + "\\" + OutPutName + ".pkl")  

    else :
        image_dataset = pd.read_pickle(OutPutPath + "\\" + OutPutName +".pkl")
        image_dataset = pd.DataFrame(image_dataset).fillna(0)

    return image_dataset

def ReadImages(DefectPath: str, UnDefectPath: str):
    train_images = []
    train_labels = [] 
    SIZE = (500, 300)
    #for directory_path in glob.glob("cell_images/train/*"):
    for img_path in glob.glob(DefectPath + r"\*"):
        label = img_path.split("\\")[-1]
        img = cv2.imread(img_path) #Reading color images
        img = cv2.resize(img, SIZE) #Resize images
        train_images.append(img)
        train_labels.append("Defect")
    
    #for directory_path in glob.glob("cell_images/test/*"): 
    for img_path in glob.glob(UnDefectPath + r"\*"):
        label = img_path.split("\\")[-1]
        img = cv2.imread(img_path) #Reading color images
        img = cv2.resize(img, SIZE) #Resize images
        train_images.append(img)
        train_labels.append("NoDefect")
    return train_images, train_labels
    
def LgbModel(x_train, y_train, x_test, y_test):
    _IsModelExist = False
    #Lgb Model
    #Class names for LGBM start at 0 so reassigning labels from 1,2,3,4 to 0,1,2,3
    le = preprocessing.LabelEncoder()
    d_train = lgb.Dataset(x_train, label = y_train)

    # https://lightgbm.readthedocs.io/en/latest/Parameters.html
    # lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    
    #             'objective':'binary',
    #             'metric': 'binary_logloss',
    #             'num_leaves':100,
    #             'max_depth':10, 
    #             'min_data':1}  #no.of unique values in the target class not inclusive of the end value

    lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    
                'objective':'multiclass',
                'metric': 'multi_logloss',
                'num_leaves':100,
                'max_depth':10,
                'num_class':2} 
    if _IsModelExist is not True:
        lgb_model = lgb.train(lgbm_params, d_train, 100) #50 iterations. Increase iterations for small learning rates
        pickle.dump(lgb_model, open(OutPutPath + "\\" +"LgbModel.sav", 'wb'))
    else:
        lgb_model = pickle.load(open(OutPutPath + "\\" +"LgbModel.sav", 'rb'))

    #Predict on test
    test_for_lgb = np.reshape(x_test, (x_test.shape[0], -1))
    test_prediction = lgb_model.predict(test_for_lgb)
    test_prediction=np.argmax(test_prediction, axis=1)
    #Inverse le transform to get original label back. 
    # test_prediction_bin = le.inverse_transform(test_prediction)

    #Print overall accuracy
    from sklearn import metrics
    print ("Accuracy = ", metrics.accuracy_score(y_test, test_prediction))

def RandomForestModel(x_train, y_train, x_test, y_test):
    #Define the classifier
    RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

    # Fit the model on training data
    if _IsModelExist is not True:
        RF_model.fit(x_train, y_train) #For sklearn no one hot encoding
        pickle.dump(RF_model, open(OutPutPath + "\\" +"RandomForest.sav", 'wb'))
    else:
        RF_model = pickle.load(open(OutPutPath + "\\" +"RandomForest.sav", 'rb'))

    #Predict on test
    test_for_RF = np.reshape(x_test, (x_test.shape[0], -1))
    test_for_RF = test_for_RF.iloc[:,:x_train.shape[1]]
    test_prediction = RF_model.predict(test_for_RF)

    #Print overall accuracy
    print ("Accuracy = ", metrics.accuracy_score(y_test, test_prediction))
    print(confusion_matrix(y_test,test_prediction))
    print(classification_report(y_test,test_prediction))#Output

def SVMModel(x_train, y_train, x_test, y_test):
    # SVM_model = svm.SVC(decision_function_shape='ovo')  #For multiclass classification
    param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf','sigmoid']} # c = 10, gama = 0.001
    SVM_model = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)

    if _IsModelExist is not True:
        SVM_model.fit(x_train, y_train) #For sklearn no one hot encoding
        pickle.dump(SVM_model, open(OutPutPath + "\\" +"SVMModel.sav", 'wb'))
    else:
        SVM_model = pickle.load(open(OutPutPath + "\\" +"SVMModel.sav", 'rb'))

    #Predict on test
    test_for_svm = np.reshape(x_test, (x_test.shape[0], -1))
    test_prediction = SVM_model.predict(test_for_svm)

    #Print overall accuracy
    print(SVM_model.best_estimator_)
    print ("Accuracy = ", metrics.accuracy_score(y_test, test_prediction))
    print(confusion_matrix(y_test,test_prediction))
    print(classification_report(y_test,test_prediction))#Output
    #Print confusion matrix
    cm = confusion_matrix(y_test, test_prediction)
    




def Main():
    
    SIZE = (500, 300)
    DefectPath = r"H:\Video\VideoAbnormalyDetection\Train\Defects" # This is the path of our positive input dataset
    # define the same for negatives
    UnDefectPath= r"H:\Video\VideoAbnormalyDetection\Train\Nodefects"

    train_images, train_labels = ReadImages(DefectPath, UnDefectPath)
    (trainData, testData, trainLabels, testLabels) = train_test_split(
        np.array(train_images), train_labels, test_size=0.20, random_state=42)
    # TrainFeatures
    FeatureList = ["ORB", "KAZE"]
    train_dataset = FeatureExtraction(trainData, "Train_" + "_".join(FeatureList), FeatureList)
    # TestFeatures
    test_features = FeatureExtraction(testData, "Test_" + "_".join(FeatureList), FeatureList)

    
    le = preprocessing.LabelEncoder()
    le.fit(trainLabels)
    train_labels_encoded = le.transform(trainLabels)
    le.fit(testLabels)
    test_labels_encoded = le.transform(testLabels)
    #Split data into test and train datasets (already split but assigning to meaningful convention)
    #If you only have one dataset then split here
    x_train, y_train, x_test, y_test = train_dataset, train_labels_encoded, test_features, test_labels_encoded



    RandomForestModel(x_train, y_train, x_test, y_test)

    SVMModel(x_train, y_train, x_test, y_test)

    LgbModel(x_train, y_train, x_test, y_test)






    # (trainData, testData, trainLabels, testLabels) = train_test_split(
    #     np.array(train_images), train_labels, test_size=0.20, random_state=42)



if __name__ == "__main__":
    Main()