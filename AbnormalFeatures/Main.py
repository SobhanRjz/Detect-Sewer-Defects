import cv2
import os
import image_similarity_measures.image_similarity_measures 
from sys import argv
import numpy as np
from image_similarity_measures.image_similarity_measures.quality_metrics import ssim as ssimi
from image_similarity_measures.image_similarity_measures.quality_metrics import rmse, sre
from skimage.metrics import structural_similarity as ssim 
from skimage.metrics import mean_squared_error

OriginalPath = r'H:\Video\PyProject\AbnormalFeatures\Images\NoDefect_3.jpg'
OriginalImage = cv2.imread(OriginalPath)

ssim_measures = {}
mse_measures = {}
ssim2_measures = {}
rmse_measures = {}
sre_measures = {}

scale_percent = 100 # percent of original img size
width = int(OriginalImage.shape[1] * scale_percent / 100)
height = int(OriginalImage.shape[0] * scale_percent / 100)
dim = (width, height)

data_dir = r'H:\Video\PyProject\AbnormalFeatures\Images'
sift = cv2.SIFT_create()
keypoints_Org, descriptors_Org = sift.detectAndCompute(OriginalImage, None)


for file in os.listdir(data_dir):
    img_path = os.path.join(data_dir, file)
    data_img = cv2.imread(img_path)
    TestImage = cv2.resize(data_img, dim, interpolation= cv2.INTER_AREA)
    # ssim_measures[img_path] = ssim(OriginalImage, resized_img, channel_axis= 2)
    mse_measures[img_path] = mean_squared_error(OriginalImage, TestImage)
    ssim2_measures[img_path] = ssimi(OriginalImage, TestImage)
    # rmse_measures[img_path] = rmse(OriginalImage, resized_img)
    sift = cv2.SIFT_create()
    keypoints_TestImage, descriptors_TestImage = sift.detectAndCompute(TestImage, None)

    Index_params = dict(algorithm = 0, trees = 5)
    Search_params = dict()
    flann = cv2.FlannBasedMatcher(Index_params, Search_params)

    Matches = flann.knnMatch(descriptors_Org, descriptors_TestImage, k = 2)
    GoodPoint =[]
    for m, n in Matches:
        if m.distance < 0.6 * n.distance:
            GoodPoint.append(m)
    NumberKeyPoint = min(len(descriptors_Org), len(descriptors_TestImage))
    Similarity = len(GoodPoint) / NumberKeyPoint * 100
    print(Similarity)






def calc_closest_val(dict, checkMax):
    result = {}
    if (checkMax):
        closest = max(dict.values())
    else:
    	closest = min(dict.values())

    for key, value in dict.items():
        print("The difference between ", key ," and the original image is : \n", value)

        if (value == closest):
            result[key] = closest
    	    
    print("The closest value: ", closest)	    
    print("######################################################################")
    return result
    
ssim = calc_closest_val(ssim_measures, True)
rmse = calc_closest_val(rmse_measures, False)
sre = calc_closest_val(sre_measures, True)

print("The most similar according to SSIM: " , ssim)
print("The most similar according to RMSE: " , rmse)
print("The most similar according to SRE: " , sre)