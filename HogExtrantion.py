from skimage._shared.utils import channel_as_last_axis
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import imutils
import cv2

#reading the image
imgpath = 'H:\\Video\\VideoAbnormalyDetection\\Train\Defects\\2_Joints_L1.27.M291-L1.P27.M292_2018.06.20_D 0084 (59).jpg'

image = imread(imgpath)
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1),
                     visualize=True, block_norm= "L2",channel_axis= 3)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis('off')
ax1.imshow(image)
ax1.set_title('Input image')
ax1.set_adjustable('box')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))

ax2.axis('off')
ax2.imshow(hog_image_rescaled)
ax2.set_title('Histogram of Oriented Gradients')
ax1.set_adjustable('box')
plt.show()

# img = imread(imgpath)
# imshow(img)
# image = cv2.imread(imgpath, 1)
# cv2.imshow('dd', image)
# cv2.waitKey(0)
# print(img.shape)
# fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8), 
#                     cells_per_block=(2, 2), visualize=True, channel_axis = 2, block_norm= "L1")
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
# cv2.imshow('dd', hog_image_rescaled)

# def s_x(img):
#     kernel = np.array([[-1, 0, 1]])
#     imgx = signal.convolve2d(img, kernel, boundary='symm', mode='same')
#     return imgx
# def s_y(img):
#     kernel = np.array([[-1, 0, 1]]).T
#     imgy = signal.convolve2d(img, kernel, boundary='symm', mode='same')
#     return imgy
 
# def grad(img):
#     imgx = s_x(img)
#     imgy = s_y(img)
#     s = np.sqrt(imgx**2 + imgy**2)
#     theta = np.arctan2(imgx, imgy) #imgy, imgx)
#     theta[theta<0] = np.pi + theta[theta<0]
#     return (s, theta)

print(fd.shape)