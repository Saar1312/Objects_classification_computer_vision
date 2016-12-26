##################################################################################################################
#	University of Porto 
#	Faculty of Engineering
#	Computer Vision
#
# Project 2: Object Recognition
#
# Authors:
#	* Nuno Granja Fernandes up201107699
#	* Samuel Arleo Rodriguez up201600802
#	* Katja Hader up201602072
##################################################################################################################

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join

# Path to the training data
path = "/home/samuel/CV2/Data2/"

# Creating a list with all pictures names
onlyfiles = [ f for f in listdir(path) if isfile(join(path,f)) ]

# Creating a numpy empty array with length = amount of images
images = np.empty(len(onlyfiles), dtype=object)

# Reading each image and storing them into the images array
for n in range(0, len(onlyfiles)):
	images[n] = cv2.imread(join(path,onlyfiles[n]))

# Changing all images to grayscale
grays = map(lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2GRAY),images)

# Instantiating sift class
sift = cv2.xfeatures2d.SIFT_create()

# Applying SIFT to all training images. This returns the tuple (keypoints, descriptor)
# for each image, and it's converted to a matrix with columns:
# |Keypoints| Descriptors|
res = np.asarray(map(lambda x: sift.detectAndCompute(x, None), grays))

# Storing all keypoints (pixels coordinates of keepoints) in a single variable
kp = res[:,0]
# Storing all descriptor in a single variable to cluster them
des = res[:,1]

#img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.jpg',img)