##################################################################################################################
#	University of Porto 
#	Faculty of Engineering
#	Computer Vision
#
# Project 2: Object Recognition
#
# Authors:
#	* Katja Hader up201602072
#	* Nuno Granja Fernandes up201107699
#	* Samuel Arleo Rodriguez up201600802
##################################################################################################################

import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import csv
import classification, representation, featurex

"""
def resize(des):
	counter = 0
	for i in des.shape[0]:
		if des[i] is not None:
			des[i] = des[i].reshape(1,-1)
			counter += 1
	np.array()
	return 
"""

def load_images(data):
	try:
		# Creating a list with all pictures names
		onlyfiles = [ f for f in listdir(data) if isfile(join(data,f)) ][:limit]
		images = np.empty(len(onlyfiles), dtype=object)
		# Reading each image and storing it into the images array
		for n in range(0, len(onlyfiles)):
			images[n] = cv2.imread(join(data,onlyfiles[n]))
		# Changing all images to grayscale
		return map(lambda x: cv2.cvtColor(x,cv2.COLOR_BGR2GRAY),images)
	except:
		print "Error opening the folder ",data,".Please check the file location."
		exit()

def load_labels(data):
	try:
		labels = np.genfromtxt(data, delimiter=',',dtype=[('id','<i8'),('label','|S5')], skip_header=1)
		labels = map(lambda (x,y):[y], labels) 	# Taking out image ids. We don't need the label id 
											   	# because it is implicit in the position in the list
		return np.asarray(labels)[:limit,:] 	# It is a list of tuples and we need it as an array of arrays
	except:
		print "Error opening the file ",data,".Please check the file location."
		exit()

def join_desc(data):
	# Data has the columns |Keypoint|Descriptors| and each row represent a keypoint
	# tmp stores just the descriptors
	tmp = [res[i][1] for i in range(0,len(res)) if res[i][1] is not None]
	# Getting descriptors size (all have the same given by SIFT: 128)
	desc_size = tmp[0][0].shape[0]
	# Counting number of descriptors
	num_desc = 0
	for img in tmp:
		for desc in img:
			num_desc += 1
	# Storing descriptors in des, but before we create it empty with the correct dimensions: [num_desc,128]
	des = np.zeros((num_desc,desc_size))
	n = 0
	for img in tmp:
		for desc in img:
			des[n,:] = desc
			n += 1
	return des

#----------------------- LOADING DATA --------------------------
# Paths to the training and test data
train_imgs = "/home/samuel/CV2/train_data/"
test_imgs = "/home/samuel/CV2/test_data/"

# File with labels of training images
train_labels = "labels_train.csv"

# Using a subset of the images set
limit = 1000

# Loading training images and labels
images_tr = load_images(train_imgs)
labels_tr = load_labels(train_labels)

#------------------- EXTRACTING FEATURES -----------------------
# Instantiating sift class
sift = cv2.xfeatures2d.SIFT_create()

# Applying SIFT to all training images. This returns the tuple (keypoints, descriptor)
# for each image, and it's transformed to a matrix with columns:
# |Keypoints| Descriptors|
res = map(lambda x: sift.detectAndCompute(x, None), images_tr)

# Storing all descriptor in a single variable to cluster them
desc = join_desc(res)

# Initiate kNN, train the data, then test it with test data for k=1
#print(labels_tr.shape,des.shape)
#------------------- REPRESENTATION STEP -----------------------
des = [res[i][1] for i in range(0,)]


#------------------- CLASSIFYING IMAGES ------------------------

knn = cv2.ml.KNearest_create()
knn.train(des,cv2.ml.ROW_SAMPLE,labels_tr)
#knn.train(des,cv2.ml.ROW_SAMPLE,labels_tr)
#ret,result,neighbours,dist = knn.find_nearest(test,k=5)

#img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.jpg',img)"""
