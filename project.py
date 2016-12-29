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
import time

class descriptor:
	def __init__(self, vector=None, label=None):
		self.vector = vector
		self.label = label

class image:
	def __init__(self, img=None, keyp=None, desc=None, hist=None):
		self.img = img
		self.id = None
		self.label = None
		self.desc = desc
		self.keyp = keyp
		self.histogram = hist

class images_set:
	def __init__(self, path_img=None, path_lab=None):
		self.images = None
		self.path_img = path_img
		self.path_lab = path_lab

	def load_images(self,limit):
		try:
			# Creating a list with all pictures names
			onlyfiles = [ f for f in listdir(self.path_img) if isfile(join(self.path_img,f)) ][:limit]
			self.images = np.empty(len(onlyfiles), dtype=object)
			# Reading each image and storing it into the images array
			for n in range(0, len(onlyfiles)):
				self.images[n] = image(cv2.imread(join(self.path_img,onlyfiles[n]),0))
		except:
			print "Error opening the folder ",self.path_img,".Please check the file location."
			exit()

	def load_labels(self,limit):
		try:
			# Loading labels into an matrix with columns |id|label|
			labels = np.genfromtxt(self.path_lab, delimiter=',',
				dtype=[('id','<i8'),('label','|S5')], skip_header=1)
		except:
			print "Error opening the file ",self.path_lab,".Please check the file location."
			exit()
		img = 0
		# Adding id and label to each image
		for (x,y) in labels:
			self.images[img].id = x
			self.images[img].label = y
			if img == limit-1:
				break
			img += 1

	def build_desc(self, desc_list):
		return map(lambda x: descriptor(x), desc_list)

	# Assings descriptors and keypoints to the correspondent image
	def get_features(self, features):
		for i in range(0,len(features)):
			if features[i][1] is not None:
				self.images[i].desc = self.build_desc(features[i][1])
				self.images[i].keyp = features[i][0]

	def set_descr_labels(self,labels,desc_size):
		index = 0
		for img in self.images:
			img.histogram = np.zeros(desc_size, dtype=object)
			if img.desc is not None:		# Discard images without keypoints
				for desc in img.desc:
					label = labels[index][0]
					desc.label = label
					img.histogram[label] += 1
					index += 1

def join_desc(res):
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
path_train_imgs = "/home/samuel/CV2/train_data/"
path_test_imgs = "/home/samuel/CV2/test_data/"

# File with labels of training images
train_labels = "/home/samuel/CV2/labels_train.csv"
test_labels = ""

# Using a subset of the images set
limit = 1000

# Creating object images_set that encapsulates methods for loading images and labels,
# and also stores the loaded images and labels
trainSet = images_set(path_train_imgs, train_labels)

# Loading limit-number of images 
trainSet.load_images(limit)

# Loading labels of previously loaded pictures
trainSet.load_labels(limit)

#------------------- EXTRACTING FEATURES -----------------------

# Instantiating sift class
sift = cv2.xfeatures2d.SIFT_create()

# Applying SIFT to all training images. This returns the tuple (keypoints, descriptor)
# for each image, and it's transformed to a matrix with columns:
# |Keypoints| Descriptors|
res = map(lambda x: sift.detectAndCompute(x.img, None), trainSet.images)

# Storing each descriptor and keypoint with its image
trainSet.get_features(res)

# Storing all descriptors in a single variable to cluster them
desc = join_desc(res)

# Changing type to float32 which is required by the kmeans function
desc = desc.astype('float32')
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# Number of cluster that will set the number of words of each bag
words_number = 250

start = time.time()

# Apllying k-means to all the descriptors
ret,label,center=cv2.kmeans(desc,words_number,None,criteria,4,cv2.KMEANS_RANDOM_CENTERS)

end = time.time()
print(end - start)


print(label,label.shape)
# Giving a label to each descriptor. Also passing size of descriptors: desc[0].shape[0]
trainSet.set_descr_labels(label, words_number)

# To try it uncomment:
# a = trainSet.images[0]
#print(a.histogram)
#print(a.desc[2].label)

#------------------- REPRESENTATION STEP -----------------------




#------------------- CLASSIFYING IMAGES ------------------------
# Initiate kNN, train the data, then test it with test data for k=1
#knn = cv2.ml.KNearest_create()
#knn.train(des,cv2.ml.ROW_SAMPLE,labels_tr)
#knn.train(des,cv2.ml.ROW_SAMPLE,labels_tr)
#ret,result,neighbours,dist = knn.find_nearest(test,k=5)

#img=cv2.drawKeypoints(gray,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.jpg',img)