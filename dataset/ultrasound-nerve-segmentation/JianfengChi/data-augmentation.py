# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Any results you write to the current directory are saved as output.
import numpy as np
from six.moves import cPickle as pickle
import os
import glob
import math
import datetime
import cv2
from random import randint
import string
#TODO install keras and uncomment from keras.preprocessing.image import ImageDataGenerator

'''
This script helps to generate more croped nerves images.
I tried it with three steps:
1) in folder ./input/generatedImages/ I placed photoshoped 
nerve images where nerve structures is more clearly extracted (size70*60)
2) I extracted elipse information about every mask and stored it into ./pickleFiles/elips.pickle
and then croped nerves with various positions in 70*60 window (see more in method extractNerves).
This part wont work if don't have elipse information (method getElipseData)
3) take images where is no nerves and randomly crop images 70*60 (optimal size)


IMPORTANT - before run script uncomment or view TODO places in code
'''

class Config():
	def __init__(self):
		self.pickle_file = './pickleFiles/elips.pickle'
		self.maxHeight = 70
		self.maxWidth = 60
		self.fullImageHeight = 420
		self.fullImageWidth = 580
		self.imgDir = './input/nervesTest/'
		self.trainDir = './input/train/'
		self.imgNervesContoursDir = './input/generatedImages/'
		self.generatedImages = './input/generatedMoreImagesTest/'
		self.imgFalseDir = './input/notNervesTest/'

class Nerves():
	def __init__(self, Config):
		self.config = Config
		
	def extractNerves(self):
		self.getElipseData()
		i= 0
		if not os.path.exists(self.config.imgDir):
			os.makedirs(self.config.imgDir)
		
		for xy,centers,something in self.elipseList:
			xRandom = np.random.uniform(0, 15)
			yRandom = np.random.uniform(0, 15)
			# TODO change here to add or subtract random value to get more images
			cropXFrom = xy[0] + xRandom - (self.config.maxWidth/2)
			cropXTo = xy[0] + xRandom + (self.config.maxWidth/2)
			cropYFrom = xy[1] - yRandom - (self.config.maxHeight/2)
			cropYTo = xy[1] - yRandom + (self.config.maxHeight/2)
			
			imageName = self.getImageNameFromMaskPath(self.filesList[i])
			image = cv2.imread(self.config.trainDir + imageName)
			if(image is not None):
				crop_img = image[cropYFrom:cropYTo, cropXFrom:cropXTo]
				cv2.imwrite(self.config.imgDir + 'a_'+ str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + imageName, crop_img)
			else:
				print('Image is not found %s' % self.config.trainDir + imageName)
			i = i+1
		
	def extractNotNerves(self):
		files = self.getAllMaskFiles()
		i= 0
		if not os.path.exists(self.config.imgFalseDir):
			os.makedirs(self.config.imgFalseDir)
		
		for oneFile in files:
			image = cv2.imread(oneFile, -1)
			empty = self.checkIfMasIsEmpty(image)
			if(empty == True):
				cropXFrom = randint(0,self.config.fullImageWidth-self.config.maxWidth)
				cropYFrom = randint(0,self.config.fullImageHeight-self.config.maxHeight)
				cropXTo = cropXFrom + self.config.maxWidth
				cropYTo = cropYFrom + self.config.maxHeight
				imageName = self.getImageNameFromMaskPath(oneFile)
				image = cv2.imread(self.config.trainDir + imageName)
				crop_img = image[cropYFrom:cropYTo, cropXFrom:cropXTo]
				cv2.imwrite(self.config.imgFalseDir + 'a_'+ str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) +imageName, crop_img)
				#if(i > 109):#2320
				#	return True
				i += 1
	
	def generateMoreNervesContoursImages(self, batchSize):

		# load data
		files = glob.glob(self.config.imgNervesContoursDir + '*.tif')
		X_train = np.ndarray(shape=(len(files), 1, self.config.maxHeight, self.config.maxWidth),
							dtype=np.float32)
		# TODO 200 is hardcoded number
		X_batch = np.ndarray(shape=(200 + 1, 1, self.config.maxHeight, self.config.maxWidth),
							dtype=np.float32)
		y_train = np.ndarray(shape=(len(files)),dtype=np.int32)
		#read data
		num_images = 0
		for image_file in files:
			try:
				image = cv2.imread(image_file, 0)
				if(image is not None):
						image = cv2.resize(image,(self.config.maxWidth, self.config.maxHeight), interpolation = cv2.INTER_CUBIC)
						X_train[num_images, ::] = image
						y_train[num_images] = 1
						num_images = num_images + 1
				else:
					print('Could not read: ' + image_file,'image_file')
			except IOError as e:
				print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
		if(num_images == 0):
			print('No contour images was found in directory %s ' % self.config.imgNervesContoursDir)
			return False
		X_train = X_train.astype('float32')
		# define data preparation
		shift = 0.2
		datagen = ImageDataGenerator(featurewise_center=False, featurewise_std_normalization=False, width_shift_range=shift, height_shift_range=shift)
		#datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
		# fit parameters from data
		X_train = X_train.astype('float32')
		datagen.fit(X_train)
		generatedIm = datagen.flow(X_train, y_train, batch_size=batchSize)
		X_batch,y_batch = generatedIm.next()
		X_batch = X_batch.reshape(batchSize, self.config.maxHeight* self.config.maxWidth)
		shuffledData = np.c_[X_batch , y_train]
		np.random.shuffle(shuffledData)
		dataset = shuffledData[:,0 : self.config.maxHeight * self.config.maxWidth]
		labels = shuffledData[:,self.config.maxHeight * self.config.maxWidth ::]
		datasetLenght = dataset.shape[0]		

		try:
			f = open(self.config.pickleName, 'wb')
			save = {
				'train_dataset': dataset,
				'train_labels' : labels,
			}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
			f.close()
		except Exception as e:
			print('Unable to save data to elis:', e)
			raise
		i = 0
		if not os.path.exists(self.config.generatedImages):
			os.makedirs(self.config.generatedImages)
		
		for generatedImages in dataset:
			oneImage = generatedImages.reshape(self.config.maxHeight, self.config.maxWidth)
			cv2.imwrite(self.config.generatedImages +'a_'+ str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + str(i) + '.png', oneImage)	
			i += 1
 		
	def getElipseData(self):		
		with open(self.config.pickle_file, 'rb') as f:
			save = pickle.load(f)
			self.elipseList = save['elipseList']
			self.filesList = save['filesList']
			del save
			
	def extractFileName(self, path):
		imagebase = os.path.basename(path)
		return imagebase

	def getImageNameFromMaskPath(self, path):
		base = self.extractFileName(path)
		imageName = base[:-9] + '.tif'
		return imageName
				
	def getAllMaskFiles(self):
		return glob.glob(self.config.trainDir + "*_mask.tif")

	def checkIfMasIsEmpty(self, mask):
		return np.sum(mask[:,:]) == 0
		
configModel = Config()
dataModel = Nerves(configModel)

#hardcoded size
batchSize = 14
'''
# TODO uncomment if you want to generate shifted your nerves images
dataModel.generateMoreNervesContoursImages(batchSize)
# TODO uncomment if you want to crop nerves images (need elipse information)
dataModel.extractNerves()
# TODO uncomment if you want to generate random not nerves images
#dataModel.extractNotNerves()
'''