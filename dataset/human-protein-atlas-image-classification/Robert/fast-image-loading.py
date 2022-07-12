'''
This script is used to create composite images that contain all
four channels and save them to the hard drive.  After that loading
an image only requires loading one composite instead of four
separate channels.
'''

import numpy as np
import cv2
import os
import glob

trainDirPath = '../input/train'
testDirPath = '../input/test'


channelColors = ['red','green','blue','yellow']

def readChannels(root_dir,imgid):
  channels = []
  for color in channelColors:
    imagePath = root_dir + '/' + imgid + '_' + color + '.png'
    chan = cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
    channels.append(chan)
  channels = np.array(channels) 
  return channels

def getImageIds(root_dir):
  imageFilepaths = glob.glob(root_dir + '/*.png')
  imgids = []
  for fp in imageFilepaths:
    d,f = os.path.split(fp)
    name,ext = os.path.splitext(f)
    fid,color = name.split('_')
    imgids.append(fid)
  imgids = list(set(imgids))
  return imgids

def makeImagePath(root_dir,imgid):
  path = root_dir + '/' + imgid + '.npy' 
  return path

def makeComposites(root_dir,force=False):
  imgids = getImageIds(root_dir)
  for imgid in imgids:
    imgPath = makeImagePath(root_dir,imgid)
    if force or not os.path.exists(imgPath):
      channels = readChannels(root_dir,imgid)
      np.save(imgPath,channels,allow_pickle=True)

def readComposite(root_dir,imgid):
  imgPath = makeImagePath(root_dir,imgid)
  channels = np.load(imgPath,allow_pickle=True)
  return channels

###############################
#These calls create the composite images that contain all 4 channels.
#Composites are saved on the hard drive so this only takes place once.
#The shape of each composite will be (4,512,512)

#makeComposites(trainDirPath)
#makeComposites(testDirPath)
###############################

##################################
# Get a list of all the image ids available in a directory
#imgIds = getImageIds(trainDirPath)

# Load a composite using the path to the directory and the image id
#img = readComposite(trainDirPath,imgIds[0])
############################

  
  
  
  
  
  
  
  
  