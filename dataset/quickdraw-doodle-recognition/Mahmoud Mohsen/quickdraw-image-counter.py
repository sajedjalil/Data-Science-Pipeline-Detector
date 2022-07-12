import os
import json
import numpy
import pandas
import time

from datetime import datetime
from PIL import Image
from PIL import ImageDraw

TRAIN_FILES_PATH = '../input/train_simplified/'

trainingFileNameArr = os.listdir(TRAIN_FILES_PATH)
totalClassesCount = len(trainingFileNameArr)
totalImagesCount = 0

print('classes:', totalClassesCount)

for trainingFileName in trainingFileNameArr:
  print(trainingFileName)
  dataset = pandas.read_csv(TRAIN_FILES_PATH + trainingFileName, header=0).values
  totalImagesCount += dataset.shape[0]

print('images:', totalImagesCount)
print('')
print('')
