# Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def find_id():
    df = pd.read_csv("../input/train_v2.csv")
    tags = df["tags"].apply(lambda x: x.split(' '))
    end = len(tags)
    id_haze = []
    id_cloudy = []
    id_partly = []
    id_clear = []
    LBP1 = []
    LBP2 = []
    for i in range (0,end):
        for x in tags[i]:
            if x == 'haze':
                id_haze.append(i)
            elif x == 'cloudy':
                id_cloudy.append(i)
            elif x == 'partly_cloudy':
                id_partly.append(i)
            elif x == 'clear':
                id_clear.append(i)
                
    return id_cloudy, id_partly, id_haze, id_clear
    
id_cloudy,_,_,_ = find_id()
for i in range (0, 10):
    print(id_cloudy[i])








