import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import random
import cv2
import os, glob

train_files = [f for f in glob.glob("../input/train_2/*")]
i_ = 0
plt.rcParams['figure.figsize'] = (32.0, 32.0)
plt.subplots_adjust(wspace=0, hspace=0)
for l in train_files[:100]:
    im = cv2.imread(l)
    im = cv2.resize(im, (32, 32)) 
    plt.subplot(10, 10, i_+1) #.set_title(l)
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 1