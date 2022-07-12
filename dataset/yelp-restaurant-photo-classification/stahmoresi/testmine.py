import time; start_time = time.time()
import warnings; warnings.filterwarnings('ignore');
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from PIL import Image
from PIL import ImageFilter
train_photos = pd.read_csv('../input/train_photo_to_biz_ids.csv')
train_attr = pd.read_csv('../input/train.csv')
train_id = pd.read_csv('../input/train_photo_to_biz_ids.csv') 
test_photos = pd.read_csv('../input/test_photo_to_biz.csv')
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
print("Train...")
for x in range(25):
        plt.subplot(5, 5, x+1)
        im = Image.open('../input/train_photos/' + str(train_photos.photo_id[x]) + '.jpg')
        im = im.resize((100, 100), Image.ANTIALIAS)
        plt.imshow(im)
        plt.axis('off')
print("Test...")
plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.subplots_adjust(wspace=0, hspace=0)
for x in range(25):
        plt.subplot(5, 5, x+1)
        im = Image.open('../input/test_photos/' + str(test_photos.photo_id[x]) + '.jpg')
        im = im.resize((100, 100), Image.ANTIALIAS)
        plt.imshow(im)
        plt.axis('off')
print("Train Photos", len(train_photos), len(train_photos.columns))
train_photos.head()
print("Train Attributes", len(train_attr), len(train_attr.columns))
train_attr.head()
print("Train ID", len(train_id), len(train_id.columns))
train_id.head()
print("Test Photos", len(test_photos), len(test_photos.columns))
test_photos.head()
label_notation = {0: 'good_for_lunch', 1: 'good_for_dinner', 2: 'takes_reservations',  3: 'outdoor_seating',
                  4: 'restaurant_is_expensive', 5: 'has_alcohol', 6: 'has_table_service', 7: 'ambience_is_classy',
                  8: 'good_for_kids'}
for l in label_notation:
    ids = train_attr[train_attr['labels'].str.contains(str(l))==True].business_id.tolist()[:9]
    plt.rcParams['figure.figsize'] = (7.0, 7.0)
    plt.subplots_adjust(wspace=0, hspace=0)
    for x in range(9):
        plt.subplot(3, 3, x+1)
        im = Image.open('../input/train_photos/' + str(train_photos.photo_id[ids[x]]) + '.jpg')
        im = im.resize((150, 150), Image.ANTIALIAS)
        plt.imshow(im)
        plt.axis('off')
    fig = plt.figure()
    fig.suptitle(label_notation[l])