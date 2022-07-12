# %% [code]


# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time, gc
import tensorflow as tf
from PIL import Image
print(tf.__version__)

from sklearn.preprocessing import LabelBinarizer

import random

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Any results you write to the current directory are saved as output.

# %% [markdown]
# ## Create Label Binarizer for the three Classes

# %% [code]
class_map_df = pd.read_csv('/kaggle/input/bengaliai-cv19/class_map.csv')
class_map_df_root = class_map_df[class_map_df.component_type=='grapheme_root']
class_map_df_vowel = class_map_df[class_map_df.component_type=='vowel_diacritic']
class_map_df_cons = class_map_df[class_map_df.component_type=='consonant_diacritic']
graphemeLB = LabelBinarizer()
vowelLB = LabelBinarizer()
consonantLB = LabelBinarizer()

graphemeLB.fit(class_map_df_root.label)
vowelLB.fit(class_map_df_vowel.label)
consonantLB.fit(class_map_df_cons.label)

# %% [code]
print(len(vowelLB.classes_))
print(len(consonantLB.classes_))
print(len(graphemeLB.classes_))

# %% [markdown]
# ## Load saved model, pre-process Test data and make predictions on Test Set

# %% [code]
BS=64
model = tf.keras.models.load_model('/kaggle/input/bengali-graphemes-multichannelcnn-using-tf2/Bengali_model_Tf2.h5')
                                   #/kaggle/input/bengali-graphemes-multichannelcnn-data-aug-2/Bengali_model_AugDSn2.h5')

row_id=[]
target=[]
for i in range(4):
    print("[INFO] Now reading parquet file test images")
    test_df = pd.read_parquet('/kaggle/input/bengaliai-cv19/test_image_data_{}.parquet'.format(i)) 
    #test_df=test_df[0:100]
    #print(test_df.head())
    print("iteration:"+str(i))

    testX=np.array(test_df.iloc[:,1:]).reshape(-1,137,236,1)
    testX=testX.astype('uint8')
    
    print("[INFO] creating test dataset from test images")
    print(testX.shape)
    resized_image=[]
    for k in range(len(testX)):
        resized_img = tf.image.resize(testX[k],[96,96])
        resized_img=np.array(resized_img)/255.
        resized_image.append(resized_img)
        
    test_set=tf.data.Dataset.from_tensor_slices(resized_image).batch(BS).prefetch(None)
    del resized_image
    del testX
    gc.collect()
    
    print("[INFO] Now predicting probability on Test set")
    (graphemeProba, vowelProba, consonantProba) = model.predict(test_set)

    del test_set
    gc.collect()
 
    print("[INFO] Now creating labels from predicted probabilities")
    
    for j in range (len(graphemeProba)):
        consonantIdx = consonantProba[j].argmax()
        consonantLabel = consonantLB.classes_[consonantIdx]
        row_id.append(test_df['image_id'][j]+"_consonant_diacritic")
        target.append(consonantLabel)
        graphemeIdx = graphemeProba[j].argmax()
        graphemeLabel = graphemeLB.classes_[graphemeIdx]
        row_id.append(test_df['image_id'][j]+"_grapheme_root")
        target.append(graphemeLabel)
        vowelIdx = vowelProba[j].argmax()
        vowelLabel = vowelLB.classes_[vowelIdx]
        row_id.append(test_df['image_id'][j]+"_vowel_diacritic")
        target.append(vowelLabel)
           
    del test_df
    gc.collect()

# %% [code]
print("[INFO] Now creating Submission data")
sub_df = pd.DataFrame()
sub_df["row_id"]=row_id
sub_df["target"]=target

# %% [code]
print(sub_df.head())
print(sub_df.tail())

# %% [code]
print("[INFO] now writing submission csv")
sub_df.to_csv("submission.csv",index=False)