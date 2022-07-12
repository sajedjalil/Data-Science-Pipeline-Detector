import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import io, bson
import matplotlib.pyplot as plt
from skimage.data import imread   # or, whatever image library you prefer

from sklearn import preprocessing


# Any results you write to the current directory are saved as output.

# Simple data processing
data = bson.decode_file_iter(open('../input/train.bson', 'rb'))
# read bson file into pandas DataFrame
with open('../input/train.bson','rb') as b:
    df = pd.DataFrame(bson.decode_all(b.read()))

#Get shape of first image
for e, pic in enumerate(df['imgs'][0]):
        picture = imread(io.BytesIO(pic['picture']))
        pix_x,pix_y,rgb = picture.shape

n = len(df.index) #cols of data in train set
X_ids = np.zeros((n,1)).astype(int)
Y = np.zeros((n,1)).astype(int) #category_id for each row
X_images = np.zeros((n,pix_x,pix_y,rgb)) #m images are 180 by 180 by 3

print("Examples:", n)
print("Dimensions of Y: ",Y.shape)
print("Dimensions of X_images: ",X_images.shape)

# prod_to_category = dict()
i = 0
for c, d in enumerate(data):
    X_ids[i] = d['_id']
    Y[i] = d['category_id']
    for e, pic in enumerate(d['imgs']):
        picture = imread(io.BytesIO(pic['picture']))
    X_images[i] = picture #add only the last image
    i+=1

#Lets take a look at the category names supplied to us:
df_categories = pd.read_csv('../input/category_names.csv', index_col='category_id')

count_unique_cats = len(df_categories.index)

print("There are ", count_unique_cats, " unique categories to predict. E.g.")
print("")
print(df_categories.head())

le=preprocessing.LabelEncoder()
le.fit(Y)
Y_=le.transform(Y)
print(len(le.classes_))

import tensorflow as tf
fc=[tf.feature_column.numeric_column("x",shape=[97200])]
classifier=tf.estimator.DNNClassifier(feature_columns=fc,hidden_units=[1024,512,1024],n_classes=36)
train_input_fn=tf.estimator.inputs.numpy_input_fn(x={"x":np.array(X_images)},y=np.array(Y_),shuffle=True)

 
classifier.train(input_fn=train_input_fn,steps=10000)


accuracy_score=classifier.evaluate(input_fn=train_input_fn)["accuracy"]
print("\n Test Accuracy: {0:f} \n" .format(accuracy_score))