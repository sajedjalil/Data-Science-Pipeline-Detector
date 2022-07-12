## This is my Kernel for Kuzushiji ##

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2

def data_viewer(num):
    print (data_list_semi_training_set[num][5])
    img_id = data_list_semi_training_set[num][5]
    img = cv2.imread('../input/train_images/'+str(img_id)+'.jpg')

    x = int(data_list_semi_training_set[num][1])
    y = int(data_list_semi_training_set[num][2])
    w = int(data_list_semi_training_set[num][3])
    h = int(data_list_semi_training_set[num][4])
    print(x,y,w,h)
    img_cropped = img[y:y+h, x:x+w]
    plt.imshow(img_cropped)
    plt.show()
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2)).astype("uint8")
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    plt.imshow(thresh)
    print(thresh)
    reshapeX = thresh.ravel()
    print(reshapeX)


print(os.listdir("../input"))

## In the description there is bounding box information on the train.csv file
## first we try to visualize those bounding box  on the image
#Code below is to access the training set
DATA = pd.read_csv('../input/train.csv')
#image ID
img_id = DATA.loc[120, 'image_id']

#labels
print(DATA.loc[120, 'labels'])
#now let's split this data and read the bounding box
split_labels = DATA.loc[120, 'labels'].split()


# Below code is to access the images
#print(os.listdir("../input/train_images/"))
print(img_id)
print(split_labels[2])
img = cv2.imread('../input/train_images/'+str(img_id)+'.jpg')
print(split_labels[5])
x = int(split_labels[6])
y = int(split_labels[7])
w = int(split_labels[8])
h = int(split_labels[9])
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 7)
print(split_labels[5+15])
x = int(split_labels[6+15])
y = int(split_labels[7+15])
w = int(split_labels[8+15])
h = int(split_labels[9+15])
cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 7)
plt.imshow(img)
plt.show()



## okay, new findings, there 575 character that dont have training set!!!!...
## What to do next?...
## let's create some stuff that view the character picture than print number of training set
## or maybe lets just straight to creating Matrix_X
## First step to create Matrix_X is parse each of the train.csv into forms like [Y_Character][img_id][bounding_rect]

i =0
j= 0
row=[]
#print(DATA.loc[126,'labels'])
## the structure of data_list_semi_training_set[] is (VectorY, x, y, w, h, img_id)
## this would be the embrio for the training set (VectorY->char_encoding, MatrixX->image_matrix(0,1)
data_list_semi_training_set = []
iterable = range(0, len(DATA))
for iter in iterable:
    ## Entire loop below are just for one row in the train.csv
    procesed_row_for_matrix_X= DATA.loc[iter,'labels']
    #print(procesed_row_for_matrix_X)
    if(len(str(DATA.loc[iter, 'labels']))>3):
        arr_procesed_row_for_matrix_X = procesed_row_for_matrix_X.split()
        img_id = DATA.loc[iter,'image_id'] # img_id of each row
        for data in arr_procesed_row_for_matrix_X:
            row.append(data)
            i = i+1
            if(i==5):
                row.append(img_id)
                data_list_semi_training_set.append(row)
                i=0
                j=j+1
                row = []
                print (j)
                print ("The Data : " + str(row))
    print ("Iteration : " + str(iter)+" filename :" + str(img_id))
    print("Data Length :" + str(len(data_list_semi_training_set)))

print (data_list_semi_training_set[6][5])
img_id = data_list_semi_training_set[6][5]
img = cv2.imread('../input/train_images/'+str(img_id)+'.jpg')

x = int(data_list_semi_training_set[6][1])
y = int(data_list_semi_training_set[6][2])
w = int(data_list_semi_training_set[6][3])
h = int(data_list_semi_training_set[6][4])
print(x,y,w,h)
img_cropped = img[y:y+h, x:x+w]
plt.imshow(img_cropped)
plt.show()


