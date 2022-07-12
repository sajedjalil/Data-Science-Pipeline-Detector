# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import scipy
import csv as csv
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
from numpy import array
import cv2
import os
import skimage as ski
import random
from PIL import Image 
from PIL import ImageFilter
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import re

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

#get full dataset
TRAIN_DIR = '../input/train/'
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for training images


TEST_DIR = '../input/test/'
test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)] # use this for test images


#get labels
predictedlabels=[]
labels=[]
train_dogs=[]
images=[]


#def baseline():         #flips a coin for each image
 #   predictedlabels=[]
  #  for i in os.listdir(TRAIN_DIR):
   #     test = random.randint(1,10)
    #    if test>5:
     #       predictedlabels.append(1)
      #  else:
       #     predictedlabels.append(0)
   
    
def train():            #gets correct class for each image

    for i in os.listdir(TRAIN_DIR):
        if 'dog' in i:
            train_dogs.append(i)
            labels.append(1)
        else:
            labels.append(0)
            
    return labels
    
    
    
def getResults(predictedlabels, labels):    #outputs accuracy

    total=0
    newpredict=[]
    for r in range(0,len(labels)):
   
        #if predictedlabels[r] == labels[r]:
        if float(predictedlabels[r])>0.5:
            newpredict.append(1)
        else:
            newpredict.append(0)
            
       
        if newpredict[r] == labels[r]:
            
            
            
            
            total+=1
         
    print("Accuracy:",total,"/",len(labels),"* 100 =","{0:.3f}".format(total/len(labels)*100),"%")
  


def svm(y):             #trains svm on first 2/3 of training set

    results=[]
    pix_val=0
    new_images=[]
    placeholder=[]
    #x = np.zeros((25000,48705))
    #clf = LinearSVC(loss='l2', penalty='l1', dual=False)
   
    #clf = sk.linear_model.LogisticRegression
    svm = LinearSVC()
    clf = LogisticRegression()
   # clf = CalibratedClassifierCV(svm, method='sigmoid') 
   # for i in train_dogs[0:1]:       
    #for i in train_images[0:24000]: #train on first 12500 images
    for i in train_images:
        pil_im = Image.open(i).convert('L')
        #pil_im = Image.open(i)
        size=64,64
      
        pil_im = pil_im.resize(size, Image.ANTIALIAS)
        #pil_im =pil_im.filter(ImageFilter.FIND_EDGES)
        pil_im=pil_im.filter(ImageFilter.GaussianBlur(255))
        pix_val=pil_im.histogram() 
        
      
        
        results.append(pix_val)
    
    
    #x should be an array (n_samples, n_features)
    
   # clf = clf.fit(results,y[0:24000])
    clf = clf.fit(results,y)
    return clf
    

def getTests(clf):          #test on last 1/3 of training set
    results=[]
    total=0
    myfile = open('results.csv', 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE,quotechar='',escapechar='\\')
    wr.writerow(["id","label"])
   # for i in test_images:  
    for i in range(0,len(test_images)):
            j = train_images[i]
            #img = Image.open(j)
            pil_im = Image.open(j).convert('L')
            #pil_im = Image.open(j)    # for pil 
       
            size=64,64
           
            pil_im = pil_im.resize(size, Image.ANTIALIAS)
       #pil_im =pil_im.filter(ImageFilter.FIND_EDGES)
            pil_im=pil_im.filter(ImageFilter.GaussianBlur(255))
            pix_val=pil_im.histogram()     
           
        #results.append(clf.predict([pix_val])) 
            x = str(clf.predict_proba([pix_val]))
            x=x[2:-2]
           
            x = re.split('\s+',x)
           # print(x)
            results.append(x[2])  
           
            wr.writerow([i+1,float(x[2])])
            #wr.writerow(["wow"])
    #print("LEn",len(results))
    flattened = [val for sublist in results for val in sublist]  
    print("LEn",len(results))
    
    
    
    
    
    #return flattened
    return results


y=train()   #get correct labels

clf=svm(y)   #get trained svm 

imageinfo = getTests(clf)       #get predictions for testing
#print("Guess",imageinfo[0:50])
#print("Result",y[16000:16050])

#getResults(imageinfo,y[16000:])     #output accuracy of predictions


#for i in range(0,len(imageinfo)):
#    print(i+1,",",str(imageinfo[i][-12:-2])) 



#print(len(imageinfo))
#print(len(y[16000:]))
#getResults(imageinfo,y[24000:]) 
