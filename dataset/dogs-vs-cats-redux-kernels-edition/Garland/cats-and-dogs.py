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
import PIL
from PIL import ImageOps
from PIL import Image 
from PIL import ImageFilter
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import re
import tensorflow as tf

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
  


def svm(testresults):             #trains svm on first 2/3 of training set
    #you need to pass in y but call it testresults
    converted=[]
    train = []
    test = np.asarray(testresults)
    test = np.resize(testresults,(10,16384))
    
    for i in range(0,10):     #test data
        img = Image.open(train_images[10000+i]).convert('L')
        size=128,128
        img = img.resize(size, Image.ANTIALIAS)
        img = list(img.getdata())
        converted.append(img)
        
    converted = np.array(converted)
   
    
    sess = tf.InteractiveSession()
    
    x = tf.placeholder(tf.float32, shape=[None, 16384])
    y_ = tf.placeholder(tf.float32, shape=[None, 2]) #how many different classes
    W = tf.Variable(tf.zeros([16384,10])) #ten outputs
    b = tf.Variable(tf.zeros([2]))      #two classes
    sess.run(tf.global_variables_initializer())
    
    y = tf.matmul(x,W) + b
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
    
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(accuracy.eval(feed_dict={x: converted, y_: test}))
    
    '''
    for i in range(0,1000):     #test data
        img = Image.open(train_images[10000+i]).convert('L')
        size=128,128
        img = img.resize(size, Image.ANTIALIAS)
        img = list(img.getdata())
        converted.append(img)
        
    x = tf.placeholder(tf.float32, [None, 16384])
    W = tf.Variable(tf.zeros([16384, 16384]))
    b = tf.Variable(tf.zeros([16384]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)
  
    
    y_ = tf.placeholder(tf.float32, [None, 16384])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.global_variables_initializer()
    
    sess = tf.Session()
    sess.run(init)
   
    for i in range(5000):
        img = Image.open(train_images[1000+i]).convert('L')
        size=128,128
        img = img.resize(size, Image.ANTIALIAS)
        img = (list(img.getdata()))
        train.append(img)
       
    
    
    batch_xs = converted
    batch_ys = converted
    batch_xs = np.asarray(batch_xs)
   # print(batch_xs.shape)
    batch_ys = np.asarray(batch_ys)
   # print(batch_ys.shape)
    
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    print("Running")
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print("finished running")    
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    print("Correct prediction",correct_prediction)  
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy",accuracy)   
    print("Total accuracy",sess.run(accuracy, feed_dict={x: train, y_: test}))
    
    prediction=tf.argmax(y,1)
    print(prediction.eval(session=sess, feed_dict={x: converted}))
    '''
   # return clf
    

def getTests(clf):          #test on last 1/3 of training set
    results=[]
    total=0
    myfile = open('results.csv', 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE,quotechar='',escapechar='\\')
    wr.writerow(["id","label"])
    #for i in range(0,len(test_images)):  
    for i in range(0,len(train_images[24000:])):
            j = train_images[24000+i]
            #img = Image.open(j)
            #pil_im = Image.open(j).convert('L')
            pil_im = Image.open(j)    # for pil 
       
            size=128,128
           
            pil_im = pil_im.resize(size, Image.ANTIALIAS)
            
            pil_im =pil_im.filter(ImageFilter.FIND_EDGES)
            pil_im = pil_im.filter(ImageFilter.MaxFilter(size=5))
            
            
            pil_im = PIL.ImageOps.autocontrast(pil_im, cutoff=0, ignore=None)
            
            
            #pil_im=pil_im.filter(ImageFilter.GaussianBlur(255))
            pix_val=pil_im.histogram()     
           
            results.append(clf.predict([pix_val])) 
           
           
            
            
    #print("LEn",len(results))
    flattened = [val for sublist in results for val in sublist]  
    #print("LEn",len(results))
    return flattened
    
    
def kaggletest(svm):
    results=[]
    total=0
    myfile = open('results.csv', 'w')
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE,quotechar='',escapechar='\\')
    wr.writerow(["id","label"])
    #for i in range(0,len(test_images)):  
    for i in range(0,len(test_images)):
            j = test_images[i]
            #img = Image.open(j)
            #pil_im = Image.open(j).convert('L')
            pil_im = Image.open(j)    # for pil 
       
            size=128,128
           
            pil_im = pil_im.resize(size, Image.ANTIALIAS)
          
            pil_im =pil_im.filter(ImageFilter.FIND_EDGES)
            
            pil_im = pil_im.filter(ImageFilter.MaxFilter(size=5))
            
            pil_im = PIL.ImageOps.autocontrast(pil_im, cutoff=0, ignore=None)
           
            #pil_im=pil_im.filter(ImageFilter.GaussianBlur(255))
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

#imageinfo = getTests(clf)       #get predictions for testing
#imageinfo =  np.asarray(imageinfo, dtype=float)


#below is for training


#getResults(imageinfo,y[24000:])     #output accuracy of predictions
#print("Log Loss",sk.metrics.log_loss(y[24000:],imageinfo))


#below is for submitting
#kaggletest(clf)

#for i in range(0,len(imageinfo)):
#    print(i+1,",",str(imageinfo[i][-12:-2])) 



#print(len(imageinfo))
#print(len(y[16000:]))
#getResults(imageinfo,y[24000:]) 


