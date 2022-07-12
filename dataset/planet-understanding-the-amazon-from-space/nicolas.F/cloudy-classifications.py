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
    
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0],:])

if __name__ == "__main__":
    print ("hello")
    id_cloudy, id_partly, id_haze, id_clear = find_id()
    H_mean = []
    H_std = []
    S_mean = []
    S_std = []
    V_mean = []
    V_std = []
    LBP1  = []
    LBP2 = []
    P = 8
    R = 4
  
    
    for i in id_cloudy[0:100]:
        #open image and convert to RGB
        im = cv2.imread('../input/train-jpg/train_'+str(i)+'.jpg')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        #sliding image to get all fenetre
        winW = 20 #size of fenetre
        winH = 20 
        fenetre = []
        for (x, y, window) in sliding_window(im, stepSize=20, windowSize=(winW, winH)):
    	#if the window does not meet our desired window size, ignore it
    	    if window.shape[0] == winH and window.shape[1] == winW:
		        fenetre.append(window)
		#compute features
        L1 = len(fenetre)
        for i in range(0,L1):
            window = cv2.cvtColor(fenetre[i], cv2.COLOR_BGR2HSV)
            H_mean.append (np.mean(window[:,:,0]))
            S_mean.append (np.mean(window[:,:,1]))
            V_mean.append (np.mean(window[:,:,2]))
            H_std.append (np.std(window[:,:,0]))
            S_std.append (np.std(window[:,:,1]))
            V_std.append (np.std(window[:,:,2]))
            
            
        #for i in range(0,L):
            #window = cv2.cvtColor(fenetre[i], cv2.COLOR_BGR2GRAY)
            #lbp = local_binary_pattern(window,P,R)
            #counts, bins, bar = plt.hist(lbp.ravel(), bins=256, range=(0., 255))
            #L = len(counts)
            #a = L-1
            #b = L-50
            #LBP1.append(sum(counts[0:50]))
            #LBP2.append(sum(counts[b:a]))
            
    print ("cloudy to clear")
    
    for i in id_clear[0:100]:
        #open image and convert to RGB
        im = cv2.imread('../input/train-jpg/train_'+str(i)+'.jpg')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        #sliding image to get all fenetre
        winW = 20 #size of fenetre
        winH = 20 
        fenetre = []
        for (x, y, window) in sliding_window(im, stepSize=20, windowSize=(winW, winH)):
    	#if the window does not meet our desired window size, ignore it
    	    if window.shape[0] == winH and window.shape[1] == winW:
		        fenetre.append(window)
		#compute features
        L2 = len(fenetre)
        for i in range(0,L2):
            window = cv2.cvtColor(fenetre[i], cv2.COLOR_BGR2HSV)
            H_mean.append (np.mean(window[:,:,0]))
            S_mean.append (np.mean(window[:,:,1]))
            V_mean.append (np.mean(window[:,:,2]))
            H_std.append (np.std(window[:,:,0]))
            S_std.append (np.std(window[:,:,1]))
            V_std.append (np.std(window[:,:,2]))
        
            
    
    end = len(H_mean)
    X_features = np.zeros((6,end))

    for i in range (0,end):
        X_features[0,i] = H_mean[i]
        
    for i in range (0,end):
        X_features[1,i] = H_std[i]
        
    for i in range (0,end):
        X_features[2,i] = V_mean[i]
        
    for i in range (0,end):
        X_features[3,i] = V_std[i]
        
    for i in range (0,end):
        X_features[4,i] = S_mean[i]
        
    for i in range (0,end):
        X_features[5,i] = S_std[i]
    
    X_features = np.transpose(X_features)
    
    y = []
    for i in range(0,14400):
        y.append (1)
    for i in range(0,14400):
        y.append (0)
    y = np.array(y)
    y = np.transpose(y)
    print(y.shape)
    print(X_features.shape)
    
    X_features, y = shuffle(X_features, y, random_state=0)  # shuffle with no random
    clf = svm.SVC(kernel='poly', C=1.0, class_weight='balanced')
    X_train, X_test, y_train, y_test = train_test_split( X_features, y, test_size=0.33)
    clf = clf.fit(X_train,y_train)
    pred = clf.predict(X_test)
    acc = accuracy_score(y_test,pred)
    print (acc)
       
        
    
        
        
    print ("end")
    
