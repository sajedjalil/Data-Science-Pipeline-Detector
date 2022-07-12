import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import cv2
from subprocess import check_output
#from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pprint
from numpy import genfromtxt, savetxt
import csv
import time
start_time = round(time.time())
tag = str(start_time)

def img_as_array(image_id, test_indicator, new_img_size):
    # returns a 1-d numpy array
    if not test_indicator:
        imagepath = '../input/train_photos/'+str(image_id)+'.jpg'
    else:
        imagepath = '../input/test_photos/'+str(image_id)+'.jpg'
    img = cv2.imread(imagepath)
    resized_image = cv2.resize(img, new_img_size)
    #print(resized_image.size)
    resized_image = resized_image.reshape(3*new_img_size[0]*new_img_size[1])
    more_features = other_features(resized_image,new_img_size)
    avg = averages(resized_image,new_img_size)
    tmp = np.append(resized_image,more_features)
    return np.append(tmp,avg)

def averages(arr,img_size):
    img = np.reshape(arr,(img_size[0],img_size[1],3))
    uAvg = np.average(img,axis=0).tolist()
    vAvg = np.average(img,axis=1).tolist()
    wAvg = np.average(img,axis=2).tolist()
    #print(len(uAvg),len(vAvg),len(wAvg))
    out = []
    for x in uAvg+vAvg+wAvg:
        out += x
    return np.array(out)
        
def other_features(arr,img_size):
    img = np.reshape(arr,(img_size[0],img_size[1],3))
    u = np.sum(img,axis=0).tolist()
    v = np.sum(img,axis=1).tolist()
    w = np.sum(img,axis=2).tolist()
    out = []
    for x in u+v+w:
        out += x
    '''
    print(arr.tolist())
    print(img.tolist())
    print(u.tolist())
    print(v.tolist())
    print(w.tolist())
    print('-------')
    '''
    return np.array(out)

#print(check_output(["ls", "../input"]).decode("utf8"))
#train_images = check_output(["ls", "../input/train_photos"]).decode("utf8")
#print(train_images[:100])
#print('time elapsed:'+str((time.time() - start_time)/60))
print('Reading data...')
train_photos = pd.read_csv('../input/train_photo_to_biz_ids.csv')
train_photos.sort_values(['business_id'],inplace=True)
train_photos.set_index(['business_id'])

test_photos = pd.read_csv('../input/test_photo_to_biz.csv')
test_photos.sort_values(['business_id'],inplace=True)
test_photos.set_index(['business_id'])

train = pd.read_csv("../input/train.csv")
train.sort_values(['business_id'],inplace=True)
train.reset_index(drop=True)
print('number of training examples:',train.shape[0])
print('number of test examples:',len(set(test_photos['business_id'])))
print('Finished reading data...')
#print('time elapsed:'+str((time.time() - start_time)/60))

#test_photos = pd.read_csv("../input/test_photo_to_biz.csv")
print('Reading/modifying images...')
img_size  = (20,20)
imgs_per_loc = 1 # we only use one image for each restaurant
arr_size = 3*img_size[0]*img_size[1]
#column_names = ['A_'+str(i) for i in range(arr_size)] #+\
               #['B_'+str(i) for i in range(arr_size)] +\
               #['C_'+str(i) for i in range(arr_size)]

i = 0
max_images = 1200
imgs_per_loc = 1
pLoc = -1
count = 0
X = []
Xtest = []
arr = []
Y = []
Ypred = []
locs = []
print('\tpreparing train data...')
for row in train_photos.itertuples():
    image_id = row[1]
    loc = row[2]
    #print(row)
    #print(loc,pLoc)
    if loc == pLoc:
        count += 1
        if count < imgs_per_loc: #never satisfied
            arr += list(img_as_array(image_id,False,img_size))    
            #print(row(1))
            i += 1
        else:
            continue
    else:
        if arr:
            locs.append(loc)
            X.append([int(x) for x in arr])
            #print(loc,type(loc))
            y_vals = train[train['business_id'] == loc]
            y = [0]*9
            for r in y_vals.itertuples():
                try:
                    for u in [int(x) for x in r[2].split(' ')]:
                        y[u] = 1
                except:
                    print(r)
            Y.append(y)
            #print(len(X),len(Y))
        #print(len(arr))
        #print(arr)
        pLoc = loc
        arr = list(img_as_array(image_id,False,img_size))    
        count = 1
        i += 1
    if i>max_images:
        break
print('\tpreparing test data...')
test_ids = []
for row in test_photos.itertuples():
    image_id = row[1]
    loc = row[2]
    #print(row)
    #print(loc,pLoc)
    if loc == pLoc:
        count += 1
        if count < imgs_per_loc: #never satisfied
            arr += list(img_as_array(image_id,True,img_size))    
            #print(row(1))
            i += 1
        else:
            continue
    else:
        if arr:
            #locs.append(loc)
            Xtest.append([int(x) for x in arr])
            test_ids.append(loc)
        pLoc = loc
        arr = list(img_as_array(image_id,True,img_size))    
        count = 1
        i += 1
    if i>max_images:
        continue
        #break

print("converting data...")
X = np.array(X)
Y = np.array(Y)
Xtest = np.array(Xtest)
#print(X.shape)
#print(Y.shape)
#print(Xtest.shape)
#print("training...")
#print('time elapsed:'+str((time.time() - start_time)/60))
num_classes = len(Y[1,:])
clf = [None]*num_classes
for i in range(num_classes):
    print("creating classifier:",i)
    #clf[i] = MLPClassifier(algorithm='l-bfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
    rf = RandomForestClassifier(n_estimators=300,max_depth=2*img_size[0],n_jobs=-1,oob_score=True,verbose=2,criterion="entropy")
    print("fitting classifier:",i)
    rf.fit(X, Y[:,i])
    #clf[i].fit(X, Y[:,i])
    print("getting predictions for attribute:",i)
    y_pred = rf.predict(Xtest)
    Ypred.append(y_pred)
    #print(y_pred)
print("preparing output...")
Ypred = np.vstack(Ypred)
#print(Ypred.shape)
Ypred = np.transpose(Ypred)
#print(Ypred.shape)
Ypred = Ypred.tolist() #why not? it's only your own time you're wasting
with open("predictions_"+tag+".csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['business_id','labels'])
    for i,r in enumerate(zip(test_ids,Ypred)):
        output = ' '.join([str(j) if x > 0 else '' for j,x in enumerate(r[1])]).strip()
        line = [r[0],output]
        #print(line)
        #print(r)
        writer.writerow(line)
        #if i > 100:
        #    break

