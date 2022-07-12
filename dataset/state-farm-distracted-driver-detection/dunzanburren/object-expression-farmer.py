from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
import cv2
import os, glob

drivers = pd.read_csv('../input/driver_imgs_list.csv')
train_files = [f for f in glob.glob("../input/train/*/*.jpg")]
test_files = ["../input/test/" + f for f in os.listdir("../input/test/")]
print(train_files[:10])
print(test_files[:10])
plt.rcParams['figure.figsize'] = (12.0, 12.0)
plt.subplots_adjust(wspace=0, hspace=0)
p1 = '/usr/local/share/OpenCV/haarcascades/'
p2 = '/usr/local/share/OpenCV/lbpcascades/'
c_files = []
c_files.append([p1 + 'haarcascade_eye.xml','eye'])
c_files.append([p1 + 'haarcascade_eye_tree_eyeglasses.xml','glasses'])
c_files.append([p1 + 'haarcascade_frontalcatface.xml','frontal'])
c_files.append([p1 + 'haarcascade_frontalcatface_extended.xml','cat ext'])
c_files.append([p1 + 'haarcascade_frontalface_alt.xml','alt'])
c_files.append([p1 + 'haarcascade_frontalface_alt2.xml','alt2'])
c_files.append([p1 + 'haarcascade_frontalface_alt_tree.xml','alt tree'])
c_files.append([p1 + 'haarcascade_frontalface_default.xml','default'])
c_files.append([p1 + 'haarcascade_fullbody.xml','body'])
c_files.append([p1 + 'haarcascade_lefteye_2splits.xml','splits'])
c_files.append([p1 + 'haarcascade_licence_plate_rus_16stages.xml','license'])
c_files.append([p1 + 'haarcascade_lowerbody.xml','lowerbody'])
c_files.append([p1 + 'haarcascade_profileface.xml','profile'])
c_files.append([p1 + 'haarcascade_righteye_2splits.xml','right eye'])
c_files.append([p1 + 'haarcascade_russian_plate_number.xml','russian'])
c_files.append([p1 + 'haarcascade_smile.xml','smile'])
c_files.append([p1 + 'haarcascade_upperbody.xml','upper'])
c_files.append([p2 + 'lbpcascade_frontalcatface.xml','cat frontal'])
c_files.append([p2 + 'lbpcascade_frontalface.xml','lbp frontal'])
c_files.append([p2 + 'lbpcascade_profileface.xml','lbp profile'])
c_files.append([p2 + 'lbpcascade_silverware.xml','silver'])

import random
fi = random.choice(train_files)
print(fi)
im = cv2.imread(fi)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
i_ = 0
plt.rcParams['figure.figsize'] = (11.0, 21.0)
plt.subplots_adjust(wspace=0, hspace=0)
for c in c_files:
    im2 = im.copy()
    gr_im = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    fc = cv2.CascadeClassifier(c[0])
    fr = fc.detectMultiScale(gr_im, scaleFactor=1.1, minNeighbors=2, minSize=(20, 20), flags = cv2.CASCADE_SCALE_IMAGE)
    if len(fr)>0:
        for (x, y, w, h) in fr:
            cv2.rectangle(im2, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    plt.subplot(7, 3, i_+1).set_title(c[1])
    plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 1
lbl = {'c0' : 'safe driving', 
'c1' : 'texting - right', 
'c2' : 'talking on the phone - right', 
'c3' : 'texting - left', 
'c4' : 'talking on the phone - left', 
'c5' : 'operating the radio', 
'c6' : 'drinking', 
'c7' : 'reaching behind', 
'c8' : 'hair and makeup', 
'c9' : 'talking to passenger'}

plt.rcParams['figure.figsize'] = (8.0, 20.0)
plt.subplots_adjust(wspace=0, hspace=0)
i_ = 0
for l in lbl:
    tf = ["../input/train/" + l + "/" + f for f in os.listdir("../input/train/" + l + "/")]
    fi = random.choice(tf)
    print(fi)
    im = cv2.imread(fi)
    plt.subplot(5, 2, i_+1).set_title(lbl[l])
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); plt.axis('off')
    i_ += 1
import time; start_time = time.time()
import warnings; warnings.filterwarnings('ignore');
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageFilter
from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.image import extract_patches_2d
#from sklearn.feature_extraction.image import img_to_graph
#from sklearn.metrics import f1_score
from sklearn import preprocessing
import multiprocessing
import random; random.seed(2016);
import cv2
import os
import re

train_drivers = pd.read_csv('../input/driver_imgs_list.csv')
train_drivers["path"] = "../input/train/" + train_drivers.classname + "/" + train_drivers.img
X_train = train_drivers[["path","img"]]
y_train = train_drivers['classname'].str.get_dummies()
id_test = os.listdir("../input/test/")
X_test = ["../input/test/" + f for f in id_test]
print("full set:",len(X_train), len(y_train), len(train_drivers), len(X_test), len(id_test))

#remove limit for outside kaggle run - every 1000th row to sample image categories
train_drivers = train_drivers.iloc[::1000, :]
train_drivers = train_drivers.reset_index(drop=True)
X_train = X_train.iloc[::1000, :]
X_train = X_train.reset_index(drop=True)
y_train = y_train.iloc[::1000, :]
y_train = y_train.reset_index(drop=True)
id_test = id_test[::1000]
X_test = X_test[::1000]
#end limit
print("limited:", len(X_train), len(y_train), len(train_drivers), len(X_test), len(id_test))

print("Start Feature Extraction: ", round(((time.time() - start_time)/60),2))

class cust_img_features(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, img_features):
        d_col_drops=['photo_id','tt','subject','classname','path']
        img_features = img_features.drop(d_col_drops,axis=1).values
        return img_features

def image_features(path, tt, photo_id):
    #to do - add more features [OpenCV haarcascade / placement stats / counts] [image filters edges, color isolations, closing kernels, blob, etc.] [Add differnt image size patches, patch stats]
    s=[tt, photo_id]
    im = Image.open(path)
    xheight, xwidth = [20,20]
    im = im.resize((xheight, xwidth), Image.ANTIALIAS)
    im = im.convert('1') #binarize
    im_data = list(im.getdata())
    im_data = np.array([r if r == 0 else 1 for r in im_data]).reshape((20, 20))
    patches = extract_patches_2d(im_data, (4, 4))
    #print(patches.shape)
    for p in patches:
        p1 = re.sub('[\[\]\n ]', '', np.array_str(p))
        s.append(float(p1[:8] + "." + p1[8:]))
    f = open("data.csv","a")
    f.write((',').join(map(str, s)) + '\n')
    f.close()
    return

f = open("data.csv","w");
col = ['tt', 'photo_id']
for i in range(289):
     col.append("patch"+str(i))
f.write((',').join(map(str,col)) + '\n')
f.close()

if __name__ == '__main__':
    j = []
    cpu = multiprocessing.cpu_count(); #print (cpu);
    
    for s_ in range(0,len(X_train),cpu):     #train
        for i in range(cpu):
            i_=s_+i
            if (i_)<len(X_train):
                if i_ % 10000 == 0:
                    print("train ", i_)
                filename = X_train.path[i_]
                p = multiprocessing.Process(target=image_features, args=(filename,'train', X_train.img[i_],))
                j.append(p)
                p.start()
    j = []
    for s_ in range(0,len(X_test),cpu):     #test
        for i in range(cpu):
            i_=s_+i
            if (i_)<len(X_test):
                if i_ % 10000 == 0:
                    print("test ", i_)
                filename = X_test[i_]
                p = multiprocessing.Process(target=image_features, args=(filename,'test', id_test[i_],))
                j.append(p)
                p.start()
    
    while len(j) > 0: #end all jobs
        j = [x for x in j if x.is_alive()]
        time.sleep(1)
 
    print("Start Training/Predictions: ", round(((time.time() - start_time)/60),2))
    df_all = pd.read_csv('data.csv', index_col=None)
    df_all = df_all.reset_index(drop=True)
    train_drivers.columns = ['subject','classname','photo_id','path']
    df_all = pd.merge(df_all, train_drivers, how='left', on='photo_id')
    df_all = df_all.reset_index(drop=True)
    X_train = df_all[df_all['tt'] == 'train']
    X_train = X_train.reset_index(drop=True)
    y_train = X_train['classname'].str.get_dummies()
    X_test = df_all[df_all['tt'] == 'test']
    X_test.fillna(0, inplace=True)
    X_test = X_test.reset_index(drop=True)
    id_test = X_test["photo_id"].values

    rfr = ensemble.RandomForestClassifier(random_state=2016, n_jobs=-1)
    ovr = OneVsRestClassifier(rfr, n_jobs=-1)
    clf = pipeline.Pipeline([
            ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_img_features()),  
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        },
                n_jobs = -1
                )), 
        ('ovr', ovr)])
    model = clf.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    df = pd.concat((pd.DataFrame(id_test), pd.DataFrame(y_pred)), axis=1)
    df.columns = ['img','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
    df = df.replace(0.0, 0.1)
    df.to_csv('submission2.csv',index=False)
    print("Ready to submit: ", round(((time.time() - start_time)/60),2))