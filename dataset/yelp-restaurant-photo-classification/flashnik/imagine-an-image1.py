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
    #fig.savefig(str(label_notation[l]) +'.png')
print("Start Training/Predictions: ", round(((time.time() - start_time)/60),2))
from sklearn import ensemble
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.image  import PatchExtractor
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
#from sklearn import svm
#from sklearn.feature_extraction.image import img_to_graph, extract_patches_2d
#from sklearn.metrics import f1_score
import multiprocessing
import random; random.seed(2016);

X_train = train_photos
X_train = X_train.groupby(['business_id'], as_index=False).first()
X_train = pd.merge(X_train, train_attr, how='left', on='business_id')
X_train = df_all = pd.concat((X_train.groupby(['labels'], as_index=False).first(), X_train.groupby(['labels'], as_index=False).last()), axis=0, ignore_index=True)
y_train = X_train['labels'].str.get_dummies(sep=' ')
X_train = X_train.drop(['labels'],axis=1)
X_test = test_photos.groupby(['business_id'], as_index=False).first()
id_test = X_test["business_id"]

print(len(X_train), len(y_train), len(X_test), len(id_test))

def image_features(path, tt, buss_id, photo_id):
    s=[tt, photo_id, buss_id]
    im = Image.open(path)
    xheight, xwidth = [100,100]
    im = im.resize((xheight, xwidth), Image.ANTIALIAS)
    qu = im.quantize(colors=10, kmeans=3) #if number of colors changes also change file columns number
    crgb = qu.convert('RGB')
    col_rank = sorted(crgb.getcolors(xwidth*xheight), reverse=True)
    for i_rgb in range(len(col_rank)):
        for t_rgb in range(4):
            if t_rgb==0:
                s.append(col_rank[i_rgb][0])
            else:
                s.append(col_rank[i_rgb][1][t_rgb-1])
    im = im.crop((10, 10, 90, 90)) #remove edges
    im = im.convert('1') #binarize
    im_data = list(im.getdata())
    im_data = [r if r == 0 else 1 for r in im_data]
    st = str("".join(map(str,im_data)))
    for i in range(0,len(im_data)//16):
        t = str(st[16*i:16*i+8]) + "." + str(st[16*i+8:16*(i+1)])
        s.append(float(t))
    f = open("data.csv","a")
    f.write((',').join(map(str, s)) + '\n')
    f.close()
    return

class cust_img_features(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, img_features):
        d_col_drops=['photo_id','tt']
        img_features = img_features.drop(d_col_drops,axis=1).values
        return img_features

class cust_patch_arr(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, img_features):
        if img_features["tt"][0]=="test":
            img_features["pic"] = img_features["photo_id"].map(lambda x: np.asarray(Image.open('../input/test_photos/' + str(x) + '.jpg')))
        else:
            img_features["pic"] = img_features["photo_id"].map(lambda x: np.asarray(Image.open('../input/train_photos/' + str(x) + '.jpg')))
        return img_features["pic"]
f = open("data.csv","w");
col = ['tt', 'photo_id','business_id']
for i_rgb in range(10):
    for t_rgb in range(4):
        col.append("col_feature_"+str(i_rgb)+"_" + "krgb"[t_rgb])
for i in range(400):
     col.append("img_pixel_set"+str(i))
f.write((',').join(map(str,col)) + '\n')
f.close()
print("Start Training/Predictions: ", round(((time.time() - start_time)/60),2))

if __name__ == '__main__':
    j = []
    cpu = multiprocessing.cpu_count(); #print (cpu);
    
    for s_ in range(0,len(X_train),cpu):     #train
        for i in range(cpu):
            i_=s_+i
            if (i_)<len(X_train):
                filename='../input/train_photos/' + str(X_train.photo_id[i_]) + '.jpg'
                p = multiprocessing.Process(target=image_features, args=(filename,'train', X_train.business_id[i_], X_train.photo_id[i_],))
                j.append(p)
                p.start()
    j = []
    for s_ in range(0,len(X_test),cpu):     #test
        for i in range(cpu):
            i_=s_+i
            if (i_)<len(X_test):
                filename='../input/test_photos/' + str(X_test.photo_id[i_]) + '.jpg'
                p = multiprocessing.Process(target=image_features, args=(filename,'test', X_test.business_id[i_], X_test.photo_id[i_],))
                j.append(p)
                p.start()
    
    df_all = pd.read_csv('data.csv', index_col=None)
    X_train = df_all[df_all['tt'] == 'train']
    X_test = df_all[df_all['tt'] == 'test']
    X_train = X_train.drop(['business_id'],axis=1)
    X_test = X_test.drop(['business_id'],axis=1)
    rfr = ensemble.GradientBoostingClassifier(random_state=2016, n_estimators = 1000, max_depth = 5, subsample = 0.6, min_samples_leaf=5, learning_rate = 0.17)
    ovr = OneVsRestClassifier(rfr, n_jobs=-1)
    patch1 = PatchExtractor(patch_size=(7,7), max_patches=20, random_state=2016)
    clf = pipeline.Pipeline([
            ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_img_features()),  
                        #('patches', pipeline.Pipeline([('patch_arr', cust_patch_arr()), ('patch', patch1)]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        #'patches': 1.0
                        },
                n_jobs = -1
                )), 
        ('ovr', ovr)])
    model = clf.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    df = pd.concat((pd.DataFrame(id_test), pd.DataFrame(y_pred)), axis=1)
    df.columns = ['business_id','0','1','2','3','4','5','6','7','8']
    df.to_csv('data.csv',index=False)
    print("End Training/Predictions: ", round(((time.time() - start_time)/60),2))
df = pd.read_csv('data.csv')
a = [['business_id','labels']]
for i in range(len(df)):
    b = []
    for j in [0,1,2,3,5,6,8]:
        if df[str(j)][i] >= 0.1:
            b.append(j)
    a.append([df['business_id'][i]," ".join(map(str,b))])
pd.DataFrame(a).to_csv('submission.csv',index=False, header=False)
print('Done, not much better than random guessing but could increase train data too.')
