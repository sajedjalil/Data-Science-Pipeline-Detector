# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.

import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
import itertools as it
from sklearn.model_selection import StratifiedShuffleSplit
from operator import itemgetter

tf.logging.set_verbosity(tf.logging.INFO)
#tf.reset_default_graph()

data = pd.read_csv('../input/train.csv')
train_data = data[~(data.Id == "new_whale")]
#train_data.to_csv("./train_data_16k.csv", index=False)
files = "../input/train/"+train_data.Image.values
labels = train_data.Id.values
le = LabelEncoder()
true_labels = le.fit_transform(np.asarray(train_data["Id"]).reshape(-1,1)).astype(np.int32)
IMG_SIZE=64
path = "../input/train/"
BATCH = 20
def SiameseNet(x):
#the distance calculation and last dense layer are moved to model_fn
#
    features = tf.reshape(x, [-1, 64, 64, 3])
        
    network = tf.layers.conv2d(inputs=features,
                            filters=64,
                            kernel_size=10,
                            padding='SAME',
                            activation = tf.nn.relu,
                            kernel_initializer = tf.contrib.layers.xavier_initializer())
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

    network = tf.layers.conv2d(inputs=network,
                            filters=128,
                            kernel_size=7,
                            padding='SAME',
                            activation = tf.nn.relu,
                            kernel_initializer = tf.contrib.layers.xavier_initializer())
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

    network = tf.layers.conv2d(inputs=network,
                         filters=128,
                         kernel_size=4,
                         padding='SAME',
                         activation = tf.nn.relu,
                         kernel_initializer = tf.contrib.layers.xavier_initializer())
    network = tf.layers.max_pooling2d(network, pool_size=[2,2], strides=2)

    network = tf.layers.conv2d(inputs=network,
                         filters=256,
                         kernel_size=4,
                         padding='SAME',
                         activation = tf.nn.relu,
                         kernel_initializer = tf.contrib.layers.xavier_initializer())
    #print(network)
    network_flat = tf.reshape(network, [-1, 256*8*8])

    dense_nw = tf.layers.dense(inputs=network_flat, units=4096, activation=tf.nn.sigmoid)
    
    return dense_nw 
    
def model_fn(mode, features, labels):
        (img1, img2) = features
        #img1, img2 = tf.split(features, 2, axis=2)
        dense1 = SiameseNet(img1)
        dense2 = SiameseNet(img2)
        #l1_dist = tf.reshape(tf.norm(tf.subtract(dense1,dense2), 1, 3), (BATCH,4096))
        l1_dist = tf.reshape(tf.abs(tf.subtract(dense1,dense2)), (-1,4096))
        #print(l1_dist)
        y_ = tf.layers.dense(inputs=l1_dist, units=1, activation=tf.nn.sigmoid)
        y_ = tf.reshape(y_, (-1,1))
        #print(y_)
        
        #print(labels)
        loss=None
        train_op=None
        predictions=None
        eval_metric_ops=None
        global_step = tf.train.get_global_step()
        if(mode==tf.estimator.ModeKeys.EVAL or
                mode==tf.estimator.ModeKeys.TRAIN):
            labels = tf.cast(tf.reshape(labels, [BATCH,1]),tf.float32)
            loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy(
                        multi_class_labels=labels, logits=y_))

        if(mode==tf.estimator.ModeKeys.TRAIN):
            train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss, global_step = global_step)

        if(mode == tf.estimator.ModeKeys.EVAL):
            eval_metric_ops = {"absolute error": tf.metrics.mean_absolute_error(labels,y_)}

        predictions = {"probabilities": y_} #"classes": tf.round(y_),

        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions,
                        loss=loss, train_op=train_op, eval_metric_ops = eval_metric_ops)

def imgprcs(file, label):
    #print(img)
    with tf.device('/gpu:0'):
        #img = path+imgg
        img = tf.io.read_file(file)
        oh = tf.image.extract_jpeg_shape(img)
        img = tf.image.decode_jpeg(img)
        img = tf.cond(tf.less(oh[2],3), lambda: tf.image.grayscale_to_rgb(img), lambda: img)
        #img = tf.image.rgb_to_grayscale(img)
        img = tf.image.resize_images(img, [IMG_SIZE, IMG_SIZE])
        #img = tf.image.per_image_standardization(img)
        img = tf.reshape(img, [64,64,3])
        return img

def input_fn():
    with tf.device('/gpu:0'):
        ds1 = tf.data.Dataset.from_tensor_slices((files, labels))
        ds1 = ds1.map(imgprcs, 4)
        ds1 = ds1.repeat()
        ds1 = ds1.shuffle(10)
        itr1=ds1.make_one_shot_iterator()
        img1,l1 = itr1.get_next()

        ds2 = tf.data.Dataset.from_tensor_slices((files, labels))
        ds2 = ds2.map(imgprcs, 4)
        ds2 = ds2.repeat()
        ds2 = ds2.shuffle(10)
        itr2=ds2.make_one_shot_iterator()
        img2,l2 = itr2.get_next()

        y=tf.cast(tf.equal(l1,l2),tf.int32)
        z=(img1,img2)
        f, l=tf.contrib.training.stratified_sample([z], y,
                            batch_size=20, enqueue_many=False, queue_capacity=100000,
                                            target_probs=tf.convert_to_tensor([0.5,0.5]),  threads_per_queue=2)

        return f, l
with tf.device('/gpu:0'):        
    est = tf.estimator.Estimator(model_fn=model_fn, model_dir='./model_dir')
    #for _ in range(1):
    #    est.train(input_fn = input_fn, steps = 1)
    #    results=est.evaluate(input_fn=input_fn,steps=1)
    #    print(results) 
    
def pred_ip_fn():
    with tf.device('/gpu:0'): 
#   this repetition of code is required to cover the same images of same class  
        test_files = glob.glob("../input/test/*.jpg")
        ds1 = tf.data.Dataset.from_tensor_slices((test_files, [1 for i in range(7960)]))
        ds1 = ds1.map(imgprcs, 4)
        ts_itr = ds1.make_one_shot_iterator()
    
        ds2 = tf.data.Dataset.from_tensor_slices((files, labels))
        ds2 = ds2.map(imgprcs,4)
        tr_itr = ds2.make_one_shot_iterator()
        
        for _ in range(10):   #len of test
            img1, l1 = ts_itr.get_next()
            for _ in range(15697):    #len of train
                img2,l2 = tr_itr.get_next()
                return (img1, img2), None 
                
imgs=train_data.Image.values                
test_files = glob.glob("../input/test/*.jpg")
product1 = it.product(test_files, "../input/train/"+imgs)
def pred_input_fn():
    with tf.device('/gpu:0'):
        img1=list()
        img2=list()
        for i,(x1,x2) in enumerate(product1):
            img1.append(imgprcs(x1,""))
            img2.append(imgprcs(x2,""))
            if((i+1)%BATCH == 0):
                return (img1, img2), None

with tf.device('/gpu:0'):
    predictions=est.predict(input_fn=pred_input_fn)
    #for p in predictions:
    #    print(p["probabilities"])

    imgs=train_data.Image.values
    train_data_dict = train_data.set_index('Image').T.to_dict('dict')
    test_files = glob.glob("../input/test/*.jpg")
    product = it.product(files[:10],imgs)

    result_top5=list()
    cnt = [i+1 for i in range(100000)]
    probs=list()
    for i,(x1,x2),pred in zip(cnt, product,predictions) :
        [p] = pred["probabilities"]
        probs.append(p)
        print(i,p)
        if(i%15697 ==0):
            print(probs)
            args = np.flip(np.argsort(probs))
            print(args)
            top5 = list(itemgetter(*args)(imgs))[:5]
            print(top5)
            tt = list()
            for t in top5:
                tt.append(train_data_dict[t]["Id"])
            print(i, x1, tt, flush=True)
            result_top5.append([x1,tt])
            probs.clear()
            break
    print(result_top5)
    
with open("whale_Detection_submit.csv","w") as ff:
    ff.writelines(pred_result_top5)
'''    
img1=list()
img2=list()
label=list()

def pre_processing():
    imgs=train_data.Image.values
    #train_data.set_index('Image')
    #print(train_data)
    train_data_dict = train_data.set_index('Image').T.to_dict('dict')
    #print(train_data_dict['0000e88ab.jpg'])
    perm = it.combinations_with_replacement(imgs, 2)
    
    for _,p in enumerate(perm):
        (x1,x2) = p
        if(train_data_dict[x1]['Id']==train_data_dict[x2]['Id']): y=1.0
        else: y=0.0
        img1.append(x1)
        img2.append(x2)
        label.append(y)
        #break for the sake of commit
        if(len(label) == 5000): break
    
    sss = StratifiedShuffleSplit(n_splits=1, train_size=1000, test_size=100, random_state=0)
    (train_index, test_index) = next(sss.split(img1, label))
    return train_index, test_index
    
train_index, test_index = pre_processing()

image1 = list(itemgetter(*train_index)(img1))
image2 = list(itemgetter(*train_index)(img2))
train_label = list(itemgetter(*train_index)(label))

test_image1  = list(itemgetter(*test_index)(img1))
test_image2  = list(itemgetter(*test_index)(img2))
test_label = list(itemgetter(*test_index)(label))

print("creating the dataset..\n")
def batched_input_fn():
    idx =np.random.choice(1000, BATCH, replace=False)
    x1 = list()
    x2 = list()
    yy=list()

    for n in idx:
        x1.append(imgprcs(image1[n],"../input/train/"))
        x2.append(imgprcs(image2[n],"../input/train/"))
        yy.append(train_label[n])

    x1 = tf.reshape(tf.convert_to_tensor(x1), (BATCH, 64, 64, 1))
    x2 = tf.reshape(tf.convert_to_tensor(x2), (BATCH, 64, 64, 1))
    yy = tf.reshape(tf.convert_to_tensor(yy), (BATCH, 1))
    #print(x1, x2,yy)
    return (x1, x2), yy
    
def eval_input_fn():
    idx =np.random.choice(100, BATCH, replace=False)
    x1 = list()
    x2 = list()
    yy=list()

    for n in idx:
        x1.append(imgprcs(test_image1[n],"../input/train/"))
        x2.append(imgprcs(test_image2[n],"../input/train/"))
        yy.append(test_label[n])
        
    x1 = tf.reshape(tf.convert_to_tensor(x1), (BATCH, 64, 64, 1))
    x2 = tf.reshape(tf.convert_to_tensor(x2), (BATCH, 64, 64, 1))
    yy = tf.reshape(tf.convert_to_tensor(yy), (BATCH, 1))
    #print(x1, x2,yy)
    return (x1, x2), yy
    
est = tf.estimator.Estimator(model_fn=model_fn, model_dir='./model_dir')
for _ in range(1):
    results= est.train(input_fn = batched_input_fn, steps = 100)
    print(results)
    results=est.evaluate(input_fn=eval_input_fn,steps=10)
    print(results) # for local run 

imgs=train_data.Image.values
train_data_dict = train_data.set_index('Image').T.to_dict('dict')
files = glob.glob("../input/test/*.jpg")
product = it.product(files[:10],"../input/train/"+imgs)
predictions_list = list()
def pred_input_fn():
    img1=list()
    img2=list()
    for i,(x1,x2) in enumerate(product):
        #print(x1, x2)
        img1.append(imgprcs(x1,""))
        img2.append(imgprcs(x2,""))
        if((i+1)%20 == 0):
            #np.reshape(img1, (20, 64,64,1))
            #np.reshape(img2, (20, 64,64,1))
            return (img1, img2), None
            
predictions = est.predict(input_fn=pred_input_fn,predict_keys="probabilities")

#for i,pred in enumerate(predictions):
#    print(pred)
result_top5=list()
cnt = [i for i in range(100000)]
probs=list()
for i,(x1,x2),pred in zip(cnt, product,predictions) :
    #print(x1,x2,pred)
    #for i in range(15697):
    #for i in range(len(product)):
    [p] = pred["probabilities"]
    probs.append(p)
    #print(i)
    if((i+1)%15697 ==0):
        print(probs)
        args = np.flip(np.argsort(probs))
        print(args)
        top5 = list(itemgetter(*args)(imgs))[:5]
        print(top5)
        #[lbltop5] = itemgetter(*top5)(train_data_dict)["Id"]
        tt = list()
        for t in top5:
            tt.append(train_data_dict[t]["Id"])
        #rslt = train_data.loc[train_data["Image"]==top5, 'Id']
        #print(rslt.drop(index))
        print(tt)
        result_top5.append([x1,tt])
        probs.clear()
        break
print(result_top5)
        
#def input_fn():
#    dataset = tf.data.Dataset.from_tensor_slices((img1[train_index], img2[train_index]), label[train_index])
#    batch_dataset = dataset.batch(20)
#    dataset = dataset.map(imgprcs(img1), 2)
#    dataset = dataset.map(imgprcs(img2), 2)
#    itr = batch_dataset.make_one_shot_iterator()
#    return itr.get_next()
    #for i in range(1000):
    #(x1, x2) = np.random.choice(15690, 2)
    #if train_data.iloc[x1,1] == train_data.iloc[x2,1] : y = 1
    #else: y = 0
    #element1 = train_data.iloc[x1,0] #itr.get_next()
    #element2 = train_data.iloc[x2,0] #itr.get_next()
    #feat1 = imgprcs(element1)
    #feat2 = imgprcs(element2)
    #feat1, feat2 = prcs(feat)
    #feat1 = tf.reshape(tf.convert_to_tensor(feat1), (1,64,64,1))
    #feat2 = tf.reshape(tf.convert_to_tensor(feat2), (1,64,64,1))
    #y = tf.convert_to_tensor(y)
    #dataset = tf.data.Dataset.from_tensor_slices((feat1, y))
    #batch_dataset = dataset.batch(1)
    #dataset = dataset.map(dummy(), 4)
    #itr = batch_dataset.make_one_shot_iterator()
    #return img1, img2, y #itr.get_next() #next_element
'''    
'''
imgs=train_data.Image.values
#print(len(imgs))
#train_data.set_index('Image')
#print(train_data)
train_data_dict = train_data.set_index('Image').T.to_dict('dict')
#print(train_data_dict)
perm = it.combinations_with_replacement(imgs, 2)
img1=list()
img2=list()
label=list()
for i,p in enumerate(perm):
#for _ in range(10): #enumerate(perm):
    #if i > 100000: break
    (x1,x2) = p #next(perm)
    if(train_data_dict[x1]['Id']==train_data_dict[x2]['Id']): y=1
    else: y=0
    #print(x1, x2, y)
    img1.append(x1)
    img2.append(x2)
    label.append(y)
print(label.count(0))
img1 = np.reshape(np.asarray(img1),(100001,1))
label = np.reshape(np.asarray(label),(100001,1))
print(len(img1))
sss = StratifiedShuffleSplit(n_splits=1, train_size=1000, test_size=None, random_state=0)
gen = sss.split(img1, label)
print(next(gen))
#train_index = sss.split(img1, label)
#print(train_index)#X_train, X_test = X[train_index], X[test_index]
#y_train, y_test = y[train_index], y[test_index]
#img1 = np.reshape(img1, (-1,1))
#img2 = np.reshape(img2, (-1,1))
#label = np.reshape(label,(-1,1))

#ip_df = pd.DataFrame.from_dict([{"img1":img1, "img2":img2, "label":label}]) #, columns=['img1', 'img2', 'label'])
#print(ip_df)
'''    