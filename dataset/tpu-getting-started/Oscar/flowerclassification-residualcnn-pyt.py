#Train a model from scratch in PyTorch and run evaluation
import tensorflow as tf
import sys
import io
import os
import subprocess
import math
import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
from collections import OrderedDict
import random
import torch
import torch.nn as nn
from PIL import Image

#Clone and import personal CNNWordReco repository
if ~os.path.isdir('CNNWordReco'):
    subprocess.call(['git', 'clone', 'https://github.com/saztorralba/CNNWordReco'])
if 'CNNWordReco' not in sys.path:
    sys.path.append('CNNWordReco')
from utils.cnn_func import train_model
from models.SimpleCNN import SimpleCNN

#Initialise all random numbers for reproducibility
def init_random(**kwargs):
    random.seed(kwargs['seed'])
    torch.manual_seed(kwargs['seed'])
    torch.cuda.manual_seed(kwargs['seed'])
    torch.backends.cudnn.deterministic = True
    
#Resize the set to the desired size
def resize_set(dataset,**kwargs):
    outset = np.zeros((dataset.shape[0],kwargs['ysize'],kwargs['xsize'],dataset.shape[3]),dtype=np.uint8)
    for j in range(dataset.shape[0]):
        img = Image.fromarray(dataset[j],mode='RGB')
        img = img.resize((kwargs['ysize'],kwargs['xsize']))
        outset[j] = np.array(img)
    return outset

#Augment by doing rotations and flips (7 per image plus original)
def augment_set(dataset,labels,train,**kwargs):
    outset = np.zeros((dataset.shape[0]*8,dataset.shape[1],dataset.shape[2],dataset.shape[3]),dtype=np.uint8)
    outset[0:dataset.shape[0]] = dataset
    for j in range(dataset.shape[0]):
        img = Image.fromarray(dataset[j],mode='RGB')
        mimg = img.transpose(Image.FLIP_LEFT_RIGHT)
        outset[dataset.shape[0]+j] = np.array(mimg)
        mimg = img.transpose(Image.ROTATE_90)
        outset[2*dataset.shape[0]+j] = np.array(mimg)
        mimg = img.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        outset[3*dataset.shape[0]+j] = np.array(mimg)
        mimg = img.transpose(Image.ROTATE_180)
        outset[4*dataset.shape[0]+j] = np.array(mimg)
        mimg = img.transpose(Image.ROTATE_180).transpose(Image.FLIP_LEFT_RIGHT)
        outset[5*dataset.shape[0]+j] = np.array(mimg)
        mimg = img.transpose(Image.ROTATE_270)
        outset[6*dataset.shape[0]+j] = np.array(mimg)
        mimg = img.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
        outset[7*dataset.shape[0]+j] = np.array(mimg)
    outlabs = np.tile(labels,8)
    return outset,outlabs

#Read the images and preprocess
def read_tfrecords(filepaths,train=False,**kwargs):
    images = list()
    classes = list()
    ids = list()
    for path in filepaths:
        for record in tf.compat.v1.io.tf_record_iterator(path):
            example = tf.train.Example()
            example.ParseFromString(record)

            img = example.features.feature['image'].bytes_list.value[0]
            img = Image.open(io.BytesIO(img))
            img = np.asarray(img)
            images.append(img)
            if 'class' in example.features.feature:
                label = example.features.feature['class'].int64_list.value[0]
                classes.append(label)
            iid = example.features.feature['id'].bytes_list.value[0].decode('ascii')
            ids.append(iid)
    images = np.array(images)
    if images.shape[1]!=kwargs['ysize'] or images.shape[2]!=kwargs['xsize']:
        images = resize_set(images,**kwargs)
    classes = np.array(classes)
    images, classes = augment_set(images,classes,train,**kwargs)
    if train:
        idx = [i for i in range(images.shape[0])]
        random.shuffle(idx)
        images = images[idx]
        classes = classes[idx]
    images = torch.from_numpy(np.transpose(images,(0,3,1,2)))
    classes = torch.from_numpy(classes)
    return images, classes, ids

#Get posteriors for a test set
def evaluate_model(testset,model,**kwargs):
    testlen = testset.shape[0]
    predictions = np.zeros((testlen,len(kwargs['vocab'])))
    nbatches = math.ceil(testlen/kwargs['batch_size'])
    with torch.no_grad():
        model = model.eval()
        with tqdm(total=nbatches,disable=(kwargs['verbose']<2)) as pbar:
            for b in range(nbatches):
                #Obtain batch
                X = testset[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size'])].clone().float().to(kwargs['device'])
                #Propagate
                posteriors = model(X)
                predictions[b*kwargs['batch_size']:min(testlen,(b+1)*kwargs['batch_size']),:] = posteriors.detach().cpu().numpy()
                pbar.set_description('Testing')
                pbar.update()
    return predictions

#Compute the class-based F1 metric
def compute_classF1(posteriors,targets):
    predictions = np.argmax(posteriors,axis=1)
    f1 = list()
    for t in np.unique(targets):
        pos = np.where(targets==t)[0]
        neg = np.where(targets!=t)[0]
        tp = len(np.where(predictions[pos]==t)[0])
        fp = len(np.where(predictions[neg]==t)[0])
        fn = len(np.where(predictions[pos]!=t)[0])
        recall = ((tp / (tp + fn)) if (tp+fn)>0 else 0)
        precision = ((tp / (tp + fp)) if (tp+fp)>0 else 0)
        f1.append((2*precision*recall/(precision+recall)) if precision>0 and recall>0 else 0.0)
    return np.array(f1)

#Arguments
args = {
    'xsize': 64,
    'ysize': 64,
    'num_blocks': 4,
    'channels': 48,
    'input_channels': 3,
    'dropout': 0.05,
    'embedding_size': 256,
    'epochs': 10,
    'batch_size': 192,
    'learning_rate': 0.001,
    'seed': 0,
    'device': ('cuda:0' if torch.cuda.is_available() else 'cpu'),
    'verbose': 1,
    'augment': False,
}

#Initialise RNGs
init_random(**args)

print('Loading data...')
train_files = glob.glob("../input/tpu-getting-started/tfrecords-jpeg-192x192/train/*.tfrec")
val_files = glob.glob("../input/tpu-getting-started/tfrecords-jpeg-192x192/val/*.tfrec")
test_files = glob.glob("../input/tpu-getting-started/tfrecords-jpeg-192x192/test/*.tfrec")
train_data, train_targets, _ = read_tfrecords(train_files,True,**args)
val_data, val_targets, _ = read_tfrecords(val_files,False,**args)
test_data, _, test_ids = read_tfrecords(test_files,False,**args)

#Mapping of outputs
args['vocab'] = OrderedDict({t:i for i,t in enumerate(np.unique(list(train_targets)))})

#Mean and standard deviation for input normalisation
args['mean'] = torch.mean(train_data.float())
args['std'] = torch.std(train_data.float())

#Build model, optimizer and weighted criterion
model = SimpleCNN(**args).to(args['device'])
optimizer = torch.optim.Adam(model.parameters(),lr=args['learning_rate'])
priors = torch.Tensor([len(np.where(train_targets.numpy()==t)[0])/train_targets.shape[0] for t in np.unique(train_targets)])
criterion = nn.NLLLoss(weight = 1 / (priors / torch.max(priors)),reduction='mean').to(args['device'])

print('Training...')
best_f1 = 0.0
targets = val_targets[0:int(val_targets.shape[0]/8)].numpy()
for ep in range(1,args['epochs']+1):
    #Train an epoch
    loss = train_model(train_data,train_targets,model,optimizer,criterion,**args)
    #Get the posteriors for the validation set
    val_preds = np.exp(evaluate_model(val_data,model,**args))
    #Compute accuracy and F1
    val_preds = np.mean([val_preds[i*int(val_preds.shape[0]/8):(i+1)*int(val_preds.shape[0]/8)] for i in range(8)],axis=0)
    acc = 100*len(np.where((np.argmax(val_preds,axis=1)-targets)==0)[0])/len(targets)
    f1 = np.mean(compute_classF1(val_preds,targets))
    print('Epoch {0:d}, loss: {1:.2f}, acc: {2:.2f}%, f1: {3:.3f}'.format(ep,loss,acc,f1))
    if f1 >= best_f1:
        #Get the posteriors for the test set
        test_preds = np.exp(evaluate_model(test_data,model,**args))
        
#Combine the posteriors for each of the 8 augmentations
predictions = np.argmax(np.mean([test_preds[i*int(test_preds.shape[0]/8):(i+1)*int(test_preds.shape[0]/8)] for i in range(8)],axis=0),axis=1)

#Write output
df_out = pd.DataFrame({'id': test_ids, 'label': predictions.astype(int)})
df_out.to_csv('/kaggle/working/submission.csv',index=False)