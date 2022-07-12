#
# This script uses a pretrained ResNet model of 152 layers in CNTK and a boosted tree in LightGBM to 
# classify the data. It takes the next to last layer of ResNet to generate the features. Then
# they are averaged and fed to a tree. 
#
# With this configuration we were able to get a score on the leaderboard of 0.55979 with a execution
# time of 54min. Using ResNet 18, the score was 0.5708 and the execution time was 31min.
#
# This script is based on
# https://www.kaggle.com/drn01z3/data-science-bowl-2017/mxnet-xgboost-baseline-lb-0-57 
#

#Load libraries
import sys,os
import numpy as np
import dicom
import glob
import cv2
import time
import pandas as pd
from sklearn import cross_validation
from cntk import load_model
from cntk.ops import combine
from cntk.io import MinibatchSource, ImageDeserializer, StreamDef, StreamDefs
from lightgbm.sklearn import LGBMRegressor


#Put here the number of your experiment
EXPERIMENT_NUMBER = '0042'

#Put here the path to the downloaded ResNet model
#model available at : https://migonzastorage.blob.core.windows.net/deep-learning/models/cntk/imagenet/ResNet_152.model
#other model: https://www.cntk.ai/Models/ResNet/ResNet_18.model
MODEL_PATH='data/ResNet_152.model'

#Maximum batch size for the network to evaluate. 
BATCH_SIZE=60

#Put here the path where you downloaded all kaggle data
DATA_PATH='data/'

# Path and variables
STAGE1_LABELS=DATA_PATH + 'stage1_labels.csv'
STAGE1_SAMPLE_SUBMISSION=DATA_PATH + 'stage1_sample_submission.csv'
STAGE1_FOLDER=DATA_PATH + 'stage1/'
FEATURE_FOLDER=DATA_PATH + 'features/features' + EXPERIMENT_NUMBER + '/'
SUBMIT_OUTPUT='submit' + EXPERIMENT_NUMBER + '.csv'


class Timer(object):
    """Timer class."""
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    def start(self):
        self.start = time.clock()

    def stop(self):
        self.end = time.clock()
        self.interval = self.end - self.start


def get_extractor():
    """Load the CNN."""
    node_name = "z.x"
    loaded_model  = load_model(MODEL_PATH)
    node_in_graph = loaded_model.find_by_name(node_name)
    output_nodes  = combine([node_in_graph.owner])
    
    return output_nodes


def get_3d_data(path):
    """Get the 3D data."""
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    return np.stack([s.pixel_array for s in slices])


def get_data_id(path):
    """Convert the images in the format accepted by the network trained on ImageNet, packing the 
    images in groups of 3 gray images with size of 224x224 and performing some operations. 
    
    """
    sample_image = get_3d_data(path)
    sample_image[sample_image == -2000] = 0
    batch = []
    cnt = 0
    dx = 40
    ds = 512
    for i in range(0, sample_image.shape[0] - 3, 3):
        tmp = []
        for j in range(3):
            img = sample_image[i + j]
            img = 255.0 / np.amax(img) * img
            img = cv2.equalizeHist(img.astype(np.uint8))
            img = img[dx: ds - dx, dx: ds - dx]
            img = cv2.resize(img, (224, 224))
            tmp.append(img)

        tmp = np.array(tmp)
        batch.append(np.array(tmp))
   
    batch = np.array(batch, dtype='int')
    return batch


def batch_evaluation(model, data, batch_size=50):
    """Evaluation of the data in batches too avoid consuming too much memory"""
    num_items = data.shape[0]
    chunks = np.ceil(num_items / batch_size)
    data_chunks = np.array_split(data, chunks, axis=0)
    feat_list = []
    for d in data_chunks:
        feat = model.eval(d)
        feat_list.append(feat)
    feats = np.concatenate(feat_list, axis=0)
    return feats
    

def calc_features(verbose=True):
    """Execute the forward propagation on the images to obtain the features
    and save them as numpy arrays.
    
    """
    if verbose: print("Compute features")
    net = get_extractor()
    for folder in glob.glob(STAGE1_FOLDER+'*'):
        foldername = os.path.basename(folder)
        if os.path.isfile(FEATURE_FOLDER+foldername+'.npy'):
            if verbose: print("Features in %s already computed" % (FEATURE_FOLDER+foldername))
            continue
        batch = get_data_id(folder)
        if verbose:
            print("Batch size:")
            print(batch.shape)
        feats = batch_evaluation(net, batch, BATCH_SIZE)
        if verbose:
            print(feats.shape)
            print("Saving features in %s" % (FEATURE_FOLDER+foldername))
        np.save(FEATURE_FOLDER+foldername, feats)


def train_lightgbm(verbose=True):
    """Train a boosted tree with LightGBM."""
    if verbose: print("Training with LightGBM")
    df = pd.read_csv(STAGE1_LABELS)
    x = np.array([np.mean(np.load(FEATURE_FOLDER+'%s.npy' % str(id)), axis=0).flatten() for id in df['id'].tolist()])
    y = df['cancer'].as_matrix()

    trn_x, val_x, trn_y, val_y = cross_validation.train_test_split(x, y, random_state=42, stratify=y,
                                                                   test_size=0.20)
    clf = LGBMRegressor(max_depth=50,
                        num_leaves=21,
                        n_estimators=5000,
                        min_child_weight=1,
                        learning_rate=0.001,
                        nthread=24,
                        subsample=0.80,
                        colsample_bytree=0.80,
                        seed=42)
    clf.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], verbose=verbose, eval_metric='l2', early_stopping_rounds=300)
    
    return clf


def compute_training(verbose=True):
    """Wrapper function to perform the training."""
    if verbose: print("Compute training")
    with Timer() as t:
        clf = train_lightgbm()
    if verbose: print("Training took %.03f sec.\n" % t.interval)
    return clf


def compute_prediction(clf, verbose=True):  
    """Wrapper function to perform the prediction."""
    if verbose: print("Compute prediction")
    df = pd.read_csv(STAGE1_SAMPLE_SUBMISSION)
    x = np.array([np.mean(np.load((FEATURE_FOLDER+'%s.npy') % str(id)), axis=0).flatten() for id in df['id'].tolist()])
    
    with Timer() as t:
        pred = clf.predict(x)
    if verbose: print("Prediction took %.03f sec.\n" % t.interval)
    df['cancer'] = pred
    return df


def save_results(df, verbose=True):
    """Wrapper function to save the results."""
    if verbose: print("Save results to csv")
    df.to_csv(SUBMIT_OUTPUT, index=False)
    if verbose: 
        print("Results:")
        print(df.head())


if __name__ == "__main__":
    
    calc_features(verbose=False)
    clf = compute_training()
    df = compute_prediction(clf)
    save_results(df)





