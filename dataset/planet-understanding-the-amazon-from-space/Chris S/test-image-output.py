import numpy as np
import pandas as pd 
from skimage import io
from skimage import transform
from tqdm import tqdm
import scipy



path_train = "../input/train-tif-v2/"
path_test = "../input/test-tif-v2/"
csv_train = pd.read_csv("../input/train_v2.csv")
csv_test = pd.read_csv("../input/sample_submission_v2.csv")

trainlen = csv_train["image_name"].count()
testlen = csv_test["image_name"].count()


## get a 64x64x4 np.array image (uint8)
def getimgarray(nmbr, train=True):
    if train:
        path = path_train
        name = csv_train["image_name"][nmbr]
    else:
        path = path_test
        name = csv_test["image_name"][nmbr]
    imgraw = io.imread(path + name + ".tif")
    return( (transform.downscale_local_mean(np.array(imgraw)/64.0, (4,4,1))).astype(dtype=np.uint8) )


def getmanyims(lower, upper, train=True):
    result = []
    if train:
        upper = np.minimum( upper, trainlen )
    else:
        upper = np.minimum( upper, testlen )
    for i in range( lower, upper ):
        result.append( getimgarray(i, train) )
    return result


imsperfile = 5000

#for i in tqdm( range(0, int(trainlen/2 / imsperfile) +  1 ), miniters=1 ):
for i in range(2,5):
    ims = getmanyims(i*imsperfile, (i+1)*imsperfile)
    np.savez_compressed("train_ims" + str(i) + ".npz", ims)

#ims = getmanyims(0,1000)

#np.savez_compressed("ims.npz", ims)