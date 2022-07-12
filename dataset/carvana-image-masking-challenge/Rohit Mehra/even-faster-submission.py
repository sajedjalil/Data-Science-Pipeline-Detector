## Modified Version from https://www.kaggle.com/c/carvana-image-masking-challenge/discussion/37523
import pandas as pd
import numpy as np
import cv2

## Import required info from your training script
from train_unet_general import INPUT_SHAPE, batch_size, model
from skimage.transform import resize

## Used to save time
from multiprocessing import Pool

import time
import gc

## Configure with number of CPUs you have or the number of processes to spin ##
CPUs = 48

## Tune it; used in generator
batch_size = batch_size + 6

## Mask properties
WIDTH_ORIG = 1918
HEIGHT_ORIG = 1280

## More Tuning
MASK_THRESHOLD = 0.6

## Submission data
df_test = pd.read_csv('input/sample_submission.csv')
print('sample_submission.csv shape:: ', df_test.shape)
print('sample_submission.csv columns:: ', df_test.columns.values.tolist())
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

## load it up
model = load_model('weights/best_model.hdf5')


## will be used in making submission
names = []
for id in ids_test:
    names.append('{}.jpg'.format(id))


## https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder
def run_length_encode(img):
    img = cv2.resize(img, (WIDTH_ORIG, HEIGHT_ORIG))
    flat_img = img.flatten()
    flat_img[0] = 0
    flat_img[-1] = 0
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    encoding = ''
    for idx in range(len(starts_ix)):
        encoding += '%d %d ' % (starts_ix[idx], lengths[idx])
    return encoding.strip()


rles = []

##  Split cz you can't keep all the images in memory at once
test_splits = 59  # Split test set (number of splits must be multiple of 59
ids_test_splits = np.split(ids_test, indices_or_sections=test_splits)
split_count = 0


## predict and collect rles here on splits
for ids_test_split in ids_test_splits:

    split_count += 1
    hm_samples_here = len(ids_test_split)
    
    ## generator on the small split we did earlier; batch variable used here
    def test_generator():
        while True:
            for start in range(0, len(ids_test_split), batch_size):
                x_batch = []
                end = min(start + batch_size, hm_samples_here)
                ids_test_split_batch = ids_test_split[start:end]
                for id in ids_test_split_batch.values:
                    img = cv2.imread('input/test/{}.jpg'.format(id))
                    img = cv2.resize(img, INPUT_SHAPE)
                    x_batch.append(img)
                x_batch = np.array(x_batch, np.float32) / 255
                yield x_batch

    print("Predicting on {} samples (split {}/{})".format(len(ids_test_split),
                                                          split_count, test_splits))
    ## Predictions
    preds = model.predict_generator(generator=test_generator(),
                                    steps=np.ceil(
                                        float(len(ids_test_split)) / float(batch_size)),
                                    max_queue_size=10, use_multiprocessing=True, verbose=1)

    print("Prediction of {} samples done. Now Generating RLE masks...".format(
        hm_samples_here))
    
    ## lets do rle computation in parallel
    start = time.clock()
    pool = Pool(CPUs)
    split_rle = pool.map(run_length_encode, preds)
    rles = rles + split_rle
    del split_rle
    del preds
    gc.collect()

    print(len(rles))
    pool.close()
    pool.join()
    del pool

    print(time.clock() - start)

print("Generating submission file...")
df = pd.DataFrame({'img': names, 'rle_mask': rles})
df.to_csv('submissions/submission.csv.gz', index=False, compression='gzip')