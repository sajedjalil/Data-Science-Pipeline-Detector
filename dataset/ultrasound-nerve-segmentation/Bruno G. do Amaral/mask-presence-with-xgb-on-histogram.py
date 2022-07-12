import cv2
import glob
import re
import numpy as np
import os
import datetime
from xgboost.sklearn import XGBClassifier

threshold = 0.7  # Theshold for prediction proba (higher values may lead to more conservative predictions)

# Return normalized histogram
def nhist(im):
    hist = np.bincount(im.ravel(), minlength=256).reshape(256, 1)
    return hist.astype(np.float32) / hist.max()

# Load train images
im_files_mask = glob.glob("../input/train/*_mask.tif")
im_files = list(map(lambda f: f.replace("_mask", ""), im_files_mask))
assert len(im_files) == len(im_files_mask)

print("Preparing training data...")
imgs = np.array([cv2.imread(f, -1) for f in im_files])
hists = [nhist(im) for im in imgs]
n_imgs = len(imgs)

# Prepare X dataset
X_train = np.zeros((n_imgs, n_imgs))
for i in range(n_imgs):
    X_train[i, i] = 1.0    # Self-comparation is always 1
    for j in range(i):
        X_train[i, j] = cv2.compareHist(hists[i], hists[j], cv2.HISTCMP_CORREL)
        X_train[j, i] = X_train[i, j]

# Free imgs memory
del imgs

# Load test images
print("Preparing test data...")
im_files_test = glob.glob("../input/test/*.tif")
imgs_ids_test = list(map(lambda f: re.findall(r'\d+', f)[0], im_files_test))
imgs_test = np.array([cv2.imread(f, -1) for f in im_files_test])
hists_test = [nhist(im) for im in imgs_test]
n_imgs_test = len(imgs_test)
del imgs_test

# Prepare X_test dataset
X_test = np.zeros((n_imgs_test, n_imgs))
for i in range(n_imgs_test):
    for j in range(n_imgs):
        X_test[i, j] = cv2.compareHist(hists_test[i], hists[j], cv2.HISTCMP_CORREL)

print("Reading mask information...")
y = np.array([cv2.imread(f, -1).max() > 0 for f in im_files_mask]) * 1

print("Training classifier...")
classifier = XGBClassifier(max_depth=12, learning_rate=0.1, n_estimators=70, subsample=0.75, colsample_bytree=0.75)
classifier.fit(X_train, y)

print("Testing and generating submission...")

# Best mask image used by https://www.kaggle.com/zfturbo/ultrasound-nerve-segmentation/keras-is-there-any-nerve/output
mask = '116909 3 117324 15 117737 27 118155 33 118573 37 118990 41 119408 44 119826 48 120244 52 120662 54 121081 56 121500 59 121918 63 ' \
    '122336 67 122754 70 123174 71 123592 74 124011 76 124430 78 124849 80 125268 81 125686 83 126106 84 126525 85 126943 88 127363 89 127782 91 ' \
    '128201 92 128620 93 129039 94 129459 95 129878 96 130298 96 130717 98 131137 98 131556 99 131975 101 132395 100 132814 101 133234 101 133653 103 ' \
    '134072 104 134492 104 134912 104 135331 105 135751 104 136170 105 136590 105 137010 105 137430 105 137849 105 138269 105 138689 105 139109 104 ' \
    '139529 104 139949 103 140369 103 140789 103 141209 103 141629 102 142048 103 142469 102 142889 101 143309 101 143729 101 144150 100 144569 100 ' \
    '144990 98 145410 98 145829 99 146249 98 146669 97 147089 96 147508 96 147928 96 148348 95 148768 94 149188 92 149608 92 150027 91 150447 90 ' \
    '150867 89 151287 88 151708 87 152128 86 152548 85 152968 84 153388 83 153809 81 154229 80 154650 78 155070 77 155491 75 155910 75 156331 72 ' \
    '156752 70 157172 69 157593 67 158013 65 158433 64 158854 61 159276 58 159698 54 160120 51 160540 49 160962 45 161383 42 161806 36 162228 31 ' \
    '162652 24 163074 17 163494 14 163922 1'

y_hat = classifier.predict_proba(X_test)[:,1]

# Submission routine
sub_file = os.path.join('submission_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
subm = open(sub_file, "w")
subm.write("img,pixels\n")
for proba, test_id in zip(y_hat, imgs_ids_test):
    subm.write(test_id + ',')
    if proba > threshold:
        subm.write(mask)
    subm.write('\n')
subm.close()