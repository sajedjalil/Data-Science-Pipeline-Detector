# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import cv2
import glob
import os
np.random.seed(2016)


def rle_encode(img, order='F', format=True):
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []
    r = 0
    pos = 1
    for c in bytes:
        if c == 0:
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    z = ''
    for rr in runs:
        z += str(rr[0]) + ' ' + str(rr[1]) + ' '
    return z[:-1]


def find_best_mask():
    files = glob.glob(os.path.join("..", "input", "train", "*_mask.tif"))
    overall_mask = cv2.imread(files[0], cv2.IMREAD_GRAYSCALE)
    overall_mask.fill(0)
    overall_mask = overall_mask.astype(np.float32)

    for fl in files:
        mask = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
        overall_mask += mask
    overall_mask /= 255
    max_value = overall_mask.max()
    print('Max mask intersection:', max_value)
    overall_mask[overall_mask < max_value] = 0
    overall_mask[overall_mask == max_value] = 255
    overall_mask = overall_mask.astype(np.uint8)
    cv2.imwrite('common_mask.jpg', overall_mask)
    return overall_mask


def create_submission(mask):
    subm = open("subm.csv", "w")
    subm.write("img,pixels\n")
    encode = rle_encode(mask)
    files = glob.glob(os.path.join("..", "input", "test", "*.tif"))
    for fl in sorted(files):
        index = os.path.basename(fl[:-4])
        subm.write(index + ',' + encode + '\n')
    subm.close()


if __name__ == '__main__':
    mask = find_best_mask()
    create_submission(mask)
