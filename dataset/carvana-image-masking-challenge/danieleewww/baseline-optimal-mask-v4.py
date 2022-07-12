# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import os
import glob
import time
import cv2
import pandas as pd
import random
from PIL import Image

random.seed(1234)
np.random.seed(5678)

NUM_OF_IMAGES_FROM_TRAIN = 600
INPUT_PATH = '../input/'
OUTPUT_PATH = './'


def rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    bytes = np.where(img.flatten() == 1)[0]
    runs = []
    prev = -2
    for b in bytes:
        if (b > prev + 1): runs.extend((b + 1, 0))
        runs[-1] += 1
        prev = b

    return ' '.join([str(i) for i in runs])


def dice(im1, im2, empty_score=1.0):
    im1 = im1.astype(np.bool)
    im2 = im2.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    intersection = np.logical_and(im1, im2)
    return 2. * intersection.sum() / im_sum


def get_score(train_masks, avg_mask, thr):
    d = 0.0
    for i in range(train_masks.shape[0]):
        d += dice(train_masks[i], avg_mask)
    return d/train_masks.shape[0]


def validation_get_optimal_thr():
    train_files = glob.glob(INPUT_PATH + 'train_masks/*.gif')[:NUM_OF_IMAGES_FROM_TRAIN]
    train_masks = []
    div_factor = 1

    avg_mask = np.zeros((1280 // div_factor, 1918 // div_factor), dtype=np.float64)
    print('AVG Mask shape: {}'.format(avg_mask.shape))
    for f in train_files:
        mask = np.array(Image.open(f), dtype=np.uint8)
        if div_factor != 1:
            mask = cv2.resize(mask, (mask.shape[1] // div_factor, mask.shape[0] // div_factor), cv2.INTER_LINEAR)
        # print(mask.min(), mask.max(), mask.mean())
        train_masks.append(mask)
        avg_mask += mask.astype(np.float64)
    avg_mask /= len(train_files)
    train_masks = np.array(train_masks, dtype=np.uint8)
    print(avg_mask.min(), avg_mask.max(), train_masks.shape)

    best_score = 0
    best_thr = -1
    for t in range(370, 400):
        thr = t/1000
        avg_mask_thr = avg_mask.copy()
        avg_mask_thr[avg_mask_thr > thr] = 1
        avg_mask_thr[avg_mask_thr <= thr] = 0
        score = get_score(train_masks, avg_mask_thr, thr)
        print('THR: {:.3f} SCORE: {:.6f}'.format(thr, score))
        if score > best_score:
            best_score = score
            best_thr = thr

    print('Best score: {} Best thr: {}'.format(best_score, best_thr))
    avg_mask_thr = avg_mask.copy()
    avg_mask_thr[avg_mask_thr > best_thr] = 1
    avg_mask_thr[avg_mask_thr <= best_thr] = 0
    avg_mask_thr = cv2.resize(avg_mask_thr, (1918, 1280), cv2.INTER_LINEAR)
    avg_mask_thr[avg_mask_thr > 0.5] = 1
    avg_mask_thr[avg_mask_thr <= 0.5] = 0
    print(avg_mask.shape, avg_mask_thr.shape)
    cv2.imwrite('avg_mask.jpg', (255*avg_mask_thr).astype(np.uint8))

    return best_score, avg_mask_thr


def create_submission(best_score, avg_mask):
    print('Create submission...')
    t = pd.read_csv(INPUT_PATH + 'sample_submission.csv')
    str = rle(avg_mask)
    t['rle_mask'] = str
    t.to_csv('subm_{}.gz'.format(best_score), index=False, compression='gzip')


if __name__ == '__main__':
    best_score, avg_mask = validation_get_optimal_thr()
    create_submission(best_score, avg_mask)

