# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import os
import cv2

def get_driver_data():
    dr = dict()
    clss = dict()
    path = os.path.join('..', 'input', 'driver_imgs_list.csv')
    print('Read drivers data')
    f = open(path, 'r')
    line = f.readline()
    while (1):
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        dr[arr[2]] = arr[0]
        if arr[0] not in clss.keys():
            clss[arr[0]] = [(arr[1], arr[2])]
        else:
            clss[arr[0]].append((arr[1], arr[2]))
    f.close()
    return dr, clss


def get_average_pics_for_each_class():
    dr, clss = get_driver_data()
    for elem in clss.keys():
        print('Start', elem, '...')
        image = []
        count = 0
        perfolder = {}
        for folder, im in clss[elem]:
            path = os.path.join('..', 'input', 'train', folder, im)
            if len(image) == 0:
                image = cv2.imread(path)
                image = np.array(image).astype(np.float32)
            else:
                image1 = cv2.imread(path)
                image += image1
                if folder not in perfolder:
                    perfolder[folder] = image1
            count += 1

        image1 = image/(count+1)
        image1 = np.array(image1).astype(np.uint8)
        out_path = elem + '_avg.png'
        print('Printing avg')
        cv2.imwrite(out_path, image1)
        for i in perfolder:
            print('Printing',i)
            diff = perfolder[i] - image1
            mini = np.min(diff)
            maxi = np.max(diff)
            diff = (diff-mini)*256.0/(maxi-mini)
            cv2.imwrite(i + out_path, diff)
            
        break # do only for the first one

get_average_pics_for_each_class()