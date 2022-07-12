#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
 * @Author: shifaqiang--[14061115@buaa.edu.cn] 
 * @Date: 2018-04-20 16:06:39 
 * @Last Modified by:   shifaqiang 
 * @Last Modified time: 2018-04-20 16:06:39 
 * @Desc:
 
 This python script provides a multi-threaded tool in Windows to help you quickly download the FGVC5 datasets.
 At the same time, I have uploaded the furniture task's dataset with .zip format, but it is missing some of them because of
 the restriction of Chinese mainland's network environment. Specifically, I welcome everyone to help me
 continue to maintain this data set so that others can directly obtain a complete data set. If you have those
 data which are missing, please contact me at 14061115@buaa.edu.cn, thanks.
 
 Here are details of this dataset:
 train :189258 / 194828 (missing 5570 images)
 validation : 6212 / 6400 (missing 188 images)
 test : 12431 / 12800 (missing 369 images)
 And the train_annotations.scv and validation_annotations.csv is the label annotations of all images by csv format.
 Other three json files are origianl file provided kaggle.
  
  this is the BaiduYun link : https://pan.baidu.com/s/1A3w4Obt-JtGe4M0ueOiWIw
  
'''

import os
import pathlib2
import time
import numpy as np
import pandas as pd
import json
import urllib
import tqdm
import threading

########################################################################
class fanshion(object):
    """
    download all image data to given directory tree in parameter(directory)
    """
    #----------------------------------------------------------------------
    def __init__(self, directory):
        """Constructor"""
        self.directory = directory
        # prepare file path
        self.train_dir = self.directory / pathlib2.Path("train")
        self.validation_dir = self.directory / pathlib2.Path("validation")
        self.test_dir = self.directory / pathlib2.Path("test")        
        if not os.path.exists(self.train_dir):
            os.system("md "+str(self.train_dir))
        if not os.path.exists(self.validation_dir):
            os.system("md "+str(self.validation_dir))
        if not os.path.exists(self.test_dir):
            os.system("md "+str(self.test_dir))        
    
    def download(self, images, image_type):
        # for seq, item in tqdm.tqdm(enumerate(images), total=len(images), unit="image"):
        for seq, item in enumerate(images):
            try:
                url = item["url"]
                filename = self.directory / pathlib2.Path(image_type) / pathlib2.Path(str(item["imageId"])+"_"+image_type+".jpg")
                if not os.path.exists(filename):
                    urllib.request.urlretrieve(url, filename)
            except:
                pass        
    def multi_threading_download(self, images, image_type, threading_size=50, step=50):
        '''
        threading_size: 同时运行的线程数量
        step：单个线程下载的图片数量
        images：所有的图片url
        image_type：图片分类文件夹，选 "train","validation","test"
        '''
        start = 0
        end = 0
        total = len(images)
        while end < total:
            start = end
            end = min([end + step, total])
            # 分支线程数多于限制数量时，主线程死循环等待
            while threading.activeCount() > threading_size:
                time.sleep(20)
            newThreading = threading.Thread(target=self.download, args=(images[start:end], image_type))
            newThreading.start()
            # newThreading.join()
            print("the size of current threading is %s" % threading.activeCount())
    def get_train_data(self, v):
        # train data
        # imges (eg. train_id.jpg) to train(folder), id and label to train.csv
        print("Downloading train data...")
        try:
            train_json_filename = self.directory / pathlib2.Path("train.json")
            train_json_file = open(train_json_filename, "r")
            train_json = json.load(train_json_file)
            # get train.csv
            train_annotations = pd.DataFrame(train_json.get("annotations"))
            train_annotations_filename = self.directory / pathlib2.Path("train_annotations.csv")
            train_annotations.to_csv(str(train_annotations_filename), index=False)
            # download all images
            images_url = train_json.get("images")
            self.multi_threading_download(images_url, "train", threading_size=v)
            train_json_file.close()
        except:
            print("No train.json file, please check and try again!")
    def get_validation_data(self, v):
        # validation data
        # imges (eg. validation_id.jpg) to validation(folder), id and label to validation.csv
        print("Downloading validation data...")
        try:
            validation_json_filename = self.directory / pathlib2.Path("validation.json")
            validation_json_file = open(validation_json_filename, "r")
            validation_json = json.load(validation_json_file)
            # get validation.csv
            validation_annotations = pd.DataFrame(validation_json.get("annotations"))
            validation_annotations_filename = self.directory / pathlib2.Path("validation_annotations.csv")
            validation_annotations.to_csv(str(validation_annotations_filename), index=False)
            # download all images
            images_url = validation_json.get("images")
            self.multi_threading_download(images_url, "validation", threading_size=v)
            validation_json_file.close()
        except:
            print("No validation.json file, please check and try again!")
    def get_test_data(self, v):
        # test data
        # imges (eg. test_id.jpg) to test(folder)
        print("Downloading test data...")
        try:
            test_json_filename = self.directory / pathlib2.Path("test.json")
            test_json_file = open(test_json_filename, "r")
            test_json = json.load(test_json_file)
            # download all images
            images_url = test_json.get("images")
            self.multi_threading_download(images_url, "test", threading_size=v)
            test_json_file.close()
        except:
            print("No test.json file, please check and try again!")   


########################################################################
class furniture(object):
    """
    download all image data to given directory tree in parameter(directory)
    """
    #----------------------------------------------------------------------
    def __init__(self, directory):
        """Constructor"""
        self.directory = directory
        # prepare file path
        self.train_dir = self.directory / pathlib2.Path("train")
        self.validation_dir = self.directory / pathlib2.Path("validation")
        self.test_dir = self.directory / pathlib2.Path("test")        
        if not os.path.exists(self.train_dir):
            os.system("md "+str(self.train_dir))
        if not os.path.exists(self.validation_dir):
            os.system("md "+str(self.validation_dir))
        if not os.path.exists(self.test_dir):
            os.system("md "+str(self.test_dir))        
    
    def download(self, images, image_type):
        # for seq, item in tqdm.tqdm(enumerate(images), total=len(images), unit="image"):
        for seq, item in enumerate(images):
            try:
                url = item["url"][0]
                filename = self.directory / pathlib2.Path(image_type) / pathlib2.Path(str(item["image_id"])+"_"+image_type+".jpg")
                if not os.path.exists(filename):
                    urllib.request.urlretrieve(url, filename)
            except:
                print(item)
                pass 
    def multi_threading_download(self, images, image_type, threading_size=50, step=100):
        '''
        threading_size: 同时运行的线程数量
        step：单个线程下载的图片数量
        images：所有的图片url
        image_type：图片分类文件夹，选 "train","validation","test"
        '''
        start = 0
        end = 0
        total = len(images)
        while end < total:
            start = end
            end = min([end + step, total])
            # 分支线程数多于限制数量时，主线程死循环等待
            while threading.activeCount() > threading_size:
                time.sleep(20)
            newThreading = threading.Thread(target=self.download, args=(images[start:end], image_type))
            newThreading.start()
            # newThreading.join()
            # print("the size of current threading is %s" % threading.activeCount())
    
    def get_train_data(self, v):
        # train data
        # imges (eg. train_id.jpg) to train(folder), id and label to train.csv
        print("Downloading train data...")
        try:
            train_json_filename = self.directory / pathlib2.Path("train.json")
            train_json_file = open(train_json_filename, "r")
            train_json = json.load(train_json_file)
            # get train.csv
            train_annotations = pd.DataFrame(train_json.get("annotations"))
            train_annotations_filename = self.directory / pathlib2.Path("train_annotations.csv")
            train_annotations.to_csv(str(train_annotations_filename), index=False)
            # download all images
            images_url = train_json.get("images")
            self.multi_threading_download(images_url, "train", threading_size=v)
            train_json_file.close()
        except:
            print("No train.json file, please check and try again!")
    def get_validation_data(self, v):
        # validation data
        # imges (eg. validation_id.jpg) to validation(folder), id and label to validation.csv
        print("Downloading validation data...")
        try:
            validation_json_filename = self.directory / pathlib2.Path("validation.json")
            validation_json_file = open(validation_json_filename, "r")
            validation_json = json.load(validation_json_file)
            # get validation.csv
            validation_annotations = pd.DataFrame(validation_json.get("annotations"))
            validation_annotations_filename = self.directory / pathlib2.Path("validation_annotations.csv")
            validation_annotations.to_csv(str(validation_annotations_filename), index=False)
            # download all images
            images_url = validation_json.get("images")
            self.multi_threading_download(images_url, "validation", threading_size=v)
            validation_json_file.close()
        except:
            print("No validation.json file, please check and try again!")
    def get_test_data(self, v):
        # test data
        # imges (eg. test_id.jpg) to test(folder)
        print("Downloading test data...")
        try:
            test_json_filename = self.directory / pathlib2.Path("test.json")
            test_json_file = open(test_json_filename, "r")
            test_json = json.load(test_json_file)
            # download all images
            images_url = test_json.get("images")
            self.multi_threading_download(images_url, "test", threading_size=v, step=20)
            test_json_file.close()
        except:
            print("No test.json file, please check and try again!")        

if __name__ == "__main__":
    
    # base_dir = pathlib2.Path("D:/kaggle/FGVC5/data")
    base_dir = pathlib2.Path(os.getcwd())
    fashion_dir = base_dir / pathlib2.Path("imaterialist-challenge-fashion-2018")
    furniture_dir = base_dir / pathlib2.Path("imaterialist-challenge-furniture-2018")
    print("fetching fashion data...")
    v = 200
    #myfanshion = fanshion(fashion_dir)
    #myfanshion.get_validation_data(v)
    #myfanshion.get_test_data(v)
    #myfanshion.get_train_data(v)        
    print("fetching furniture data...")
    #myfurniture = furniture(furniture_dir)
    #myfurniture.get_train_data(v)
    #myfurniture.get_validation_data(v)
    #myfurniture.get_test_data(v)

    """
    1. pleas uncomment line 250-253 for fanshion task dataset
    2. pleas uncomment line 255-258 for furniture task dataset
    3. if you want to run it in linux, pleas change "md" to "mkdir" in line 50-55 and 152-157
    """
    print("That's all, thanks!")