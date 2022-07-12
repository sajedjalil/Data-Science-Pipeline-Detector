#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 07:54:40 2018
@author: Gunnvant
"""
"""
Usage:
python script.py path_to_test/train_file dest_file_location
python translate.py /home/ubuntu/train.csv.zip train.csv
"""

import pandas as pd
import sys
import textblob
from tqdm import tqdm,tqdm_pandas

def read(x):
    return pd.read_csv(x)
def desc_missing(x):
    '''
    Takes data frame as input, then searches and fills missing description with недостающий
    '''
    if x['description'].isnull().sum()>0:
        print("Description column has missing values, filling up with недостающий")
        x['description'].fillna("недостающий",inplace=True)
        return x
    else:
        return x
def translate(x):
    try:
        return textblob.TextBlob(x).translate(to="en")
    except:
        return x
def map_translate(x):
    print("Begining to translate")
    tqdm.pandas(tqdm())
    x['en_desc']=x['description'].progress_map(translate)
    print("Done translating decription")
    print("Begining to translate Title")
    x['en_title']=x['title'].progress_map(translate)
    print("Done translating")
    return x

def exporter(x,dest_path):
    print("Writting to {}".format(dest_path))
    x.to_csv(dest_path,index=False)
    print("Done")


def main():
    file=sys.argv[1]
    dest=sys.argv[2]
    data=read(file)
    data=desc_missing(data)
    data=map_translate(data)
    exporter(data,dest)
''' 
uncomment 
if __name__ == '__main__':
    main() 
'''