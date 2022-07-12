# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import sys
import os
import json
import argparse
from pandas.io.json import json_normalize
import urllib
import time
from concurrent.futures import ThreadPoolExecutor

def parse_json(filename):
    try:
        json_file = json.load(open(filename, 'r'))
    except:
        raise IOError('Failed to open file: {}'.format(filename))

    images = json_normalize(json_file['images'])
    images['url'] = images['url'].apply(lambda x: x[0])
    if 'annotations' in json_file:
        annotations = json_normalize(json_file['annotations'])
        data = images.merge(annotations, on = u'image_id')
        return data
    else:
        return images
    
def download_image(image_id, url, path):
    if url.split('.')[-1] not in ['jpg', 'png']:
        extension = 'jpg'
    else:
        extension = url.split('.')[-1]

    name = os.path.join(path, "{}.{}".format(image_id, extension))
    if not os.path.isfile(name):
        for i in range(100):
            try:
                urllib.request.urlretrieve(url, name)
                break;
            except:
                time.sleep(0.1)

def download(json_filename, path):
    data = parse_json(json_filename)
    if not os.path.exists(path):
        os.makedirs(path)

    if 'label_id' in data.columns:
        data.to_csv('{}.csv'.format(path), columns = ['image_id', 'label_id'], index = False)
    
    with ThreadPoolExecutor(max_workers = 128) as executor:
        for index, row in data.iterrows():
            executor.submit(download_image, row['image_id'], row['url'], path)
def main():
    parser = argparse.ArgumentParser(description = "Script for downloads images")
    parser.add_argument('--train', default = None, help = 'Path to json file with train images')
    parser.add_argument('--test', default = None, help = 'Path to json file with test images')
    args = parser.parse_args()
    
    if not os.path.exists('../input"'):
        os.makedirs('../input"')


    if args.train is not None:
        print("Downloading train dataset...")
        try:
            download(json_filename = args.train, path = './input/train')
        except Exception as ex:
            print(ex)

    if args.test is not None:
        print("Downloading test dataset...")
        try:
            download(json_filename = args.test, path = './input/test')
        except Exception as ex:
            print(ex)

if __name__ == "__main__":
    main()
