#!/usr/bin/python

# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

# Downloads images from the Google Landmarks dataset using multiple threads.
# Images that already exist will not be downloaded again, so the script can
# resume a partially completed download. All images will be saved in the JPG
# format with 90% compression quality.

#Derivative from https://www.kaggle.com/xiuchengwang/python-dataset-download
#ADDING SUPPORT PYTHON 3.+ + log files + progressbar + handling interrupt

import os, multiprocessing, urllib.request, csv
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import logging
logging.basicConfig(filename='downloader.log', format="%(asctime)-15s %(levelname)s %(message)s",
                        datefmt="%F %T", level=logging.DEBUG)

file='train' #change after with test
if not os.path.exists('images'):
    os.mkdir('images')
out_dir = 'images/{}'.format(file)
data_file = 'input/Google-Landmarks-Dataset/{}.csv'.format(file)
#download data from : https://www.kaggle.com/c/landmark-recognition-challenge/data
def ParseData(data_file):
    csvfile = open(data_file, 'r')
    csvreader = csv.reader(csvfile)
    key_url_list = [line[:2] for line in csvreader]
    return key_url_list[1:]  # Chop off header


def DownloadImage(key_url):
    (key, url) = key_url
    filename = os.path.join(out_dir, '%s.jpg' % key)

    if os.path.exists(filename):
        logging.warning('Image %s already exists. Skipping download.' % filename)
        return

    try:
        response = urllib.request.urlopen(url)
    except:
        logging.warning('Warning: Could not download image %s from %s' % (key, url))
        return

    try:
        pil_image = Image.open(BytesIO(response.read()))
    except:
        logging.warning('Warning: Failed to parse image %s' % key)
        return

    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        logging.warning('Warning: Failed to convert image %s to RGB' % key)
        return

    try:
        pil_image_rgb.save(filename, format='JPEG', quality=90)
    except:
        logging.warning('Warning: Failed to save image %s' % filename)
        return


def Run():
    return print('COMMENT ME ! Please run this code on your machine not on a kernel and comment this line :)')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    key_url_list = ParseData(data_file)
    pool = multiprocessing.Pool(processes=100)  #Adjust number of process regarding to your machine performance
    try:
        with tqdm(total=len(key_url_list)) as pbar:
            for i, _ in tqdm(enumerate(pool.imap_unordered(DownloadImage, key_url_list))):
                pbar.update()
    except KeyboardInterrupt:
        logging.warning("got Ctrl+C")
    finally:
        pool.terminate()
        pool.join()

if __name__ == '__main__':
        Run()