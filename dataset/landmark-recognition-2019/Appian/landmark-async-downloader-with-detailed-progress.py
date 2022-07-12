#import gevent.monkey; gevent.monkey.patch_all()

import argparse
import sys
import os
from io import BytesIO
import collections

import pandas as pd
import requests
from PIL import Image
import gevent.pool


def download(record, output_dir, counter):

    print('\r%s' % counter, end='')
    sys.stdout.flush()

    counter['total'] += 1

    # if an image exists, skip
    image_path = os.path.join(output_dir, record.id)
    if os.path.exists(image_path):
        counter['already-exists'] += 1
        return

    if record.url == 'None':
        counter['no-url'] += 1
        return

    try:
        response = requests.get(record.url, timeout=10)
    except requests.RequestException as e:
        counter['requests.%s' % e.__class__.__name__] += 1
        return

    try:
        image = Image.open(BytesIO(response.content))
    except OSError as e:
        counter['image-parse-error'] += 1
        return

    with open(image_path, 'wb') as fp:
        fp.write(response.content)

    counter['saved'] += 1

        
def download_images(input_path, output_dir, pool_size):

    with open(input_path) as fp:
        df = pd.read_csv(input_path)

    skip = 0
    df = df.iloc[skip:]

    os.makedirs(output_dir, exist_ok=True)
    counter = collections.Counter()
    _download = lambda x: download(x, output_dir, counter)

    print('Start downloading %d images...' % len(df))

    pool = gevent.pool.Pool(pool_size)
    for record in df.itertuples():
        pool.spawn(_download, record)
    pool.join()


def main():
    
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('input')
    arg('output_dir')
    arg('--pool-size', type=int, default=10)
    args = parser.parse_args()

    download_images(args.input, args.output_dir, args.pool_size)


if __name__ == '__main__':

    # usage
    # Please uncomment the 1st line to enable monkey patching (it makes requests async)
    # python downloader.py $input $output_dir

    # to download test images
    # python downloader.py ../input/test.csv ../input/test_images

    # increase pool size for faster download 
    # python downloader.py ../input/train.csv ../input/train_images --pool-size 20

    #main()
    pass