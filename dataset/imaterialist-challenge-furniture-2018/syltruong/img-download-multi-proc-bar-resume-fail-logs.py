#!/usr/bin/python3.6
# Note to Kagglers: This script will not run directly in Kaggle kernels. You
# need to download it and run it on your local machine.

"""
Downloads all images
- Usage:
python download_images.py path_to_json output_dir

Features
- Already downloaded images will not be downloaded again. The script can thus be interrupted and re-ran.
- Takes into account that url field is actually an array. Will test all urls in the array.
- Multiprocessing
- 30-second timeout
- Saves all images in jpg format, converts to jpg only if necessary
- logs failures to current directory in download_from_{filename.json}.log
"""

import requests
import sys
from PIL import Image
from io import BytesIO
import json
import os
import argparse
import multiprocessing as mp
import logging
from tqdm import tqdm

LOGGER = logging.getLogger(__file__)

def download_single(data):
    """
    Downloads a single image
    data is (url, image_id, target_path)
    """
    url = data[0]
    image_id = data[1]
    target_path = data[2]

    if os.path.exists(target_path):
        return

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except:
        LOGGER.warning('Failed to fetch url %s (id=%d)', url, image_id)
        return

    try:
        content = response.content
        image = Image.open(BytesIO(content))
    except:
        LOGGER.warning('Failed to capture image at url %s (id=%d)', url, image_id)
        return

    if not image.format == 'JPEG':
        try:
            image = image.convert('RGB')
        except:
            logging.warning('Failed to convert RGB, %s (id=%d)', url, image_id)
            return

    try:
        image.save(target_path, format='JPEG', quality=100)
    except:
        LOGGER.warning('Failed to save url %s (id=%d)', url, image_id)
        return

    return

def download_batch(input_json_file, target_dir):
    """
    Core function
    """
    os.makedirs(target_dir, exist_ok=True)

    with open(input_json_file) as json_f:
        input_info = json.load(json_f)

    url_id_path = [\
                    (\
                        url, \
                        image['image_id'], \
                        os.path.join(target_dir, str(image['image_id']) + '.jpg')\
                    )\
                    for image in input_info['images'] for url in image['url']\
                  ]

    total = len([elt[2] for elt in url_id_path if not os.path.exists(elt[2])])

    LOGGER.info('%d images already downloaded. %d to download', len(url_id_path) - total, total)

    pool = mp.Pool(mp.cpu_count())

    with tqdm(total=total) as t:
        for _ in pool.imap_unordered(download_single, url_id_path):
            t.update(1)

def parser():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser(description='Download from urls, iMaterialist Challenge')
    parser.add_argument('input_json_file',
                   help='path to data json file')
    parser.add_argument('output_dir',
                   help='destination_dir')

    args = parser.parse_args()

    return args

def main():
    args = parser()

    handler = logging.FileHandler('download_from_%s.log' % os.path.basename(args.input_json_file))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

    download_batch(args.input_json_file, args.output_dir)

if __name__ == '__main__':

    # this is just to escape Kaggle platform's run check
    # the correct usage is:
    # python download_images.py path_to_json output_dir
    if len(sys.argv) == 1:
        sys.exit(0)

    main()
