#!/usr/bin/python3.6
# This script is based on https://www.kaggle.com/syltruong/img-download-multi-proc-bar-resume-fail-logs
# with additional support to enter percentage as parameter
#
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
from collections import defaultdict
from math import ceil

LOGGER = logging.getLogger(__file__)
IMAGE_ID = 'image_id'
LABEL_ID = 'label_id'
URL = 'url'
PATH = 'path'


def download_single(data):
    """
    Downloads a single image
    data is (url, image_id, target_path)
    """
    url = data[URL]
    image_id = data[IMAGE_ID]
    target_path = data[PATH]

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


def filter_on_percentage(data, percentage):
    data_points = len(data) * percentage
    res = defaultdict(list)
    for data in data:
        res[data[LABEL_ID]].append(data)
    number_of_labels = len(res.keys())
    per_label_data_to_download = ceil(data_points / number_of_labels)

    data_to_be_downloaded = []
    for key in res.keys():
        data_to_be_downloaded += res[key][:per_label_data_to_download]

    return data_to_be_downloaded


def download_batch(input_json_file, target_dir, percentage):
    """
    Core function
    """
    os.makedirs(target_dir, exist_ok=True)

    with open(input_json_file) as json_f:
        input_info = json.load(json_f)

    annotations = {}
    for annotation in input_info['annotations']:
        annotations[annotation[IMAGE_ID]] = annotation[LABEL_ID]

    image_data = []
    for image in input_info['images']:
        for url in image[URL]:
            image_data.append(
                {
                    URL: url,
                    IMAGE_ID: image[IMAGE_ID],
                    PATH: os.path.join(target_dir, str(image[IMAGE_ID]) + '.jpg'),
                    LABEL_ID: annotations[image[IMAGE_ID]]
                }
            )

    data_to_be_downloaded = [elt for elt in image_data if not os.path.exists(elt[PATH])]
    data_to_be_downloaded = filter_on_percentage(data_to_be_downloaded, percentage)
    total_required_download = len(data_to_be_downloaded)

    LOGGER.info('%d images already downloaded. %d to download', len(image_data) - total_required_download,
                total_required_download)

    pool = mp.Pool(mp.cpu_count())

    with tqdm(total=total_required_download) as t:
        for _ in pool.imap_unordered(download_single, data_to_be_downloaded):
            t.update(1)
    print('done!!!!')


def parser():
    """
    Argument parser
    """
    parser = argparse.ArgumentParser(description='Download from urls, iMaterialist Challenge')
    parser.add_argument('input_json_file',
                        help='path to data json file')
    parser.add_argument('output_dir',
                        help='destination_dir')
    parser.add_argument('--percentage', dest='percentage',
                        help='percentage of data to be downloaded', type=float, default=1)

    args = parser.parse_args()

    return args


def main():
    args = parser()

    handler = logging.FileHandler('download_from_%s.log' % os.path.basename(args.input_json_file))
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)

    download_batch(args.input_json_file, args.output_dir, args.percentage)


if __name__ == '__main__':

    # this is just to escape Kaggle platform's run check
    # the correct usage is:
    # python download_images.py path_to_json output_dir
    if len(sys.argv) == 1:
        sys.exit(0)

    main()
