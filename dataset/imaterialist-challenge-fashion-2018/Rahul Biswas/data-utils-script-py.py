# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import io
import os
import sys
import json
import urllib3
import multiprocessing

from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from optparse import OptionParser

parser = OptionParser()
parser.add_option("-f", "--inputfile", help="input FILE to read data", metavar="FILE")
parser.add_option("-o", "--outputDir", help="output directory to store files", metavar="FILE")
parser.add_option("-n", "--numberofimages", type="int", default=10000, help="Number of images to download")

(opts, args) = parser.parse_args()


def download_image(fnames_and_urls):
    """
    download image and save its with 90% quality as JPG format
    skip image downloading if image already exists at given path
    :param fnames_and_urls: tuple containing absolute path and url of image

    Usage:
    python script.py --inputfile="./test.json" --outputDir=<path_to_output_dir> --numberofimages=100
    """
    fname, url = fnames_and_urls
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save(fname, format='JPEG', quality=90)


def parse_dataset(_dataset, _outdir, _max=10000):
    """
    parse the dataset to create a list of tuple containing absolute path and url of image
    :param _dataset: dataset to parse
    :param _outdir: output directory where data will be saved
    :param _max: maximum images to download (change to download all dataset)
    :return: list of tuple containing absolute path and url of image
    """
    _fnames_urls = []
    with open(dataset, 'r') as f:
        data = json.load(f)
        for image in data["images"]:
            url = image["url"]
            fname = os.path.join(outdir, "{}.jpg".format(image["imageId"]))
            _fnames_urls.append((fname, url))
    return _fnames_urls[:_max]


if __name__ == '__main__':

    # get args and create output directory
    dataset = opts.inputfile
    outdir = opts.outputDir
    numImages = opts.numberofimages

    if not os.path.exists(os.path.abspath(outdir)):
        os.makedirs(os.path.abspath(outdir))

    # parse json dataset file
    fnames_urls = parse_dataset(dataset, outdir, numImages)

    # download data
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(fnames_urls)) as progress_bar:
        for _ in pool.imap_unordered(download_image, fnames_urls):
            progress_bar.update(1)

    sys.exit(1)