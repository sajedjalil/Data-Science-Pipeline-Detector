"""
Preprocess image data for Dogs vs. Cats competition
https://www.kaggle.com/gauss256

DESCRIPTION

Most solutions for this competition will need to deal with the fact that the
training and test images are of varying sizes and aspect ratios, and were taken
under a variety of lighting conditions. It is typical in cases like this to
preprocess the images to make them more uniform.

By default this script will normalize the image luminance and resize to a
square image of side length 224. The resizing preserves the aspect ratio and
adds gray bars as necessary to make them square. The resulting images are
stored in a folder named data224.

The location of the input and output files, and the size of the images, can be
controlled through the processing parameters below.

INSTALLATION

This script has been tested on Ubuntu 14.04 using the Anaconda distribution for
Python 3.5. The only additional requirement is for the pillow library for image
manipulation which can be installed via:
    conda install pillow
"""
import glob
from multiprocessing import Process
import os
import re

import numpy as np
import PIL
from PIL import Image

# Processing parameters
SIZE = 224  # for ImageNet models compatibility
TEST_DIR = 'data/test/'
TRAIN_DIR = 'data/train/'
BASE_DIR = '.'


def natural_key(string_):
    """
    Define sort key that is integer-aware
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]


def norm_image(img):
    """
    Normalize PIL image

    Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
    """
    img_y, img_b, img_r = img.convert('YCbCr').split()

    img_y_np = np.asarray(img_y).astype(float)

    img_y_np /= 255
    img_y_np -= img_y_np.mean()
    img_y_np /= img_y_np.std()
    scale = np.max([np.abs(np.percentile(img_y_np, 1.0)),
                    np.abs(np.percentile(img_y_np, 99.0))])
    img_y_np = img_y_np / scale
    img_y_np = np.clip(img_y_np, -1.0, 1.0)
    img_y_np = (img_y_np + 1.0) / 2.0

    img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)

    img_y = Image.fromarray(img_y_np)

    img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

    img_nrm = img_ybr.convert('RGB')

    return img_nrm

def resize_image(img, size):
    """
    Resize PIL image

    Resizes image to be square with sidelength size. Pads with black if needed.
    """
    # Resize
    n_x, n_y = img.size
    if n_y > n_x:
        n_y_new = size
        n_x_new = int(size * n_x / n_y + 0.5)
    else:
        n_x_new = size
        n_y_new = int(size * n_y / n_x + 0.5)

    img_res = img.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

    # Pad the borders to create a square image
    img_pad = Image.new('RGB', (size, size), (128, 128, 128))
    ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
    img_pad.paste(img_res, ulc)

    return img_pad


def prep_images(paths, out_dir):
    """
    Preprocess images

    Reads images in paths, and writes to out_dir

    """
    for count, path in enumerate(paths):
        if count % 100 == 0:
            print(path)
        img = Image.open(path)
        img_nrm = norm_image(img)
        img_res = resize_image(img_nrm, SIZE)
        basename = os.path.basename(path)
        path_out = os.path.join(out_dir, basename)
        img_res.save(path_out)


def main():
    """Main program for running from command line"""

    # Get the paths to all the image files
    train_cats = sorted(glob.glob(os.path.join(TRAIN_DIR, 'cat*.jpg')), key=natural_key)
    train_dogs = sorted(glob.glob(os.path.join(TRAIN_DIR, 'dog*.jpg')), key=natural_key)

    test_all = sorted(glob.glob(os.path.join(TEST_DIR, '*.jpg')), key=natural_key)

    # Make the output directories
    base_out = os.path.join(BASE_DIR, 'data{}'.format(SIZE))
    train_dir_out = os.path.join(base_out, 'train')
    test_dir_out = os.path.join(base_out, 'test')
    os.makedirs(train_dir_out, exist_ok=True)
    os.makedirs(test_dir_out, exist_ok=True)

    # Preprocess the training files
    procs = dict()
    procs[1] = Process(target=prep_images, args=(train_cats, train_dir_out, ))
    procs[1].start()
    procs[2] = Process(target=prep_images, args=(train_dogs, train_dir_out, ))
    procs[2].start()
    procs[3] = Process(target=prep_images, args=(test_all, test_dir_out, ))
    procs[3].start()

    procs[1].join()
    procs[2].join()
    procs[3].join()


if __name__ == '__main__':
    main()
