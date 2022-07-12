from __future__ import print_function
import pandas as pd
import argparse
from glob import glob
from sys import argv
from itertools import chain
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.externals import joblib

def parse_input():
    """
    For command line use.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data",
            help="path to directory containing sample images (*.tif only)")
    parser.add_argument("labels",
            help="path to csv file containing sample labels")
    parser.add_argument("outputFile",
            help="filename of resulting csv file")
    parser.add_argument("--binarizer",
            help="a pickled MultiLabelBinarizer object")
    args = parser.parse_args()
    return args

def format_labels(data_path, labels_path):
    """
    Returns a filtered pandas DataFrame of labels such that only
    images available in the data_path directory are present.
    """
    labels = pd.read_csv(labels_path)
    images = glob("{}/*.tif".format(data_path))
    image_names = set([s[len(data_path):-4] for s in images])
    labels = labels[[l in image_names for l in labels['image_name']]]
    return labels

def binarize_labels(labels, binarizer):
    """
    Returns a MultiLabelBinarizer object (for later use on a test set),
    a 2-d ndarray of binarized labels, and ordered list of column names.
    """
    split_tags = labels["tags"].map(lambda t : t.split(" "))
    classes = sorted(set(chain(*split_tags)))
    if not binarizer:
        binarizer = MultiLabelBinarizer(classes=classes).fit(split_tags)
    binarized_labels = binarizer.transform(split_tags)
    return binarizer, binarized_labels, classes

def get_data(data_path, labels_path, binarizer=None):
    """
    Main functionality of this script. Returns a pandas DataFrame with columns
    ['image_name', 'agriculture', 'artisinal', ...] as a binary representation
    of the sample labels, as well as the MultiLabelBinarizer object used to
    generate the binary rows for each sample.
    """
    labels = format_labels(data_path, labels_path)
    binarizer, binarized_labels, col_names = binarize_labels(labels, binarizer)
    labels = labels.drop("tags", axis=1)
    labels = labels.reset_index(drop=True)
    labels = labels.join(pd.DataFrame(binarized_labels, columns=col_names))
    return labels, binarizer

if __name__ == "__main__":
    """
    If you invoke this script from the command line, it will write the
    resulting pandas DataFrame as a csv file and pickle the binarizer used,
    both to the current directory.
    """
    try:
        args = parse_input()
    except:
        # you must be using this within Kaggle Kernels.
        # we will binarize the training set.
        class Bunch:
            __init__ = lambda self, **kw: setattr(self, '__dict__', kw)
        args = Bunch()
        args.data = "../input/train-tif-v2/"
        args.labels = "../input/train_v2.csv"
        args.outputFile = "train_v2_bin.csv"
        args.binarizer = None
    if args.binarizer:
        args.binarizer = joblib.load(args.binarizer)
    labels, binarizer = get_data(args.data, args.labels, args.binarizer)
    labels.to_csv(args.outputFile, index=False)
    joblib.dump(binarizer, "binarizer.pkl")
