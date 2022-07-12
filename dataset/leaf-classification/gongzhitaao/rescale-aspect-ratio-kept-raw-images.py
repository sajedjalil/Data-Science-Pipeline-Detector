import os

import numpy as np
import pandas as pd

from PIL import Image


def prepare_data(nrow=30, ncol=30):
    print('Preparing training data')

    df_train = pd.read_csv('../input/train.csv')
    df_test = pd.read_csv('../input/test.csv')

    label_train = df_train[['id', 'species']]
    label_test = df_test[['id']]

    # # find out the max nrow and ncol
    # nrow, ncol = 0, 0
    # for imgfile in os.listdir('../input/images'):
    #     img = imread(os.path.join('../input/images', imgfile))
    #     if img.shape[0] > nrow:
    #         nrow = img.shape[0]
    #     if img.shape[1] > ncol:
    #         ncol = img.shape[1]
    # print(nrow, ncol)
    # maxrow, maxcol = 1100, 1710

    def rescale(fpath, nrow, ncol):
        print('Scaling ' + fpath)
        im = Image.open(fpath)

        # down scale image
        im.thumbnail((nrow, ncol), Image.ANTIALIAS)

        # pad image to desired size (nrow,ncol)
        row_offset = int(max((nrow-im.size[0]) / 2, 0))
        col_offset = int(max((ncol-im.size[1]) / 2, 0))
        thumb = Image.new(mode='L', size=(nrow,ncol))
        thumb.paste(im, (row_offset, col_offset))

        # image are column-major while ndarray are row-major
        return np.asarray(thumb).T.flatten()

    print('Generate training data')
    X_train = np.zeros((label_train.shape[0], nrow*ncol),
                       dtype=np.uint8)
    for i, img in enumerate(label_train['id']):
        X_train[i] = rescale('../input/images/{0}.jpg'.format(img),
                             nrow, ncol)
    df_train = pd.concat([label_train, pd.DataFrame(X_train)], axis=1)
    # save training data
    # df_train.to_csv('../input/train-pixel.csv', index=False)
    print(df_train.shape)

    print('Generating test data')
    X_test = np.zeros((label_test.shape[0], nrow*ncol))
    for i, img in enumerate(label_test['id']):
        X_test[i] = rescale('../input/images/{0}.jpg'.format(img),
                            nrow, ncol)
    df_test = pd.concat([label_test, pd.DataFrame(X_test)], axis=1)
    # save testing data
    # df_test.to_csv('../input/test-pixel.csv', index=False)
    print(df_test.shape)

# need to run once only
prepare_data()