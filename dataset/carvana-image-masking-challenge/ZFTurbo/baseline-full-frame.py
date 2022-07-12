# -*- coding: utf-8 -*-
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import pandas as pd

if __name__ == '__main__':
    t = pd.read_csv('../input/sample_submission.csv')
    t['rle_mask'] = '1 2455039'
    t.to_csv('submit.csv', index=False)
