# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path
from fastai import *
from fastai.vision import *
import torch
from torch import nn
from torch.utils import *
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

file_paths = Path('../input')
train = pd.read_csv('../input/train.csv')
tfms = get_transforms(max_rotate=10.0, 
                        flip_vert=True, 
                        max_zoom=1.1, 
                        do_flip=True, 
                        max_lighting=0.4, max_warp=0.2)
train_x = (ImageList.from_df(train, path=file_paths/'train', folder='train')
            .split_by_rand_pct(0.01)
            .label_from_df()
            .databunch(path='.', 
                        bs=32,
                        device=torch.device('cuda'))
            
            
        )

learn = cnn_learner(train_x, models.densenet161, metrics=[accuracy])

test_files = pd.read_csv("../input/sample_submission.csv")
test_x = ImageList.from_df(test_files, path=file_paths/'test', folder='test')
train_x = train_x.add_test(test_x)

lr = 1e-02
learn.fit_one_cycle(5, slice(lr))


preds,na = learn.get_preds(ds_type=DatasetType.Test)
test_files.has_cactus = preds.numpy()[:, 0]
test_files.to_csv('submission.csv', index=False)