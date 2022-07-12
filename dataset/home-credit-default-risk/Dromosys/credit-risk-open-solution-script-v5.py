# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import sys
sys.stdout = open('stdout.log', 'w')
sys.stderr = open('stderr.log', 'w')

# Any results you write to the current directory are saved as output.

#from src.pipeline_manager import PipelineManager

#pipeline_manager = PipelineManager()

from shutil import copyfile
from shutil import rmtree

copyfile('/opt/conda/lib/python3.6/site-packages/src/kaggle.yaml', '/kaggle/working/neptune.yaml')

import warnings
warnings.simplefilter(action = "ignore", category = RuntimeWarning)

from src.pipeline_manager import PipelineManager

pipeline_manager = PipelineManager()

from src.utils import read_params
from deepsense import neptune
ctx = neptune.Context()
params = read_params(ctx, fallback_file='neptune.yaml')
import src.pipeline_config as cfg
cfg.DEV_SAMPLE_SIZE = 200000
dev_mode = True
submit_predictions = True
pipeline_name = 'lightGBM'
model_level = 'first'

#%prun
pipeline_manager.train_evaluate_predict_cv(pipeline_name, model_level, dev_mode, submit_predictions)

copyfile('/kaggle/working/result/lightGBM_test_predictions_rank_mean.csv', '/kaggle/working/lightGBM_test_predictions_rank_mean.csv')
rmtree('/kaggle/working/result/') 