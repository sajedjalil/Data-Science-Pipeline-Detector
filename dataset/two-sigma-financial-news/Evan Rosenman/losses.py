# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

def computeSigmaScore(preds, r, u, d):
    x_t_i = preds * r * u
    data = {'day' : d, 'x_t_i' : x_t_i}
    df = pd.DataFrame(data)
    x_t = df.groupby('day').sum().values.flatten()
    mean = np.mean(x_t)
    std = np.std(x_t)
    score_valid = mean / std
    return(score_valid)
    
def computeCrossEntropyLoss(probs, r, eps = 1e-12):
    labels = (r >= 0).astype(int)
    probs_clipped = np.clip(probs, eps, 1.0-eps)
    return(np.mean(labels*np.log(probs_clipped) + (1-labels)*np.log(1-probs_clipped)))