# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.spatial.distance import cosine
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

test_data = pd.read_csv("../input/test_ver2.csv")
print(test_data.head())

limit_rows = 100
train = pd.read_csv("../input/train_ver2.csv",dtype={"sexo":str,
                                                    "ind_nuevo":str,
                                                    "ult_fec_cli_1t":str,
                                                    "indext":str}, nrows=limit_rows)
print(train.head())

def getScore(history, similarities):
   return sum(history*similarities)/sum(similarities)
   
data_sims = pd.DataFrame(index=train.index,columns=train.columns)
data_sims.ix[:,:1] = train.ix[:,:1]

