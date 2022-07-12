# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=5)

trains = pd.read_csv("../input/train.csv", chunksize=3000000)
for train in trains:
    train["book_year"] = (np.where(train['srch_ci'].isnull(), train['date_time'].str[:4], train['srch_ci'].str[:4]))
    train["book_month"] = (np.where(train['srch_ci'].isnull(), train['date_time'].str[5:7], train['srch_ci'].str[5:7]))
    train = train.drop(["hotel_cluster", "date_time", "srch_ci", "srch_co", "cnt", "is_booking"], 1)
    train.fillna(0, inplace=True)
    ipca.partial_fit(train)
    new_train = ipca.transform(train)
    
print(ipca.components_)
print(ipca.explained_variance_)
print(ipca.explained_variance_ratio_)

    