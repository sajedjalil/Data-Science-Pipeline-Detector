# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#file_list
# app_events.csv
# app_labels.csv
# events.csv
# gender_age_test.csv
# gender_age_train.csv
# label_categories.csv
# phone_brand_device_model.csv
# Any results you write to the current directory are saved as output.
app_labels=pd.read_csv('../input/app_labels.csv')
print(app_labels.head())
print(app_labels.shape)
print(app_labels.app_id.nunique())
print(app_labels.label_id.nunique())