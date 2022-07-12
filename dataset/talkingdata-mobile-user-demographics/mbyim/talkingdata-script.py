# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
#CSVs to Dataframes to explore
app_events_df = pd.read_csv('../input/app_events.csv')
app_labels_df = pd.read_csv('../input/app_labels.csv')
events_df = pd.read_csv('../input/events.csv')
genderage_test_df = pd.read_csv('../input/gender_age_test.csv')
genderage_train_df = pd.read_csv('../input/gender_age_train.csv')
label_categories_df = pd.read_csv('../input/label_categories.csv')
phone_info_df = pd.read_csv('../input/phone_brand_device_model.csv')
sample_submission_df = pd.read_csv('../input/sample_submission.csv')

#Exploring dfs
print("Sample Submission:\n",sample_submission_df.head())
print("Genderage Train:\n", genderage_train_df.head())
print("Genderage test:\n", genderage_test_df.head())
print("Phone/Device/Model data:\n", phone_info_df.head())
print("Events:\n", events_df.head())
print("app events:\n",app_events_df.tail())

#Split train data into train and crossv (test)
#print("gender train shape:", genderage_train_df.shape)
genderagetrain = genderage_train_df.iloc[0:60000, :]
print('shape:', genderagetrain.shape)
genderagecross = genderage_train_df.iloc[60000:, :]
print('shape:', genderagecross.shape)

#Making Location Feature(s)


#Making event Feature(s)


#Making app Feature(s)
print('# unique apps:', len(app_events_df.app_id.unique()))
appgroup = app_events_df.groupby("app_id")
app_event_count = appgroup.is_installed.aggregate(np.sum).to_frame() #['event_id'].describe()
print(app_event_count.columns)
print("\n\n\n\n")
print(app_event_count.head())
print(app_event_count.shape)
print(app_event_count.is_installed.sort_values(ascending=False))
print("test")
app_event_count.plot(kind='hist')

print('done')

