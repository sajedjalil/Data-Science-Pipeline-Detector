# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Super Robust Prediction algorithm
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0



# Any results you write to the current directory are saved as output.

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

# You can only iterate through a result from `get_prediction_days()` once
# so be careful not to lose it once you start iterating.
days = env.get_prediction_days()


(market_train_df, news_train_df) = env.get_training_data()

(market_obs_df, news_obs_df, predictions_template_df) = next(days)

make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)


for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions(predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')

env.write_submission_file()
print([filename for filename in os.listdir('.') if '.csv' in filename])