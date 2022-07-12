# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
market_train_df.tail()
news_train_df.head()
news_train_df.tail()

days = env.get_prediction_days()

(market_obs_df, news_obs_df, predictions_template_df) = next(days)
market_obs_df.head()
news_obs_df.head()
predictions_template_df.head()

next(days)


import numpy as np
def make_random_predictions(predictions_df):
    predictions_df.confidenceValue = 2.0 * np.random.rand(len(predictions_df)) - 1.0
    
make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)

(market_obs_df, news_obs_df, predictions_template_df) = next(days)
market_obs_df.head()
news_obs_df.head()
predictions_template_df.head()

make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)

(market_obs_df, news_obs_df, predictions_template_df) = next(days)
market_obs_df.head()
news_obs_df.head()
predictions_template_df.head()
make_random_predictions(predictions_template_df)
env.predict(predictions_template_df)

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    make_random_predictions(predictions_template_df)
    env.predict(predictions_template_df)
print('Done!')