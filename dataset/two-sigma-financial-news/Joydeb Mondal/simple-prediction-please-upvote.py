# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
import os
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()


days = env.get_prediction_days()
eps = 1.8745641033852102e-20

for (market_obs_df, news_obs_df, predictions_template_df) in days:
    market_obs_df['returnsOpenPrevMktres10'] = market_obs_df['returnsOpenPrevMktres10'].fillna(eps)
    predictions_template_df['confidenceValue'] = 1/market_obs_df['returnsOpenPrevMktres10']*eps
    env.predict(predictions_template_df)
    
    
env.write_submission_file()