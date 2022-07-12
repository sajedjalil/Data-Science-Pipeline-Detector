# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
from sklearn.linear_model import PassiveAggressiveRegressor
import time; start_time = time.time()
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import numpy as np
import pandas as pd
import sqlite3

reg = PassiveAggressiveRegressor(warm_start=True, random_state=123)
chunksize_ = 100000
mc_y_label = []
mc_y_pred = []
id_test = []
y_pred = []

df_train = pd.read_csv('../input/clicks_train.csv')
#promo = pd.read_csv('../input/promoted_content.csv', usecols=['ad_id', 'document_id'])
#docc = pd.read_csv('../input/documents_categories.csv', usecols=['document_id','confidence_level'])
#docs = pd.read_csv('../input/documents_topics.csv', usecols=['document_id','confidence_level'])
#docm = pd.read_csv('../input/documents_meta.csv') #['document_id', 'source_id', 'publisher_id', 'publish_time']
#events = pd.read_csv('../input/events.csv') #['display_id', 'uuid', 'document_id', 'timestamp', 'platform', 'geo_location'] 
#pagev = pd.read_csv('../input/page_views_sample.csv') #['uuid', 'document_id', 'timestamp', 'platform', 'geo_location', 'traffic_source'] 
#doce = pd.read_csv('../input/documents_entities.csv') #['document_id', 'entity_id', 'confidence_level'] 
print(df_train.head())
