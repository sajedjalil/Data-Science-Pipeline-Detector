# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import datetime
import lightgbm as lgb
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
stop = set(stopwords.words('english'))


import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

#Set 
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market, news_train_df) = env.get_training_data()
market
#How do we deal with the missing value?

#Get market prediction days 
top_10_idx = np.argsort(market["returnsClosePrevRaw1"])[-10:]
days = env.get_prediction_days()
(market, news_obs_df, predictions_template_df) = next(days)
market.head()

predictions_template_df.head()

#Analyze news data 
news_text=news_obs_df["headline"]

frequency_words = {}
for data in news_obs_df["headline"]:
    tokens = nltk.wordpunct_tokenize(data)
    for token in tokens:
        if token in frequency_words:
            count = frequency_words[token]
            count = count + 1
            frequency_words[token] = count
        else:
            frequency_words[token] = 1

# Generate Wordcloud
frequency_words
wordcloud = WordCloud()
wordcloud.generate_from_frequencies(frequency_words)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

# Convert the dict to a dataframe
freq = pd.DataFrame.from_dict(frequency_words, orient = 'index')

# Let us sort them in descinding order
freq.sort_values(by = 0, ascending=False).head(20)

## Remove the stop words 
from nltk.corpus import stopwords
frequency_words_wo_stop = {}

for data in news_obs_df["headline"]:
    tokens = nltk.wordpunct_tokenize(data)
    for token in tokens:
        if token.lower() not in stop:
            if token in frequency_words_wo_stop:
                count = frequency_words_wo_stop[token]
                count = count + 1
                frequency_words_wo_stop[token] = count
            else:
                frequency_words_wo_stop[token] = 1
wordcloud.generate_from_frequencies(frequency_words_wo_stop)
plt.figure(figsize=(14,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

type(stop)
stop.update('<', '>', '<>', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}','/','-','.','&')
frequency_words_wo_stop = {}

def generate_word_frequency(row):
    data = row["headline"]
    tokens = nltk.wordpunct_tokenize(data)
    token_list = []
    for token in tokens:
        if token.lower() not in stop:
            token_list.append(token.lower())
            if token.lower() in frequency_words_wo_stop:
                count = frequency_words_wo_stop[token.lower()]
                count = count + 1
                frequency_words_wo_stop[token.lower()] = count
            else:
                frequency_words_wo_stop[token.lower()] = 1
    
    return ','.join(token_list)

news_obs_df["text"] = news_obs_df.apply(generate_word_frequency,axis=1)
news_obs_df.head()
wordcloud.generate_from_frequencies(frequency_words_wo_stop)
plt.figure(figsize=(24,10))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

##Topic modeling 
#lda modeling 
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
cv = CountVectorizer(min_df = 2,
                     max_features = 100000,
                     analyzer = "word",
                     ngram_range = (1, 2),
                     stop_words = "english")


count_vectors = cv.fit_transform(news_obs_df["headline"])


lda_model = LatentDirichletAllocation(n_components = 20, 
                                      # we choose a small n_components for time convenient
                                      learning_method = "online",
                                      max_iter = 20,
                                      random_state = 32)

news_topics = lda_model.fit_transform(count_vectors)

n_top_words = 10
topic_summaries = []
topic_word = lda_model.components_
vocab = cv.get_feature_names()

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(" ".join(topic_words))
    print("Topic {}: {}".format(i, " | ".join(topic_words)))

from sklearn.manifold import TSNE
tsne_model = TSNE(n_components = 2, verbose = 1, random_state = 32, n_iter = 500)
tsne_lda = tsne_model.fit_transform(news_topics)

news_topics = np.matrix(news_topics)
doc_topics = news_topics/news_topics.sum(axis = 1)

lda_keys = []
for i, tweet in enumerate(news_obs_df["headline"]):
    lda_keys += [doc_topics[i].argmax()]
    
tsne_lda_df = pd.DataFrame(tsne_lda, columns = ["x", "y"])
tsne_lda_df["headline"] = news_obs_df["headline"].values
tsne_lda_df["asset_name"] = news_obs_df["assetName"].values
tsne_lda_df["assetCodes"] = news_obs_df["assetCodes"].values
tsne_lda_df["relevance"] = news_obs_df["relevance"].values
tsne_lda_df["sentimentNegative"]= news_obs_df["sentimentNegative"]
tsne_lda_df["sentimentNeutral"]=news_obs_df["sentimentNeutral"]
tsne_lda_df["sentimentPositive"]=news_obs_df["sentimentPositive"]
tsne_lda_df["topics"] = lda_keys
tsne_lda_df["topics"] = tsne_lda_df["topics"].map(int)

#Data Exploration
news_obs_df.dtypes
news_obs_df.isna().sum()
tsne_lda_df.dtypes
tsne_lda_df["topics"].nunique()
tsne_lda_df.isna().sum()
tsne_lda_df['assetCodes'].describe()
news_obs_df['assetCodes'].describe()
market_train_df['assetCode'].describe()

###############
#Market Data###
###############
# check if there are abnormal prices changes in a single day
market['price_diff'] = market['close'] - market['open']
grouped = market.groupby('time').agg({'price_diff':['std','min','max']}).reset_index()
grouped.sort_values(('price_diff', 'std'), ascending = False)[:10]

market['close/open'] = market['close'] / market['open']
grouped_2 = market.groupby('time').agg({'close/open':['std','min','max']}).reset_index()
grouped_2.sort_values(('close/open', 'std'), ascending = False)[:10]


for i,row in market.loc[market['close/open'] >= 1.5].iterrows():
    market.iloc[i, 4] = np.random.uniform(1.01,1.20) * row['open']

for i,row in market.loc[market['close/open'] <= 0.5].iterrows():
    market.iloc[i, 5] = np.random.uniform(1.01,1.20) * row['close']
    
market['price_diff'] = market['close'] - market['open']  

# check missing data and fill missing data with preceding value


plt.bar(market.columns, market.isna().sum(),bottom=None, align='center')
plt.title('Total Missing Value by Column')
plt.xlabel('Column Variables')
plt.ylabel('Count')
plt.xticks(rotation = 60, horizontalalignment="right")
plt.show()


# sort data by 'assetCode' and then by 'time'
market = market.sort_values(by = ['assetCode','time'], ascending=[True, True])
market.head()

# fill missing value using the next valid observation
def _fill_na(dataset):
    for i in dataset.columns:
        if dataset[i].dtype == 'float64':
            dataset[i] = dataset[i].fillna('bfill')
        else:
            pass
    return dataset

market = _fill_na(market)

market['price_diff'] = market['close'] - market['open']    
market['close/open'] = market['close'] / market['open']

market.isna().sum()

# build heatmap to visualize correlation
temp = market.drop(['time','assetCode','assetName', 'close/open', 'price_diff'], axis = 1)
temp.info()

def to_numeric(dataset):
    for i in dataset.columns:
        if dataset[i].dtype == 'object':
            dataset[i] = dataset[i].apply(pd.to_numeric, errors='coerce')
        else:
            pass
    return dataset

temp = to_numeric(temp)
temp['time'], temp['assetCode'] = market['time'], market['assetCode']
corr = temp.corr()

fig = sns.heatmap(corr)
fig.set_yticklabels(fig.get_yticklabels(), rotation=0)
fig.set_xticklabels(fig.get_yticklabels(), rotation=90)

# normalizing data
market_temp = temp
market_temp.info()

def normalization(dataset):
    for i in dataset.columns:
        if dataset[i].dtype == 'float64':
            dataset[i] = (dataset[i] - dataset[i].mean()) / (dataset[i].max() - dataset[i].min())
        else:
            pass
    return dataset
    
market_normalized = normalization(market_temp)
market_normalized.describe()
market_normalized.info()

#Combination prepration 
from itertools import chain

def join_market_news(market_normalized, tsne_lda_df):
    # Fix asset codes (str -> list)
    tsne_lda_df['assetCodes'] = tsne_lda_df['assetCodes'].str.findall(f"'([\w\./]+)'")    
    
    # Expand assetCodes
    assetCodes_expanded = list(chain(*tsne_lda_df['assetCodes']))
    assetCodes_index = tsne_lda_df.index.repeat( tsne_lda_df['assetCodes'].apply(len) )

    assert len(assetCodes_index) == len(assetCodes_expanded)
    df_assetCodes = pd.DataFrame({'level_0': assetCodes_index, 'assetCode': assetCodes_expanded})

# Create expandaded news (will repeat every assetCodes' row)
    news_cols = ['time', 'assetCodes'] + sorted(news_cols_agg.keys())
    news_train_df_expanded = pd.merge(df_assetCodes, news_train_df[news_cols], left_on='level_0', right_index=True, suffixes=(['','_old']))

    del news_train_df, df_assetCodes

    # Aggregate numerical news features
    news_train_df_aggregated = news_train_df_expanded.groupby(['time', 'assetCode']).agg(news_cols_agg)
    
    # Free memory
    del news_train_df_expanded

    # Convert to float32 to save memory
    news_train_df_aggregated = news_train_df_aggregated.apply(np.float32)

    # Flat columns
    news_train_df_aggregated.columns = ['_'.join(col).strip() for col in news_train_df_aggregated.columns.values]

    # Join with train
    market_normalized_df = market_normalized.join(news_train_df_aggregated, on=['time', 'assetCode'])

    # Free memory
    del news_train_df_aggregated
    
    return market_train_df
