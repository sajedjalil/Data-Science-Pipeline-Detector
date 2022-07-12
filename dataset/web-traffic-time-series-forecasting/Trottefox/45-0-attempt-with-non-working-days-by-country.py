# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


#This script is a fork of clustifier's job
#https://www.kaggle.com/clustifier/weekend-weekdays
#my attempt was to go a bit further by identifying the main non-working days by countries

import pandas as pd
import numpy as np
import re
import gc; gc.enable()
from sklearn.feature_extraction import text
from sklearn import naive_bayes

train = pd.read_csv("../input/train_1.csv")
#determine idiom with URL
train['origine']=train['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])
'''
This is what you get with a value counts on train.origine
en    24108
ja    20431
de    18547
fr    17802
zh    17229
ru    15022
es    14069
ts    13556
er     4299
'''
#we have english, japanese, deutch, french, chinese (taiwanese ?), russian, spanish
#ts and er are undetermined; in the next lines, I try to replace them by learning from special chars
#Note : this step wasn't tuned, and can't be perfect because other idioms are available in those Pages (such as portuguese for example)

#let's make a train, target, and test to predict language on ts and er pages
orig_train=train.loc[~train.origine.isin(['ts', 'er']), 'Page']
orig_target=train.loc[~train.origine.isin(['ts', 'er']), 'origine']
orig_test=train.loc[train.origine.isin(['ts', 'er']), 'Page']
#keep only interesting chars
orig_train2=orig_train.apply(lambda x:x.split(".wikipedia")[0][:-3]).apply(lambda x:re.sub("[a-zA-Z0-9():\-_ \'\.\/]", "", x))
orig_test2=orig_test.apply(lambda x:x.split(".wikipedia")[0][:-3]).apply(lambda x:re.sub("[a-zA-Z0-9():\-_ \'\.\/]", "", x))
#run TFIDF on those specific chars
tfidf=text.TfidfVectorizer(input='content', encoding='utf-8', decode_error='strict', strip_accents=None, 
                     lowercase=True, preprocessor=None, tokenizer=None, 
                     analyzer='char', #stop_words=[chr(x) for x in range(97,123)]+[chr(x) for x in range(65,91)]+['_','.',':'], 
                     token_pattern='(?u)\\b\\w\\w+\\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=True, norm='l2', 
                     use_idf=True, smooth_idf=True, sublinear_tf=False)
orig_train2=tfidf.fit_transform(orig_train2)
#apply a simple naive bayes on the text features
model=naive_bayes.BernoulliNB()
model.fit(orig_train2, orig_target)
result=model.predict(tfidf.transform(orig_test2))
result=pd.DataFrame(result, index=orig_test)
result.columns=['origine']
#result will be used later to replace 'ts' and 'er' values
#we need to remove train.origine so that the train can be flattened with melt
del train['origine']

#let's flatten the train as did clustifier and initialize a "ferie" columns instead of a weekend column
train = pd.melt(train[list(train.columns[-49:])+['Page']], id_vars='Page', var_name='date', value_name='Visits')
train['date'] = train['date'].astype('datetime64[ns]')
train['ferie'] = ((train.date.dt.dayofweek) >=5).astype(float)
train['origine']=train['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])

#let's join with result to replace 'ts' and 'er'
join=train.loc[train.origine.isin(["ts","er"]), ['Page']]
join['origine']=0 #init
join.index=join["Page"]
join.origine=result
train.loc[train.origine.isin(["ts","er"]), ['origine']]=join.origine.values #replace

#official non working days by country (manual search with google)
#I made a lot of shortcuts considering that only Us and Uk used english idiom, 
#only Spain for spanich, only France for french, etc
train_us=['2015-07-04','2015-11-26','2015-12-25']+\
['2016-07-04','2016-11-24','2016-12-26']
test_us=[]
train_uk=['2015-12-25','2015-12-28'] +\
['2016-01-01','2016-03-28','2016-05-02','2016-05-30','2016-12-26','2016-12-27']
test_uk=['2017-01-01']
train_de=['2015-10-03', '2015-12-25', '2015-12-26']+\
['2016-01-01', '2016-03-25', '2016-03-26', '2016-03-27', '2016-01-01', '2016-05-05', '2016-05-15', '2016-05-16', '2016-10-03', '2016-12-25', '2016-12-26']
test_de=['2017-01-01']
train_fr=['2015-07-14', '2015-08-15', '2015-11-01', '2015-11-11', '2015-12-25']+\
['2016-01-01','2016-03-28', '2016-05-01', '2016-05-05', '2016-05-08', '2016-05-16', '2016-07-14', '2016-08-15', '2016-11-01','2016-11-11', '2016-12-25']
test_fr=['2017-01-01']
train_ru=['2015-11-04']+\
['2016-01-01', '2016-01-02', '2016-01-03', '2016-01-04', '2016-01-05', '2016-01-06', '2016-01-07', '2016-02-23', '2016-03-08', '2016-05-01', '2016-05-09', '2016-06-12', '2016-11-04']
test_ru=['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06', '2017-01-07', '2017-02-23']
train_es=['2015-08-15', '2015-10-12', '2015-11-01', '2015-12-06', '2015-12-08', '2015-12-25']+\
['2016-01-01', '2016-01-06', '2016-03-25', '2016-05-01', '2016-08-15', '2016-10-12', '2016-11-01', '2016-12-06', '2016-12-08', '2016-12-25']
test_es=['2017-01-01', '2017-01-06']
train_ja=['2015-07-20','2015-09-21', '2015-10-12', '2015-11-03', '2015-11-23', '2015-12-23']+\
['2016-01-01', '2016-01-11', '2016-02-11', '2016-03-20', '2016-04-29', '2016-05-03', '2016-05-04', '2016-05-05', '2016-07-18', '2016-08-11', '2016-09-22', '2016-10-10', '2016-11-03', '2016-11-23', '2016-12-23']
test_ja=['2017-01-01', '2017-01-09', '2017-02-11']
train_zh=['2015-09-27', '2015-10-01', '2015-10-02','2015-10-03','2015-10-04','2015-10-05','2015-10-06','2015-10-07']+\
['2016-01-01', '2016-01-02', '2016-01-03', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-04-04', '2016-05-01', '2016-05-02', '2016-06-09', '2016-06-10', '2016-09-15', '2016-09-16', '2016-10-03', '2016-10-04','2016-10-05','2016-10-06','2016-10-07']
test_zh=['2017-01-02', '2017-02-27', '2017-02-28', '2017-03-01']
#in China some saturday and sundays are worked
train_o_zh=['2015-10-10','2016-02-06', '2016-02-14', '2016-06-12', '2016-09-18', '2016-10-08', '2016-10-09']
test_o_zh=['2017-01-22', '2017-02-04']

#let's replace values in 'ferie' columns
train.loc[(train.origine=='en')&(train.date.isin(train_us+train_uk)), 'ferie']=1
train.loc[(train.origine=='de')&(train.date.isin(train_de)), 'ferie']=1
train.loc[(train.origine=='fr')&(train.date.isin(train_fr)), 'ferie']=1
train.loc[(train.origine=='ru')&(train.date.isin(train_ru)), 'ferie']=1
train.loc[(train.origine=='es')&(train.date.isin(train_es)), 'ferie']=1
train.loc[(train.origine=='ja')&(train.date.isin(train_ja)), 'ferie']=1
train.loc[(train.origine=='zh')&(train.date.isin(train_zh)), 'ferie']=1
train.loc[(train.origine=='zh')&(train.date.isin(train_o_zh)), 'ferie']=0

#same with test
test = pd.read_csv("../input/key_1.csv")
test['date'] = test.Page.apply(lambda a: a[-10:])
test['Page'] = test.Page.apply(lambda a: a[:-11])
test['date'] = test['date'].astype('datetime64[ns]')
test['ferie'] = ((test.date.dt.dayofweek) >=5).astype(float)
test['origine']=test['Page'].apply(lambda x:re.split(".wikipedia.org", x)[0][-2:])

#joint with result
join=test.loc[test.origine.isin(["ts","er"]), ['Page']]
join['origine']=0
join.index=join["Page"]
join.origine=result
test.loc[test.origine.isin(["ts","er"]), ['origine']]=join.origine.values

test.loc[(test.origine=='en')&(test.date.isin(test_us+test_uk)), 'ferie']=1
test.loc[(test.origine=='de')&(test.date.isin(test_de)), 'ferie']=1
test.loc[(test.origine=='fr')&(test.date.isin(test_fr)), 'ferie']=1
test.loc[(test.origine=='ru')&(test.date.isin(test_ru)), 'ferie']=1
test.loc[(test.origine=='es')&(test.date.isin(test_es)), 'ferie']=1
test.loc[(test.origine=='ja')&(test.date.isin(test_ja)), 'ferie']=1
test.loc[(test.origine=='zh')&(test.date.isin(test_zh)), 'ferie']=1
test.loc[(test.origine=='zh')&(test.date.isin(test_o_zh)), 'ferie']=0

train_page_per_dow = train.groupby(['Page','ferie']).median().reset_index()
test = test.merge(train_page_per_dow, how='left')

test.loc[test.Visits.isnull(), 'Visits'] = 0
test['Visits']=(test['Visits']*10).astype('int')/10
test[['Id','Visits']].to_csv('mad49_guess_idiom.csv', index=False)