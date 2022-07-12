import pickle
import pandas as pd
import numpy as np
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()

(market_train, news_train) = env.get_training_data()

market_test = []
news_test = []
predictions_test = []
for (market_obs, news_obs, predictions_template) in env.get_prediction_days():
    market_test.append(market_obs)
    news_test.append(news_obs)
    predictions_test.append(predictions_template)
    
    predictions_template.confidenceValue = 0.0
    env.predict(predictions_template)

# market_train[0:1418]      ->  market_train2007-02-01.csv
# news_train[59619:64338]   ->  news_train2007-02-01.csv

# creare un DataFrame completo vuoto (colonne nominate)
# conosciamo la grandezza del DataFrame completo (colonne=market+news, righe=market)
#completeDF = pd.DataFrame(index=np.arange(0, 5000), columns=market_train.columns.append(news_train.columns))

# adds to a DataFrame a row when a news deals a market
# da migliorare la velocità:
# https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe
# https://stackoverflow.com/a/17496530
# NON È POSSIBILE ITERARE SU TUTTE LE NEWS PER OGNI MARKET
#i=0
#for rowMarket in market_train[0:1418].itertuples(index=False):
#    for rowNews in news_train[59619:64338].itertuples(index=False):
#        if "'"+rowMarket.assetCode+"'" in rowNews.assetCodes:
#            completeDF.iloc[i] = pd.concat([market_train[i:i+1], news_train[i:i+1]], axis=1, sort=False, join='outer').iloc[0]
#            i += 1

# eliminare una colonna
# news_train.columns.drop(['volumeCounts7D'])

# inserire in una riga di un DataFrame una riga di due dataframe concatenati
#completeDF.iloc[1] = pd.concat([market_train[0:1], news_train[0:1]], axis=1, sort=False, join='outer').iloc[0]

# merge tra due DataFrame
#pd.merge(market_train[0:1], news_train[0:1])

# conta valori diversi per colonna di market
#market_train.assetCode.value_counts()

# salva in un DataFrame tutte le righe di market con un determinato assetCode
dfMarketAN = market_train.loc[market_train['assetCode'] == 'A.N']
dfMarketAN.to_csv("market_trainAN.csv", index = None, header = True)

# salva in un DataFrame tutte le righe di news che trattano un determinato assetCode
dfNewsAN = news_train.loc[news_train['assetCodes'].apply(lambda x: "'A.N'" in x)]
dfNewsAN.to_csv("news_trainAN.csv", index = None, header = True)



# stampa righe market con determinato assetCode [NO]
#for rowMarket in market_train.itertuples(index=False):
#    if rowMarket.assetCode == "A.N":
#        print(rowMarket)

# stampa righe market e news che trattano stesso assetCode
#for rowMarket in market_train[0:1418].itertuples(index=False):
#    for rowNews in news_train[0:5962].itertuples(index=False):
#        if "'A.N'" in rowNews.assetCodes:
#            print(rowNews)

# stampa assetCode di market
# print(market_train[0:1].assetCode.to_string(index=False))

# create a Timestamp
#d1 = pd.Timestamp('2007-02-01 22:00:00', tz='UTC')

# verifica in news se time, sourceTimestamp, firstCreated sono diversi
#ab=0
#ac=0
#bc=0
#abc=0
#for rowNews in news_train.itertuples():
#    if rowNews.time != rowNews.sourceTimestamp:
#        ab = ab+1
#    if rowNews.time != rowNews.firstCreated:
#        ac = ac+1
#    if rowNews.sourceTimestamp != rowNews.firstCreated:
#        bc = bc+1
#    if (rowNews.time != rowNews.sourceTimestamp) and (rowNews.time != rowNews.firstCreated) and (rowNews.sourceTimestamp != rowNews.firstCreated):
#        abc = abc+1

# riga 59659
# for row in news_train[59657:59658].itertuples(index=True):
#    print("num elementi: {}, {}".format(len(row), row))
# riga 59698
# for row in news_train[59696:59698].itertuples(index=True):
#    print("num elementi: {}, {}".format(len(row), row))
    

#news_train[59696:59698].to_csv("news_train1.csv", index=None, header=True)
#news_train[59696:59698].to_csv("news_train2.csv", index=None, header=True, sep=',')

# 2007-02-01
# market_train[0:1418].to_csv("market_train2007-02-01.csv", index = None, header = True)
# news_train[59619:64338].to_csv("news_train2007-02-01.csv", index = None, header = True)

#market_test = []
#news_test = []
#predictions_test = []
#for (market_obs, news_obs, predictions_template) in env.get_prediction_days():
#    market_test.append(market_obs)
#    news_test.append(news_obs)
#    predictions_test.append(predictions_template)
    
#    predictions_template.confidenceValue = 0.0
#    env.predict(predictions_template)

#print("START market_test")
#market_test.to_csv("market_test.csv", sep = '~')
#print("START news_test")
#news_test.to_csv("news_test.csv", sep = '~')
#print("START predictions_test")
#predictions_test.to_csv("predictions_test.csv", sep = '~')

print("ALL DONE!!!")