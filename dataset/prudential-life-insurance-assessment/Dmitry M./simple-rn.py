import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv

print("Load the data using pandas")
train = pd.read_csv("../input/train.csv", header=0)
test = pd.read_csv("../input/test.csv", header=0)

# Сразу забираем сопроводительные данные и избавляемся от них
# Искомая величина
target = train.Response
# Убираем лишнее из тренировочного сета
train = train.drop(['Id', 'Response'], axis=1)
#
# Из тестового сета тоже, приводим их к одному виду тем самым
ids = test['Id'].values
# Уберем Id из данных для теста, т.к. по нему решений не применяется
test = test.drop(['Id'], axis=1)
#
# Факторизуем продукт инфо 2
new_pi_train, pi_index = pd.factorize(train['Product_Info_2'])
train['Product_Info_2'] = new_pi_train
# Факторизуем продукт инфо по индексам из тренировчного сета
test['Product_Info_2'] = pi_index.get_indexer(test.Product_Info_2)

# Теперь уберем все NaN данные из ОБОИХ сетов
for (name_train, series_train), (name_test, series_test) in zip(train.iteritems(), test.iteritems()):
    tmp_len_train = len(train[series_train.isnull()])
    if tmp_len_train > 0:
        # print(name)
        # print(tmp_len)
        #
        # Сразу заменим
        train.loc[series_train.isnull(), name_train] = series_train.mean()
    #
    tmp_len_test = len(test[series_test.isnull()])
    if tmp_len_test > 0:
        # Среднее ставим из TRAIN !!!
        test.loc[series_test.isnull(), name_test] = series_train.mean()
    #

# Собственно обучимся теперь
print('Training...')
forest = RandomForestClassifier(n_estimators=1000)
forest = forest.fit(train.values, target.values)
# clf = SVC()
# clf.fit(train_data[0::, 1::], train_data[0::, 0])

print('Predicting...')

output = forest.predict(test.values).astype(int)
# output = clf.predict(test_data).astype(int)

predictions_file = open("res1.csv", "w")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["Id", "Response"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print('Done.')