# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import math
from sklearn.linear_model import LinearRegression

traindata = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv',encoding='UTF-8')
testdata = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv',encoding='UTF-8')
submissiondata = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv',encoding='UTF-8')


def covid_cc_fc(data_R1, testdata_R1):
    data_R1_0 = data_R1[~data_R1['ConfirmedCases'].isin([0])].reset_index()
    data_R1_0.loc[:, 'Date'] = pd.to_datetime(data_R1_0['Date'])
    date1 = data_R1_0.loc[0, 'Date']
    data_R1_0.loc[:, 'Date'] = data_R1_0['Date'].map(lambda x: x - date1).astype('timedelta64[D]').astype('float')
    data_R1_0.loc[:, 'ConfirmedCases'] = data_R1_0['ConfirmedCases'].map(lambda x: math.log(x))

    data2d_d = np.array(data_R1_0.Date).reshape(-1, 1)
    data2d_lncc = np.array(data_R1_0.ConfirmedCases).reshape(-1, 1)

    model = LinearRegression()
    model.fit(data2d_d, data2d_lncc)
    # mc = model.coef_
    # mi = model.intercept_
    # transmission_rate = math.exp(mc)

    testdata_R1.loc[:, 'Date'] = pd.to_datetime(testdata_R1.loc[:, 'Date']).map(lambda x: x - date1).astype(
        'timedelta64[D]').astype('float')
    testdata2d_d = np.array(testdata_R1.Date).reshape(-1, 1)
    testdata2d_lncc = model.predict(testdata2d_d)
    for x in np.nditer(testdata2d_lncc, op_flags=['readwrite']):
        x[...] = math.exp(x)
    testdata2d_cc = testdata2d_lncc.astype('int')
    sub_R1_cc = pd.DataFrame(testdata2d_cc, columns=['ConfirmedCases'])

    return sub_R1_cc

def covid_f_fc(data_R1,sub_R1_cc):
    data_R1_F0 = data_R1[~data_R1['Fatalities'].isin([0])].reset_index()
    data_FCR = data_R1_F0.loc[:,'Fatalities']/data_R1_F0.loc[:,'ConfirmedCases']
    FCR = data_FCR.describe()['mean']
    sub_R1 = sub_R1_cc
    sub_R1.loc[:,'Fatalities'] = (sub_R1['ConfirmedCases']*FCR).astype('int')
    return sub_R1

amount_cc = traindata.ConfirmedCases.describe()['count']*traindata.ConfirmedCases.describe()['mean']
amount_f = traindata.Fatalities.describe()['count']*traindata.Fatalities.describe()['mean']
overallFCR = amount_f/amount_cc

for i in range(0, 306):

    data_R1 = traindata[i * 73:(i + 1) * 73]
    if data_R1.ConfirmedCases[(i + 1) * 73 - 1] == 0:
        submissiondata.loc[i * 43:(i + 1) * 43 - 1, ['ConfirmedCases', 'Fatalities']] = 0
        continue
    testdata_R1 = testdata[i * 43:(i + 1) * 43]
    sub_R1_cc = covid_cc_fc(data_R1, testdata_R1)

    if data_R1.Fatalities[(i + 1) * 73 - 1] != 0:
        sub_R1 = covid_f_fc(data_R1, sub_R1_cc)

    else:
        sub_R1 = sub_R1_cc
        sub_R1.loc[:, 'Fatalities'] = (sub_R1['ConfirmedCases'] * overallFCR).astype('int')

    sub_R1.index = pd.Series(list(range(i * 43, (i + 1) * 43)))
    submissiondata.loc[i * 43:(i + 1) * 43 - 1, ['ConfirmedCases', 'Fatalities']] = sub_R1[['ConfirmedCases', 'Fatalities']]

submissiondata.to_csv('submission.csv',index=None)