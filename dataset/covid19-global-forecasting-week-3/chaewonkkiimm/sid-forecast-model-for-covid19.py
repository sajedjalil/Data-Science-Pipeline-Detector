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

import csv
import math
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import matplotlib.pyplot as plt

def get_values_from_csv():
    location = []
    trainset = pd.read_csv(root+'/train.csv', engine='python')
    #groupby = trainset.groupby(['Province_State', 'Country_Region']).last()

    groupby = trainset.drop_duplicates(['Province_State', 'Country_Region'], keep='last')
    province = groupby.loc[:, 'Province_State'].values
    country = groupby.loc[:, 'Country_Region'].values
    for i in range(len(province)):
        location.append((province[i], country[i]))
    return location

def loss(point, confirmed, deceased):
    """
    RMSE between actual confirmed cases and the estimated infectious people with given beta and gamma.
    """
    size = len(confirmed) #67
    beta, gamma = point
    def SID(t, y):
        S = y[0]
        I = y[1]
        D = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    solution = solve_ivp(SID, [0, size], [S_0,I_0,D_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] + solution.y[2] - confirmed.values)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - deceased.values)**2))

    return l1 + l2


class Learner(object):
    def __init__(self, province, country, loss):
        self.province = province
        self.country = country
        self.loss = loss

    def load_confirmed(self):
        trainset = pd.read_csv(root+'/train.csv', engine='python')
        location_df = trainset[trainset['Country_Region'] == self.country]
        if self.province is np.nan:
            return location_df.iloc[:].loc[:, 'ConfirmedCases']
        location_df = location_df[location_df['Province_State'] == self.province]
        return location_df.iloc[:].loc[:, 'ConfirmedCases']
    
    def load_fatalities(self):
        trainset = pd.read_csv(root+'/train.csv', engine='python')
        location_df = trainset[trainset['Country_Region'] == self.country]
        if self.province is np.nan:
            return location_df.iloc[:].loc[:, 'Fatalities']
        location_df = location_df[location_df['Province_State'] == self.province]
        return location_df.iloc[:].loc[:, 'Fatalities']

    def extend_index(self, index, new_size):
        values = index.values
        current = index[-1]
        while len(values)<new_size:
            current = current + 1
            values = np.append(values, current)
        return values
        
    def predict(self, beta, gamma, confirmed, deceased):
        predict_range = 107
        # data index : INT64Index([9782, ..., 9848])
        new_index = self.extend_index(confirmed.index, predict_range)
        #[9782, 9283, ..., 9881]
        size = len(new_index)
        def SID(t, y):
            S = y[0]
            I = y[1]
            D = y[2]
            return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
        extended_confirmed = np.concatenate((confirmed.values, [None] * (size - len(confirmed.values))))
        extended_deceased = np.concatenate((deceased.values, [None] * (size - len(deceased.values))))
        return new_index, extended_confirmed, extended_deceased, solve_ivp(SID, [0, size], [S_0,I_0,D_0], t_eval=np.arange(0, size, 1))

    def test_results(self, new_index, extended_confirmed, extended_deceased, prediction):
        S = prediction.y[0]
        I = prediction.y[1]
        D = prediction.y[2]
        result = pd.read_csv(root+'/test.csv', engine='python')
        country_df = result[result['Country_Region'] == self.country]
        if self.province is np.nan:
            forecast_ID = country_df.iloc[:].loc[:, 'ForecastId']
        else:
            location_df = country_df[result['Province_State']==self.province]
            forecast_ID = location_df.iloc[:].loc[:, 'ForecastId']
        index = forecast_ID.index
        start = index.values[0] + 1

        r = csv.reader(open(root+'/submission.csv')) # Here your csv file
        lines = list(r)
        for i in range(12):
            lines[start+i][1] = extended_confirmed[64+i]
            lines[start+i][2] = extended_deceased[64+i]
        for j in range(31):
            lines[start+12+j][1] = math.floor(I[76+j] + D[76+j])
            lines[start+12+j][2] = math.floor(D[76+j])
        writer = csv.writer(open('submission.csv', 'w'))
        writer.writerows(lines)

    def train(self):
        confirmed = self.load_confirmed()
        deceased = self.load_fatalities()
        infected = confirmed-deceased
        optimal = minimize(
            loss,
            [0.001, 0.001],
            args=(confirmed, deceased),
            method='L-BFGS-B',
            bounds=[(0.00000001, 0.4), (0.00000001, 0.4)]
        )
        beta, gamma = optimal.x
        print(self.country, 'beta:',beta, 'gamma:', gamma)
        new_index, extended_infected, extended_deceased, prediction = self.predict(beta, gamma, infected, deceased)
        self.test_results(new_index, extended_infected, extended_deceased, prediction)
        '''
        df = pd.DataFrame({
            'confirmed': extended_infected,
            'deceased' : extended_deceased,
            'S': prediction.y[0],
            'I': prediction.y[1],
            'D': prediction.y[2]
        }, index=new_index)
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        df.plot(ax=ax)
        fig.savefig(f"./plots_countries/{self.country}.png")
        '''


if __name__ == "__main__":
    root = '/kaggle/input/covid19-global-forecasting-week-3'

    list_region = get_values_from_csv()
    
    for i in range(len(list_region)):
        province = list_region[i][0]
        country = list_region[i][1]
        print(province, country)
        S_0, I_0, R_0, D_0 = 100000, 1, 0, 0
        SID_model = Learner(province, country, 0)
    
        SID_model.train()
        
    sub = pd.read_csv('submission.csv')
    sub.to_csv(root+'submission.csv', index=False)

