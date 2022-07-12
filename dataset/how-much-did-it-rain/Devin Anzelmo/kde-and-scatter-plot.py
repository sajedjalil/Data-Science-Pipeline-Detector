# coding: utf-8
import pandas as pd
import numpy as np
import zipfile
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None
#counts the number of radar scans in the row by using white spaces
def get_num_radar_scans(timetoend_row):
    return timetoend_row.count(' ') + 1

def mean_of_row(row):
    temp = np.array(list(map(np.double, row.split(' '))))
    temp[temp < -5000] = np.nan
    return np.mean(temp)


z = zipfile.ZipFile('../input/train_2013.csv.zip')
train = pd.read_csv(z.open('train_2013.csv'),usecols=['Expected','Reflectivity','DistanceToRadar'])


train['mean_ref'] = train.Reflectivity.apply(mean_of_row)
train['mean_dist'] = train.DistanceToRadar.apply(mean_of_row)

train.drop(['Reflectivity','DistanceToRadar'],axis=1,inplace=True)

train.columns = ['Expected','mean_Reflectance','mean_DistanceToRadar']
train = train.dropna()
train.loc[:,'Expected'] = np.ceil(train.Expected)

rain_amount = ['0mm to 15mm']*len(train)
for i in range(len(train)):
    if train.Expected.iloc[i] > 200:
        rain_amount[i] = 'Greater than 200mm'
    elif train.Expected.iloc[i] >15:
        rain_amount[i] = '15mm to 200mm'
train['Amount of rain '] = rain_amount

g = sns.pairplot(train,hue='Amount of rain ',diag_kind='kde', vars=['mean_DistanceToRadar','mean_Reflectance'],kind='scatter')
txt = '''
Top left: Kde plot showing the DistanceToRadar density for different three subsets 0mm to 15mm,
15mm to 200mm and greater then 200mm. The red line shows that there are specific distances with 
many more bad rain gauge readings. Bottom right: Shows that reflectivity for 200m and greater is
closer to 0mm then 15mm showing that large values are do to a faulty mechanism. 
Scatter plots: green is actual high rain rate, red are bad rain gauge readings.'''
g.fig.text(0.08,-0.12,txt,style='italic',fontsize=10)
g.fig.suptitle('Kde and Scatter Plot of mean Reflectivity and DistanceToRadar', fontsize=14, x=.45, y=1.02)
g.savefig('figure3.png', bbox_inches='tight')