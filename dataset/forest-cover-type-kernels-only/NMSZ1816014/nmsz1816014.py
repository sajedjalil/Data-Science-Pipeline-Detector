
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
#from sklearn import cross_validation
#from sklearn.learning_curve import learning_curve
from pylab import *
from sklearn.metrics import accuracy_score
mpl.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.info()
rem=[]
for c in train.columns:
    if train[c].std()==0:
        rem.append(c)
print("We will drop column",rem)
train.drop(rem,axis=1,inplace=True)
train.groupby('Cover_Type').size()
size=10
data=train.iloc[:,:size]
cols=data.columns
data_corr=data.corr()
data.corr()
threshold=0.5
corr_list=[]
for i in range(0,size):
    for j in range(i+1,size):
        if(data_corr.iloc[i,j]>=threshold and data_corr.iloc[i,j]<1) or (data_corr.iloc[i,j]>-1 and data_corr.iloc[i,j]<=-0.5):
            corr_list.append([data_corr.iloc[i,j],i,j])
s_corr_list=sorted(corr_list,key=lambda x: -abs(x[0]))
for v,i,j in s_corr_list:
    print("%s and %s =%.2f" % (cols[i],cols[j],v))
    train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])
train['ele_vert'] = train.Elevation-train.Vertical_Distance_To_Hydrology

train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x)


train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3 
train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2 

train['Shadiness_mean_hillshade']=(train['Hillshade_9am']+train['Hillshade_Noon']+train['Hillshade_3pm'])/3
train['Slope*Elevation']=train['Slope']*train['Elevation']

test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])
test['ele_vert'] = test.Elevation-test.Vertical_Distance_To_Hydrology

test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any


test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2

test['Shadiness_mean_hillshade']=(test['Hillshade_9am']+test['Hillshade_Noon']+test['Hillshade_3pm'])/3
test['Slope*Elevation']=test['Slope']*test['Elevation']

feature=[col for col in train.columns if col not in ['Cover_Type','Id']]
X_train=train[feature]
X_test=test[feature]
etc=ensemble.ExtraTreesClassifier(n_estimators=350)  
etc.fit(X_train, train['Cover_Type'])
sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": etc.predict(X_test)})
sub.to_csv("submit.csv", index=False)