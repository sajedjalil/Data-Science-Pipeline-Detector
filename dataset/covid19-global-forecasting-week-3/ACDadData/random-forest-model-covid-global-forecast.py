# %% [code]
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

# %% [code]
#import associated libraries
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

pd.set_option("display.max_columns", 10000)
pd.set_option("display.max_rows", 50000)

# %% [code]
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')
train["train_or_test"]="train"
train_no_dep = train.drop([ 'ConfirmedCases',
       'Fatalities'], axis=1)
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')
test["train_or_test"]="test"
test.rename(columns={'ForecastId':'Id'}, 
                 inplace=True)
frames = [train_no_dep, test]
df= pd.concat(frames)

# %% [code]
df.Country_Region = df.Country_Region.str.strip().str.replace(' ', '_')
df.Country_Region = df.Country_Region.str.strip().str.replace('-', '_')
df.Country_Region = df.Country_Region.str.strip().str.replace('(', '_')
df.Country_Region = df.Country_Region.str.strip().str.replace(')', '_')
df.Country_Region = df.Country_Region.str.strip().str.replace('*', '_')
df.Country_Region = df.Country_Region.str.strip().str.replace("'", '_')
df.Country_Region = df.Country_Region.str.strip().str.replace("Korea,_South", 'South_Korea') 
df.Country_Region = df.Country_Region.replace(np.nan, "NOT_STATED")
country=df[['Country_Region','Province_State']]
# Get dummies
country = pd.get_dummies(country, prefix_sep='_', drop_first=False)
df = pd.concat( [df, country], axis=1)

# %% [code]
#split the entire dataframe into smaller dataframesbased on country_region names
for i, g in df.groupby('Country_Region'):
    globals()['df_' + str(i)] =  g

# %% [code]
#runthrough each of the several country dataframes created and 
#convertdate to datetime
#create new column called start-up which is the date of first report for each country
#subtract each date from start-up to get days from the first report date
for df in (df_Afghanistan,df_Albania,df_Algeria,df_Andorra,df_Angola,df_Antigua_and_Barbuda,df_Argentina,df_Armenia,df_Australia,df_Austria,df_Azerbaijan,df_Bahamas,df_Bahrain,df_Bangladesh,df_Barbados,df_Belarus,df_Belgium,df_Belize,df_Benin,df_Bhutan,df_Bolivia,df_Bosnia_and_Herzegovina,df_Botswana,df_Brazil,df_Brunei,df_Bulgaria,df_Burkina_Faso,df_Burma,df_Burundi,df_Cabo_Verde,df_Cambodia,df_Cameroon,df_Canada,df_Central_African_Republic,df_Chad,df_Chile,df_China,df_Colombia,df_Congo__Brazzaville_,df_Congo__Kinshasa_,df_Costa_Rica,df_Cote_d_Ivoire,df_Croatia,df_Cuba,df_Cyprus,df_Czechia,df_Denmark,df_Diamond_Princess,df_Djibouti,df_Dominica,df_Dominican_Republic,df_Ecuador,df_Egypt,df_El_Salvador,df_Equatorial_Guinea,df_Eritrea,df_Estonia,df_Eswatini,df_Ethiopia,df_Fiji,df_Finland,df_France,df_Gabon,df_Gambia,df_Georgia,df_Germany,df_Ghana,df_Greece,df_Grenada,df_Guatemala,df_Guinea,df_Guinea_Bissau,df_Guyana,df_Haiti,df_Holy_See,df_Honduras,df_Hungary,df_Iceland,df_India,df_Indonesia,df_Iran,df_Iraq,df_Ireland,df_Israel,df_Italy,df_Jamaica,df_Japan,df_Jordan,df_Kazakhstan,df_Kenya,df_South_Korea,df_Kosovo,df_Kuwait,df_Kyrgyzstan,df_Laos,df_Latvia,df_Lebanon,df_Liberia,df_Libya,df_Liechtenstein,df_Lithuania,df_Luxembourg,df_Madagascar,df_Malaysia,df_Maldives,df_Mali,df_Malta,df_Mauritania,df_Mauritius,df_Mexico,df_Moldova,df_Monaco,df_Mongolia,df_Montenegro,df_Morocco,df_Mozambique,df_MS_Zaandam,df_Namibia,df_Nepal,df_Netherlands,df_New_Zealand,df_Nicaragua,df_Niger,df_Nigeria,df_North_Macedonia,df_Norway,df_Oman,df_Pakistan,df_Panama,df_Papua_New_Guinea,df_Paraguay,df_Peru,df_Philippines,df_Poland,df_Portugal,df_Qatar,df_Romania,df_Russia,df_Rwanda,df_Saint_Kitts_and_Nevis,df_Saint_Lucia,df_Saint_Vincent_and_the_Grenadines,df_San_Marino,df_Saudi_Arabia,df_Senegal,df_Serbia,df_Seychelles,df_Sierra_Leone,df_Singapore,df_Slovakia,df_Slovenia,df_Somalia,df_South_Africa,df_Spain,df_Sri_Lanka,df_Sudan,df_Suriname,df_Sweden,df_Switzerland,df_Syria,df_Taiwan_,df_Tanzania,df_Thailand,df_Timor_Leste,df_Togo,df_Trinidad_and_Tobago,df_Tunisia,df_Turkey,df_Uganda,df_Ukraine,df_United_Arab_Emirates,df_United_Kingdom,df_Uruguay,df_US,df_Uzbekistan,df_Venezuela,df_Vietnam,
           df_West_Bank_and_Gaza,df_Zambia,df_Zimbabwe):
    df['Date'] = pd.to_datetime(df['Date'])
    df['start_up'] = df['Date'].iloc[0]
    df['start_up'] = pd.to_datetime(df['start_up'])
    df['days'] = (df['Date'] - df['start_up']) / pd.offsets.Day(1)

# %% [code]
#runthrough each of the several country dataframes created and 
#convertdate to datetime
#create new column called start-up which is the date of first report for each country
#subtract each date from start-up to get days from the first report date
for df in (df_Afghanistan,df_Albania,df_Algeria,df_Andorra,df_Angola,df_Antigua_and_Barbuda,df_Argentina,df_Armenia,df_Australia,df_Austria,df_Azerbaijan,df_Bahamas,df_Bahrain,df_Bangladesh,df_Barbados,df_Belarus,df_Belgium,df_Belize,df_Benin,df_Bhutan,df_Bolivia,df_Bosnia_and_Herzegovina,df_Botswana,df_Brazil,df_Brunei,df_Bulgaria,df_Burkina_Faso,df_Burma,df_Burundi,df_Cabo_Verde,df_Cambodia,df_Cameroon,df_Canada,df_Central_African_Republic,df_Chad,df_Chile,df_China,df_Colombia,df_Congo__Brazzaville_,df_Congo__Kinshasa_,df_Costa_Rica,df_Cote_d_Ivoire,df_Croatia,df_Cuba,df_Cyprus,df_Czechia,df_Denmark,df_Diamond_Princess,df_Djibouti,df_Dominica,df_Dominican_Republic,df_Ecuador,df_Egypt,df_El_Salvador,df_Equatorial_Guinea,df_Eritrea,df_Estonia,df_Eswatini,df_Ethiopia,df_Fiji,df_Finland,df_France,df_Gabon,df_Gambia,df_Georgia,df_Germany,df_Ghana,df_Greece,df_Grenada,df_Guatemala,df_Guinea,df_Guinea_Bissau,df_Guyana,df_Haiti,df_Holy_See,df_Honduras,df_Hungary,df_Iceland,df_India,df_Indonesia,df_Iran,df_Iraq,df_Ireland,df_Israel,df_Italy,df_Jamaica,df_Japan,df_Jordan,df_Kazakhstan,df_Kenya,df_South_Korea,df_Kosovo,df_Kuwait,df_Kyrgyzstan,df_Laos,df_Latvia,df_Lebanon,df_Liberia,df_Libya,df_Liechtenstein,df_Lithuania,df_Luxembourg,df_Madagascar,df_Malaysia,df_Maldives,df_Mali,df_Malta,df_Mauritania,df_Mauritius,df_Mexico,df_Moldova,df_Monaco,df_Mongolia,df_Montenegro,df_Morocco,df_Mozambique,df_MS_Zaandam,df_Namibia,df_Nepal,df_Netherlands,df_New_Zealand,df_Nicaragua,df_Niger,df_Nigeria,df_North_Macedonia,df_Norway,df_Oman,df_Pakistan,df_Panama,df_Papua_New_Guinea,df_Paraguay,df_Peru,df_Philippines,df_Poland,df_Portugal,df_Qatar,df_Romania,df_Russia,df_Rwanda,df_Saint_Kitts_and_Nevis,df_Saint_Lucia,df_Saint_Vincent_and_the_Grenadines,df_San_Marino,df_Saudi_Arabia,df_Senegal,df_Serbia,df_Seychelles,df_Sierra_Leone,df_Singapore,df_Slovakia,df_Slovenia,df_Somalia,df_South_Africa,df_Spain,df_Sri_Lanka,df_Sudan,df_Suriname,df_Sweden,df_Switzerland,df_Syria,df_Taiwan_,df_Tanzania,df_Thailand,df_Timor_Leste,df_Togo,df_Trinidad_and_Tobago,df_Tunisia,df_Turkey,df_Uganda,df_Ukraine,df_United_Arab_Emirates,df_United_Kingdom,df_Uruguay,df_US,df_Uzbekistan,df_Venezuela,df_Vietnam,
           df_West_Bank_and_Gaza,df_Zambia,df_Zimbabwe):
    df['after_day_1'] = df['days'].shift(periods=1)
    df['after_day_2'] = df['days'].shift(periods=2)
    df['after_day_3'] = df['days'].shift(periods=3)
    df['after_day_4'] = df['days'].shift(periods=4)
    df['after_day_5'] = df['days'].shift(periods=5)
    df['after_day_6'] = df['days'].shift(periods=6)
    df['after_day_7'] = df['days'].shift(periods=7)
    df['after_day_8'] = df['days'].shift(periods=8)
    df['after_day_9'] = df['days'].shift(periods=9)
    df['after_day_10'] = df['days'].shift(periods=10)
    df['after_day_11'] = df['days'].shift(periods=11)
    df['after_day_12'] = df['days'].shift(periods=12)
    df['after_day_13'] = df['days'].shift(periods=13)
    df['after_day_14'] = df['days'].shift(periods=14)
    df['after_day_15'] = df['days'].shift(periods=15)
    df['after_day_16'] = df['days'].shift(periods=16)
    df['after_day_17'] = df['days'].shift(periods=17)
    df['after_day_18'] = df['days'].shift(periods=18)
    df['after_day_19'] = df['days'].shift(periods=19)
    df['after_day_20'] = df['days'].shift(periods=20)
    df['after_day_21'] = df['days'].shift(periods=21)
    df['after_day_22'] = df['days'].shift(periods=22)
    df['after_day_23'] = df['days'].shift(periods=23)
    df['2_dayperiod_rolling_av'] = (df['after_day_1']+ df['after_day_2'])*0.5
    df['18_dayperiod_rolling_sum'] = df['days']+df['after_day_18']+ df['after_day_17']+ df['after_day_16']+ df['after_day_15']+ df['after_day_14']+ df['after_day_13']+ df['after_day_12']+ df['after_day_11']+ df['after_day_10']+ df['after_day_9']+ df['after_day_8']+ df['after_day_7']+ df['after_day_6']+ df['after_day_5']+ df['after_day_4']+ df['after_day_3']+ df['after_day_2']+ df['after_day_1']
    df['18_dayperiod_weighted_sum'] = df['days']*20+df['after_day_18']*2 + df['after_day_17']*3+ df['after_day_16']*4+ df['after_day_15']*5 + df['after_day_14']*6 + df['after_day_13']*7 + df['after_day_12']*8 + df['after_day_11']*9 + df['after_day_10']*10 + df['after_day_9']*11 + df['after_day_8']*12 + df['after_day_7']*13 + df['after_day_6']*14 + df['after_day_5']*15 + df['after_day_4']*16 + df['after_day_3']*17 + df['after_day_2']*18 + df['after_day_1']*19

# %% [code]
frames = [df_Afghanistan,df_Albania,df_Algeria,df_Andorra,df_Angola,df_Antigua_and_Barbuda,df_Argentina,df_Armenia,df_Australia,df_Austria,df_Azerbaijan,df_Bahamas,df_Bahrain,df_Bangladesh,df_Barbados,df_Belarus,df_Belgium,df_Belize,df_Benin,df_Bhutan,df_Bolivia,df_Bosnia_and_Herzegovina,df_Botswana,df_Brazil,df_Brunei,df_Bulgaria,df_Burkina_Faso,df_Burma,df_Burundi,df_Cabo_Verde,df_Cambodia,df_Cameroon,df_Canada,df_Central_African_Republic,df_Chad,df_Chile,df_China,df_Colombia,df_Congo__Brazzaville_,df_Congo__Kinshasa_,df_Costa_Rica,df_Cote_d_Ivoire,df_Croatia,df_Cuba,df_Cyprus,df_Czechia,df_Denmark,df_Diamond_Princess,df_Djibouti,df_Dominica,df_Dominican_Republic,df_Ecuador,df_Egypt,df_El_Salvador,df_Equatorial_Guinea,df_Eritrea,df_Estonia,df_Eswatini,df_Ethiopia,df_Fiji,df_Finland,df_France,df_Gabon,df_Gambia,df_Georgia,df_Germany,df_Ghana,df_Greece,df_Grenada,df_Guatemala,df_Guinea,df_Guinea_Bissau,df_Guyana,df_Haiti,df_Holy_See,df_Honduras,df_Hungary,df_Iceland,df_India,df_Indonesia,df_Iran,df_Iraq,df_Ireland,df_Israel,df_Italy,df_Jamaica,df_Japan,df_Jordan,df_Kazakhstan,df_Kenya,df_South_Korea,df_Kosovo,df_Kuwait,df_Kyrgyzstan,df_Laos,df_Latvia,df_Lebanon,df_Liberia,df_Libya,df_Liechtenstein,df_Lithuania,df_Luxembourg,df_Madagascar,df_Malaysia,df_Maldives,df_Mali,df_Malta,df_Mauritania,df_Mauritius,df_Mexico,df_Moldova,df_Monaco,df_Mongolia,df_Montenegro,df_Morocco,df_Mozambique,df_MS_Zaandam,df_Namibia,df_Nepal,df_Netherlands,df_New_Zealand,df_Nicaragua,df_Niger,df_Nigeria,df_North_Macedonia,df_Norway,df_Oman,df_Pakistan,df_Panama,df_Papua_New_Guinea,df_Paraguay,df_Peru,df_Philippines,df_Poland,df_Portugal,df_Qatar,df_Romania,df_Russia,df_Rwanda,df_Saint_Kitts_and_Nevis,df_Saint_Lucia,df_Saint_Vincent_and_the_Grenadines,df_San_Marino,df_Saudi_Arabia,df_Senegal,df_Serbia,df_Seychelles,df_Sierra_Leone,df_Singapore,df_Slovakia,df_Slovenia,df_Somalia,df_South_Africa,df_Spain,df_Sri_Lanka,df_Sudan,df_Suriname,df_Sweden,df_Switzerland,df_Syria,df_Taiwan_,df_Tanzania,df_Thailand,df_Timor_Leste,df_Togo,df_Trinidad_and_Tobago,df_Tunisia,df_Turkey,df_Uganda,df_Ukraine,df_United_Arab_Emirates,df_United_Kingdom,df_Uruguay,df_US,df_Uzbekistan,df_Venezuela,df_Vietnam,
           df_West_Bank_and_Gaza,df_Zambia,df_Zimbabwe]

df= pd.concat(frames)
df_train = df[df['train_or_test'] == "train"]
df_test = df[df['train_or_test'] == "test"]
train_dep_variable1 = train.drop(['Province_State', 'Country_Region', 'Date',
       'train_or_test'], axis=1)
train_dep_variable1.rename(columns={'Id':'Double_check_Id'}, 
                 inplace=True)
df_finaltrain = pd.concat( [df_train, train_dep_variable1], axis=1) 
train =df_finaltrain
X_P = train.drop(['Id', 'Date', 'train_or_test', 'start_up','Double_check_Id', 'ConfirmedCases',
       'Fatalities','Country_Region', 'Province_State'], axis=1)
y = train['ConfirmedCases']

# %% [code]
from numpy import float32
X2 = X_P.astype(float32)
X= X2.fillna(X2.median(axis=0))

# %% [code]
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf2 = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=0)
rf2.fit(X_train, y_train)

# %% [code]
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
predicted_train = rf2.predict(X_train)
predicted_test = rf2.predict(X_test)
test_score = r2_score(y_test, predicted_test)
spearman = spearmanr(y_test, predicted_test)
pearson = pearsonr(y_test, predicted_test)
print(f'Out-of-bag R-2 score estimate: {rf2.oob_score_:>5.3}')
print(f'Test data R-2 score: {test_score:>5.3}')
print(f'Test data Spearman correlation: {spearman[0]:.3}')
print(f'Test data Pearson correlation: {pearson[0]:.3}')

# %% [code]
y_pred_train2  = rf2.predict(X)
y_pred_train2

# %% [code]
#now lets predict confirmed cases for the test dataset
test=df_test
X_test = test.drop(['Id', 'Date', 'train_or_test', 'start_up', 'Country_Region', 'Province_State'], axis=1)
from numpy import float32
test22 = X_test.astype(float32)
test2= test22.fillna(test22.median(axis=0))
predicted_test2 = rf2.predict(test2)


# %% [code]
test2.shape

# %% [code]
>>> print('predicted response:', predicted_test2, sep='\n')
COVID_confirmed_cases=pd.DataFrame(predicted_test2, columns=['ConfirmedCases']) 
COVID_confirmed_cases.head()

# %% [code]
X_test_with_id= test[['Id']]
merged_test_prediction_finalsubmission = pd.concat([X_test_with_id,COVID_confirmed_cases], axis=1)

# %% [code]
#merged_test_prediction_finalsubmission.to_csv("COVID_test_Final Prediction Confirmed Cases Results.csv")

# %% [code]
#now that we have finished predicting confirmed cases the next step is to predict Fatalities
X_P = train.drop(['Id', 'Date', 'train_or_test', 'start_up','Double_check_Id', 'Fatalities',
                  'Country_Region', 'Province_State'], axis=1)
y = train['Fatalities']
from numpy import float32
X = X_P.astype(float32)
X= X.fillna(X.median(axis=0))

# %% [code]
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf3 = RandomForestRegressor(n_estimators=1000, oob_score=True, random_state=0)
rf3.fit(X_train, y_train)

# %% [code]
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, pearsonr
predicted_train = rf3.predict(X_train)
predicted_test = rf3.predict(X_test)
test_score = r2_score(y_test, predicted_test)
spearman = spearmanr(y_test, predicted_test)
pearson = pearsonr(y_test, predicted_test)
print(f'Out-of-bag R-2 score estimate: {rf3.oob_score_:>5.3}')
print(f'Test data R-2 score: {test_score:>5.3}')
print(f'Test data Spearman correlation: {spearman[0]:.3}')
print(f'Test data Pearson correlation: {pearson[0]:.3}')

# %% [code]
y_pred_train2  = rf3.predict(X)
y_pred_train2

# %% [code]
>>> print('predicted response:', y_pred_train2, sep='\n')
Predicted_fatalities=pd.DataFrame(y_pred_train2, columns=['Fatalities']) 
Predicted_fatalities.head()

# %% [code]
X_test2_with_id2= train[['Id']]

# %% [code]


# %% [code]
merged_train_prediction_finalsubmission2 = pd.concat([X_test2_with_id2,Predicted_fatalities], axis=1)
merged_test_with_confirmedcases = pd.concat([test,merged_test_prediction_finalsubmission], axis=1)

# %% [code]
#merged_train_prediction_finalsubmission2.to_csv("COVID_train_Final Prediction Fatalities Results.csv")

# %% [code]
#now lets start predicting the test data but remember to append the predicted confirmedcases

# %% [code]
X_test_fat = merged_test_with_confirmedcases.drop(['Id', 'Date', 'train_or_test', 'start_up', 'Country_Region', 'Province_State'], axis=1)
from numpy import float32
test3 = X_test_fat.astype(float32)
predicted_test3 = rf3.predict(test3)

# %% [code]
>>> print('predicted response:', predicted_test3, sep='\n')
COVID_fatality_test=pd.DataFrame(predicted_test3, columns=['Fatalities']) 
X_test_with_id= test[['Id']]
merged_test_prediction_finalsubmission_fatality = pd.concat([X_test_with_id,COVID_fatality_test], axis=1)
#merged_test_prediction_finalsubmission_fatality.to_csv("COVID_test_Final Prediction Fatalities Results.csv")
dec_places = 0
merged_test_prediction_finalsubmission['ConfirmedCases new']= merged_test_prediction_finalsubmission['ConfirmedCases'].round(dec_places) 
dec_places = 0
merged_test_prediction_finalsubmission_fatality['Fatalities new']= merged_test_prediction_finalsubmission_fatality['Fatalities'].round(dec_places) 
merged_test_prediction_finalsubmission_fatality_2 = merged_test_prediction_finalsubmission_fatality.drop(['Id','Fatalities'], axis=1)
merged_test_prediction_finalsubmission_2 =merged_test_prediction_finalsubmission.drop(['ConfirmedCases'], axis=1)
Final_submission = pd.concat([merged_test_prediction_finalsubmission_2, merged_test_prediction_finalsubmission_fatality_2], axis=1)
Final_submission.rename(columns={'Id':'ForecastId', 'ConfirmedCases new':'ConfirmedCases', 'Fatalities new':'Fatalities'}, 
                 inplace=True)
Final_submission.head()

# %% [code]
Final_submission.to_csv('submission.csv', index=False)