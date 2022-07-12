# vectorized error calc
def rmsle(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn import model_selection, preprocessing
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn import preprocessing 

# Reading all three files
train = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])
test = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])
macro = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

# Removing outliers for full_sq
#train = train[(train.full_sq<=265) & (train.full_sq>6)]

# Merging macro data with train and test
train = pd.merge(train, macro, how='left', on='timestamp')
test = pd.merge(test, macro, how='left', on='timestamp')

# normalize prize feature
train["price_doc"] = np.log1p(train["price_doc"])
# store it as Y
Y_train = train["price_doc"]
# Dropping price column
train.drop("price_doc", axis=1, inplace=True)

# Merging both dataframes
all_data = pd.concat((train.loc[:,'id':'apartment_fund_sqm'],test.loc[:,'id':'apartment_fund_sqm']))

all_data["month_year"]=all_data.timestamp.dt.month + all_data.timestamp.dt.year * 100

# Add month-year
month_year = (all_data.timestamp.dt.month + all_data.timestamp.dt.year * 100)
month_year_cnt_map = month_year.value_counts().to_dict()
all_data['month_year_cnt'] = month_year.map(month_year_cnt_map)

# Add week-year count
week_year = (all_data.timestamp.dt.weekofyear + all_data.timestamp.dt.year * 100)
week_year_cnt_map = week_year.value_counts().to_dict()
all_data['week_year_cnt'] = week_year.map(week_year_cnt_map)

# Add month and day-of-week
all_data['month'] = all_data.timestamp.dt.month
all_data['dow'] = all_data.timestamp.dt.dayofweek

all_data['apartment_name'] = pd.factorize(all_data.sub_area + all_data['metro_km_avto'].astype(str))[0]

# Remove timestamp column (may overfit the model in train)
all_data.drop(['timestamp'], axis=1, inplace=True)

# Cleaning full_sq
all_data.loc[((all_data["full_sq"]<=6) | (all_data["full_sq"]>300)) & (all_data["life_sq"]>=6) & (all_data["life_sq"]<300),"full_sq"]=all_data[((all_data["full_sq"]<=6) | (all_data["full_sq"]>300)) & (all_data["life_sq"]>=6) & (all_data["life_sq"]<300)].life_sq

#convert objects / non-numeric data types into numeric
for f in all_data.columns:
    if all_data[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(all_data[f].values)) 
        all_data[f] = lbl.transform(list(all_data[f].values))

all_data.drop(['area_m',
'raion_popul',
'green_zone_part',
'indust_part',
'children_preschool',
'preschool_quota',
'preschool_education_centers_raion',
'children_school',
'school_quota',
'school_education_centers_raion',
'school_education_centers_top_20_raion',
'hospital_beds_raion',
'healthcare_centers_raion',
'university_top_20_raion',
'sport_objects_raion',
'additional_education_raion',
'culture_objects_top_25',
'culture_objects_top_25_raion',
'shopping_centers_raion',
'office_raion',
'thermal_power_plant_raion',
'incineration_raion',
'oil_chemistry_raion',
'radiation_raion',
'railroad_terminal_raion',
'big_market_raion',
'nuclear_reactor_raion',
'detention_facility_raion',
'full_all',
'male_f',
'female_f',
'young_all',
'young_male',
'young_female',
'work_all',
'work_male',
'work_female',
'ekder_all',
'ekder_male',
'ekder_female',
'0_6_all',
'0_6_male',
'0_6_female',
'7_14_all',
'7_14_male',
'7_14_female',
'0_17_all',
'0_17_male',
'0_17_female',
'16_29_all',
'16_29_male',
'16_29_female',
'0_13_all',
'0_13_male',
'0_13_female',
'raion_build_count_with_material_info',
'build_count_block',
'build_count_wood',
'build_count_frame',
'build_count_brick',
'build_count_monolith',
'build_count_panel',
'build_count_foam',
'build_count_slag',
'build_count_mix',
'raion_build_count_with_builddate_info',
'build_count_before_1920',
'build_count_1921-1945',
'build_count_1946-1970',
'build_count_1971-1995',
'build_count_after_1995'], axis=1, inplace=True)

#log transform skewed numeric features:
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]

model_lasso = LassoCV(alphas = [0.0005]).fit(X_train, Y_train)
Y_pred = np.expm1(model_lasso.predict(X_test))
Y_train_pred = np.expm1(model_lasso.predict(X_train))

print(rmsle(Y_train_pred,np.expm1(Y_train)))

submission = pd.DataFrame({"id": test["id"],"price_doc": Y_pred})
submission.to_csv('submission_00005.csv', index=False)