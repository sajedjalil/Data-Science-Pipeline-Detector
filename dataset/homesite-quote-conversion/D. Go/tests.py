import pandas as pd
import numpy as np
import datetime as dt
from sklearn import preprocessing

print(' -> Starting')

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print(' -> Loading OK')

#train.set_index(['QuoteNumber'])
#train.ix[train.PersonalField10A==-1, ['QuoteConversion_Flag']] = -1

#print(pd.get_dummies(train, columns=['GeographicField64']))

#scaler = preprocessing.StandardScaler().fit(train['Field8'])
#train['Field8'] = scaler.transform(train['Field8'])
#print(train['Field8'])

#normalizer = preprocessing.Normalizer().fit(train['Field8'])
#print(normalizer.transform(train['Field8']))

#train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
#train['before_june'] = train['Date'].apply(lambda x: int(str(x)[5:7])) < 6
#print(train['before_june'])

#cov_fields = ['CoverageField1A', 'CoverageField1B', 'CoverageField2A', 'CoverageField2B',
# 'CoverageField3A', 'CoverageField3B', 'CoverageField4A', 'CoverageField4B',
# 'CoverageField5A', 'CoverageField5B', 'CoverageField6A', 'CoverageField6B']
#train['mean_cov_fields'] = train[cov_fields].mean(axis=1)

#train['mean_cov_fields'] = train[cov_fields].mean(axis=1)
#train['min_cov_fields'] = train[cov_fields].min(axis=1)
#train['max_cov_fields'] = train[cov_fields].max(axis=1)
#train['median_cov_fields'] = train[cov_fields].median(axis=1)

#train["sum_fields_na"] = np.sum(train == -1, axis = 1)
#train["sum_fields_0"] = np.sum(train == 0, axis = 1)
#print(train["sum_fields_na"])
#print(train["sum_fields_0"])

#print(np.sum(train["sum_fields_na"], axis=0)/len(train))
#print(int(np.sum(train["sum_fields_0"], axis=0)/len(train)))

#### Groupir ####

train = train.fillna(-1)
test  = test.fillna(-1)
train['type'] = 'train'
test['type']  = 'test'
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
test['Date']  = pd.to_datetime(pd.Series( test['Original_Quote_Date']))
test['QuoteConversion_Flag'] = 0

#datas = pd.concat([train, test])
datas = train

print(' -> Datas len: '+str(len(datas)))

# Sort by date
datas.sort_values('Original_Quote_Date', inplace=True)

personal_cols = [col for col in datas.columns if col.startswith('PersonalField')]
grouped_personal = datas.groupby(personal_cols).count()
print(' -> Grouped personal len: '+str(len(grouped_personal)))

geographic_cols = [col for col in datas.columns if col.startswith('GeographicField')]
grouped_geographic = datas.groupby(geographic_cols).count()
print(' -> Grouped geographic len: '+str(len(grouped_geographic)))

sales_cols = [col for col in datas.columns if col.startswith('SalesField')]
grouped_sales = datas.groupby(sales_cols).count()
print(' -> Grouped sales len: '+str(len(grouped_sales)))

properties_cols = [col for col in datas.columns if col.startswith('PropertyField')]
grouped_properties = datas.groupby(properties_cols).count()
print(' -> Grouped properties len: '+str(len(grouped_properties)))

coverage_cols = [col for col in datas.columns if col.startswith('CoverageField')]
grouped_coverage = datas.groupby(coverage_cols).count()
print(' -> Grouped coverage len: '+str(len(grouped_coverage)))

field_cols = [col for col in datas.columns if col.startswith('Field')]
grouped_field = datas.groupby(field_cols).count()
print(' -> Grouped field len: '+str(len(grouped_field)))

# Derivating prefixed fields columns
def custom_derivated_fields(datas, prefix):
    cus_prefix = 'Custom_'+prefix
    cols = [col for col in datas.columns if col.startswith(prefix)]
    g = datas.groupby(cols)
    datas.set_index(cols, inplace=True)
    datas[cus_prefix+'_cols_index']            = g['QuoteNumber'].min()
    datas[cus_prefix+'_cols_count']            = g.size()
    datas[cus_prefix+'_cols_nb_convertion']    = g['QuoteConversion_Flag'].sum()
    datas[cus_prefix+'_cols_cumulative']       = g.cumcount()
    datas[cus_prefix+'_cols_min_date']         = g['Date'].min()
    datas[cus_prefix+'_cols_diff_min_date']    = (datas['Date'] - datas[cus_prefix+'_cols_min_date']).apply(lambda x: x.astype('timedelta64[D]')/np.timedelta64(1, 'D'))

    datas.drop(cus_prefix+'_cols_min_date', axis=1, inplace=True)
    
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(datas[cus_prefix+'_cols_index'].values))
    datas[cus_prefix+'_cols_index'] = lbl.transform(list(datas[cus_prefix+'_cols_index'].values))

    datas.reset_index(inplace=True)
    
    return datas

# Cumulative count
custom_derivated_fields(datas, 'PersonalField')
custom_derivated_fields(datas, 'GeographicField')
custom_derivated_fields(datas, 'SalesField')
custom_derivated_fields(datas, 'PropertyField')
custom_derivated_fields(datas, 'CoverageField')
custom_derivated_fields(datas, 'Field')

# Nb quotes by day
cols = ['Original_Quote_Date']
g = datas.groupby(cols)
datas.set_index(cols, inplace=True)
datas['Custom_Date_count']           = g.size()
datas['Custom_Date_converted']       = g['QuoteConversion_Flag'].sum()

print(datas)

# Export
#train.to_csv('train_customized.csv', index=False)
