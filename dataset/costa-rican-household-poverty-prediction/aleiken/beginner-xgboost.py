# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
# Any results you write to the current directory are saved as output.
null_feature = ['v2a1', 'v18q1', 'rez_esc', 'meaneduc', 'SQBmeaned']



#idhogar
not_equal_df = df_train.groupby('idhogar')['Target'].nunique()
not_equal_idhogar = not_equal_df[not_equal_df>1].index.values
for idhogar in not_equal_idhogar:
    true_target = df_train[(df_train['idhogar'] == idhogar)&(df_train['parentesco1'] == 1)].Target
    true_target = int(true_target)
    df_train.loc[(df_train['idhogar']== idhogar) & (df_train['parentesco1'] == 0), 'Target'] = true_target

#remove no parentesco1
household = df_train.groupby('idhogar')['parentesco1'].sum()
null_parentesco1 = household[household == 0].index.values
for idhogar in null_parentesco1:
    df_train.drop(df_train[df_train['idhogar'] == idhogar].index, axis=0, inplace=True)


num_train = df_train.shape[0]
target = df_train['Target']
mapping = {1: 4, 2: 4, 3: 7, 4: 1}
weight = df_train['Target'].replace(mapping)
test_ID = df_test['Id']
df_train.drop(['Target'],inplace=True,axis=1)
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
all_data.drop(['Id'],inplace = True,axis=1)


def process_data(df):

    df['idhogar'] = pd.factorize(df['idhogar'])[0]

    df['poor'] = (df['cielorazo'] == 0) + df['pisonotiene'] + df['abastaguano'] + df['noelec'] + df['sanitario1']
    df.loc[df['poor'] > 2, 'poor'] = 2

    df.loc[(df['v2a1'].isnull()) & (df['tipovivi1'] == 1), 'v2a1'] = 0
    df['miss_v2a1'] = df['v2a1'].isnull()

    df['v18q1'].fillna(0, inplace=True)

    df.loc[((df['age'] > 17) | (df['age'] < 7)) & (df['rez_esc'].isnull()), 'rez_esc'] = 0
    df['miss_rezesc'] = df['rez_esc'].isnull()
    df.loc[df['rez_esc'] > 5, 'rez_esc'] = 5

    df['meaneduc'].fillna(df['meaneduc'].mode()[0], inplace=True)

    df['SQBmeaned'].fillna((df['meaneduc'].mode()[0]) ** 2, inplace=True)

    df.loc[df['dependency'] == 'no', 'dependency'] = 0
    df.loc[df['dependency'] == 'yes', 'dependency'] = 1

    df['dependency'] = df['dependency'].astype('float64')

    df.loc[df['edjefe'] == 'no', 'edjefe'] = 0
    df.loc[df['edjefe'] == 'yes', 'edjefe'] = 1

    df.loc[df['edjefa'] == 'no', 'edjefa'] = 0
    df.loc[df['edjefa'] == 'yes', 'edjefa'] = 1

    df.loc[df['male'] == 1, 'sex'] = 1
    df.loc[df['male'] == 0, 'sex'] = 0
    df.drop(['female', 'male'], axis=1, inplace=True)


    bins = [0, 10, 20, 40, 60, 80, 100]
    age = pd.cut(df['age'], bins)
    df['age_bin'] = age
    df['age_bin'] = pd.factorize(df['age_bin'])[0]
    # df.drop(['age'], axis=1, inplace=True)


    df['mobilephone_per_man'] = df['qmobilephone'] / df['tamviv']
    df['rooms_per_man'] = df['rooms'] / df['tamviv']
    df['bedrooms_per_man'] = df['bedrooms'] / df['tamviv']

    df.drop(columns=['tamhog','hhsize','r4t3'],inplace=True)

    #electricity level
    df.loc[df['public'] == 1, 'electricity'] = 1
    df.loc[df['planpri'] == 1, 'electricity'] = 1
    df.loc[df['noelec'] == 1, 'electricity'] = 2
    df.loc[df['coopele'] == 1, 'electricity'] = 3
    df['electricity'].fillna(4, inplace=True)
    df['electricity'] = df['electricity'].astype('uint8')
    df.drop(columns=['public', 'planpri', 'noelec','coopele'], inplace=True)

    instlevel = []
    for i in range(1, 10):
        df.loc[df['instlevel%d' % (i)] == 1, 'education'] = i
        instlevel.append('instlevel%d' % (i))

    df['education'].fillna(2, inplace=True)
    df['education'] = df['education'].astype('uint8')
    df.drop(columns=instlevel, inplace=True)

    feature = ['age', 'meaneduc', 'escolari','education']
    algorithm = ['min', 'max', 'sum', 'mean', 'std']
    group_data = df.groupby('idhogar')[feature].agg(algorithm).reset_index()
    new_columns = ['idhogar']
    for f in feature:
        for a in algorithm:
            new_columns.append('{feature}_{algorithm}'.format(feature=f, algorithm=a))
    group_data.columns = new_columns
    df = pd.merge(df, group_data, on=['idhogar'], how='left')

    df['wall'] = np.argmax(np.array(df[['epared1', 'epared2', 'epared3']]),axis = 1)
    df.drop(columns=['epared1', 'epared2', 'epared3'], inplace=True)

    df['floor'] = np.argmax(np.array(df[['eviv1', 'eviv2', 'eviv3']]), axis=1)
    df.drop(columns=['eviv1', 'eviv2', 'eviv3'], inplace=True)

    df['roof'] = np.argmax(np.array(df[['etecho1', 'etecho2', 'etecho3']]), axis=1)
    df.drop(columns=['etecho1', 'etecho2', 'etecho3'], inplace=True)

    df['house_quality'] = df['wall']+df['floor']+df['roof']
    
    feature_0 = ['v14a', 'escolari', 'rez_esc', 'paredfibras', 'paredother', 'pisoother',
     'pisonatur' ,'techocane', 'techootro' ,'abastaguafuera', 'sanitario1',
     'elimbasu2', 'elimbasu3' ,'elimbasu5' ,'elimbasu6' ,'estadocivil1',
     'estadocivil2', 'estadocivil4', 'estadocivil5', 'estadocivil6',
     'estadocivil7', 'parentesco1' ,'parentesco2' ,'parentesco3' ,'parentesco4',
     'parentesco5', 'parentesco6', 'parentesco7' ,'parentesco8' ,'parentesco9',
     'parentesco10' ,'parentesco11' ,'parentesco12' ,'lugar1', 'age', 'SQBescolari',
     'SQBage', 'agesq', 'miss_rezesc' ,'sex' ,'age_bin', 'electricity', 'education',
     'meaneduc_max', 'meaneduc_mean' ,'meaneduc_std']
    df.drop(columns=feature_0, inplace=True)
    df.drop(columns=['idhogar'], inplace=True)

 
    train = df[:num_train]
    test = df[num_train:]
    return train,test
    
    
train, test = process_data(all_data)
xgb_model = xgb.XGBClassifier(objective='multi:softmax', n_jobs=-1, num_class = 4,
                              random_state = 1021,silent=1,learning_rate=0.1,min_child_weight = 5,
                              max_depth=7,colsample_bytree= 0.85,subsample=0.85,
                              reg_alpha= 0.7,reg_lambda=0.2)

xgb_model.fit(train.values,target.values,sample_weight=weight.values)
predict = xgb_model.predict(test.values)

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['Target'] = predict
sub.to_csv('submission.csv', index=False)