# Costa Rican Household Poverty Level Prediction

"""
Created on Sat Jul 21 16:15:16 2018

@author: top max
"""
# import the libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load the dataset
train_data = pd.read_csv('..//input/train.csv')
test_data  = pd.read_csv('..//input/test.csv')



print ("Train Dataset: Rows, Columns: ", train_data.shape)
# take a look at the data information
train_data.info()    
train_data.describe()
train_data.columns[train_data.dtypes == object]
train_data.head()


# Groupby the household and figure out the number of unique values
all_equal = train_data.groupby('idhogar')['Target'].apply(lambda x: x.nunique() == 1)

# Households where targets are not all equal
not_equal = all_equal[all_equal != True]
print('There are {} households where the family members do not all have the same target.'.format(len(not_equal)))

# Iterate through each household
for household in not_equal.index:
    # Find the correct label (for the head of household)
    true_target = int(train_data[(train_data['idhogar'] == household) & (train_data['parentesco1'] == 1.0)]['Target'])
    
# Set the correct label for all members in the household
    train_data.loc[train_data['idhogar'] == household, 'Target'] = true_target
    
# deal with missing data (for non object types)
# first: check the columns that contains missing values (nan)
train_data.isna().any()
train_data['rez_esc'].isnull().sum() # number of missing data in this column

print ("Top Columns having missing values")
missmap = train_data.isnull().sum().to_frame().sort_values(0, ascending = False)
missmap.head()


from sklearn.preprocessing import Imputer
imputer1 = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0, )

train_data['v2a1'] = imputer1.fit_transform(train_data[['v2a1']]).ravel()
train_data['v18q1'] = imputer1.fit_transform(train_data[['v18q1']]).ravel()

train_data['rez_esc'] = train_data['rez_esc'].fillna(0.0)
train_data['meaneduc'] = train_data['meaneduc'].fillna(0.0)
train_data['SQBmeaned'] = train_data['SQBmeaned'].fillna(0.0)


train_data.drop(['v18q1'], axis = 1,inplace = True)
train_data.drop(['rez_esc'], axis = 1,inplace = True)
train_data.drop(['v2a1'], axis = 1,inplace = True)



# for object data types, we have 3 column with missing data: dependency, edjefe, edjefa
train_data['dependency'] = train_data['dependency'].replace('yes',1)
train_data['dependency'] = train_data['dependency'].replace('no',0)
train_data['edjefe'] = train_data['edjefe'].replace('yes',1)
train_data['edjefe'] = train_data['edjefe'].replace('no',0)
train_data['edjefa'] = train_data['edjefa'].replace('yes',1)
train_data['edjefa'] = train_data['edjefa'].replace('no',0)




missmap_test = test_data.isnull().sum().to_frame().sort_values(0, ascending = False)
missmap_test.head()

test_data.drop(['v18q1'], axis = 1,inplace = True)
test_data.drop(['rez_esc'], axis = 1,inplace = True)
test_data.drop(['v2a1'], axis = 1,inplace = True)
test_data['meaneduc'] = imputer1.fit_transform(test_data[['meaneduc']]).ravel()
test_data['SQBmeaned'] = imputer1.fit_transform(test_data[['SQBmeaned']]).ravel()


# for object data types, we have 3 column with missing data: dependency, edjefe, edjefa
test_data['dependency'] = test_data['dependency'].replace('yes',1)
test_data['dependency'] = test_data['dependency'].replace('no',0)
test_data['edjefe'] = test_data['edjefe'].replace('yes',1)
test_data['edjefe'] = test_data['edjefe'].replace('no',0)
test_data['edjefa'] = test_data['edjefa'].replace('yes',1)
test_data['edjefa'] = test_data['edjefa'].replace('no',0)



# arrange the dataset
X_train = train_data.iloc[:, 1:139]
y_train = train_data.iloc[:, 139]
#drop the 'idhogar' feature
X_train.drop(['idhogar'], axis = 1,inplace = True)

X_test = test_data.iloc[:, 1:139]
#drop the 'idhogar' feature
X_test.drop(['idhogar'], axis = 1,inplace = True)


# Run Naive Bayes algorithm
'''from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)'''

'''from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric= 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)'''

from sklearn.ensemble import RandomForestClassifier
classifier =  RandomForestClassifier(n_estimators=5, criterion= 'entropy')
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# create submission file 

Y_id = test_data['Id'] 
sbt = pd.DataFrame({'Id':Y_id, 'Target': y_pred})
sbt.to_csv('submission.csv', index=False)