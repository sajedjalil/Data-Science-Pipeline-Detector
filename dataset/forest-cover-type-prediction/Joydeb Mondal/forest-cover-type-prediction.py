import pandas as pd
from sklearn import ensemble
from sklearn import cross_validation
from sklearn import metrics

# Load the training and test data sets
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Create numpy arrays for use with scikit-learn
train_X = train.drop(['Id','Cover_Type'],axis=1).values
train_y = train.Cover_Type.values
test_X = test.drop('Id',axis=1).values

# Split the training set into training and validation sets
X,X_,y,y_ = cross_validation.train_test_split(train_X,train_y,test_size=0.1)

# Train and predict with the random forest classifier
rf = ensemble.RandomForestClassifier()
rf.fit(X,y)
y_rf = rf.predict(X_)
print (metrics.classification_report(y_,y_rf))
print (metrics.accuracy_score(y_,y_rf))

# Retrain with entire training set and predict test set.
rf.fit(train_X,train_y)
y_test_rf = rf.predict(test_X)

# Write to CSV
pd.DataFrame({'Id':test.Id.values,'Cover_Type':y_test_rf})\
            .sort_index(ascending=False,axis=1).to_csv('sampleSubmission.csv',index=False)



