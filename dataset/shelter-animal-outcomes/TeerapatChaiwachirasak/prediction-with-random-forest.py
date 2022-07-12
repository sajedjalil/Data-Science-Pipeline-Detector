# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
print("HE")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


#Data preprocessing
train['Year'] = 0
train['Month'] = 0
train['Day'] = 0
train['DayOfWeek'] = ''
train['Hour'] = 0

def convert_age_today(row):
    age_string = row['AgeuponOutcome']
    [age,unit] = age_string.split(" ")
    unit = unit.lower()
    if("day" in unit):
        if age=='0': return 1
        return int(age)
    if("week" in unit):
        if(age)=='0': return 7
        return int(age)*7
    elif("month" in unit):
        if(age)=='0': return 30
        return int(age) * 4*7
    elif("year" in unit):
        if(age)=='0': return 365
        return int(age) * 4*12*7

#Extract DateTime and Color
train['DateTime'] = pd.to_datetime(train['DateTime'])
test['DateTime'] = pd.to_datetime(test['DateTime'])
for index,row in train.iterrows():
    dateTime = row['DateTime']
    train.set_value(index,'Year',dateTime.year)
    train.set_value(index,'Month',dateTime.month)
    train.set_value(index,'Day',dateTime.day)
    train.set_value(index,'DayOfWeek',dateTime.dayofweek)
    train.set_value(index,'Hour',dateTime.hour)
    
    color = row['Color']
    #Seperate color by space so that we can use CountVectorizer
    train.set_value(index,'Color'," ".join(color.split("/")))
    
for index,row in test.iterrows():
    dateTime = row['DateTime']
    test.set_value(index,'Year',dateTime.year)
    test.set_value(index,'Month',dateTime.month)
    test.set_value(index,'Day',dateTime.day)
    test.set_value(index,'DayOfWeek',dateTime.dayofweek)
    test.set_value(index,'Hour',dateTime.hour)
    
    color = row['Color']
    #Seperate color by space so that we can use CountVectorizer
    test.set_value(index,'Color'," ".join(color.split("/")))
    
#Handle Age Upon Outcome
train['AgeuponOutcome'] = train['AgeuponOutcome'].fillna('0 weeks')
train['AgeInDays'] = train.apply(convert_age_today, axis=1)
test['AgeuponOutcome'] = test['AgeuponOutcome'].fillna('0 weeks')
test['AgeInDays'] = test.apply(convert_age_today, axis=1)


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
transformed_outcome = le.fit_transform(train['OutcomeType'])
train['OutcomeType'] = transformed_outcome
train['AnimalType'] = le.fit_transform(train['AnimalType'])
test['AnimalType'] = le.transform(test['AnimalType'])
train['SexuponOutcome'] =train['SexuponOutcome'].fillna("Unknown")
test['SexuponOutcome'] =test['SexuponOutcome'].fillna("Unknown")


#Handle animals with no name

train['Name'] = train['Name'].fillna('NoName')
test['Name'] = test['Name'].fillna('NoName')

selectCol = [ 'Name', 
       'AnimalType', 'SexuponOutcome', 'AgeuponOutcome', 'Breed', 'Color',
       'Year', 'Month', 'Day', 'DayOfWeek', 'Hour']
target = ['OutcomeType']
X = train[selectCol]
y = train[target]



from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
'''
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
'''

# Create sparse matrix from color, sex, and breed
color_cv = CountVectorizer()
Xtrain_colors = color_cv.fit_transform(X_train['Color']).toarray()    #Main Feature
Xtest_colors = color_cv.transform(X_test['Color']).toarray()
sex_cv = CountVectorizer(min_df=100)
Xtrain_SexuponOutcome = sex_cv.fit_transform(X_train['SexuponOutcome']).toarray() #Main Feature
Xtest_SexuponOutcome = sex_cv.transform(X_test['SexuponOutcome']).toarray() #Main Feature
breed_cv = CountVectorizer(min_df=100)
Xtrain_BreedTypeInfo = breed_cv.fit_transform(X_train['Breed']).toarray()
Xtest_BreedTypeInfo = breed_cv.transform(X_test['Breed']).toarray()
Xtrain_sparse = np.concatenate((Xtrain_colors,Xtrain_SexuponOutcome,Xtrain_BreedTypeInfo),axis=1)
Xtest_sparse = np.concatenate((Xtest_colors,Xtest_SexuponOutcome,Xtest_BreedTypeInfo),axis=1)
selectCol = ['AnimalType', 'Year', 'Month', 'Day', 'DayOfWeek', 'Hour']
X_train_selected_col = X_train[selectCol]
X_test_selected_col = X_test[selectCol]
X_train = np.concatenate((Xtrain_sparse,X_train_selected_col),axis=1)
X_test = np.concatenate((Xtest_sparse,X_test_selected_col),axis=1)

X_train = X_train.astype(np.float)
X_test = X_test.astype(np.float)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
y_train = y_train.values.ravel()

'''
# Fitting Naive Bayes to the Training set (accuracy : 0.73 , f1 : 0.77186)
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
'''
# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 30, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_prob = classifier.predict_proba(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()




# CreateSubmission
#Handle CV 
submission_color = color_cv.transform(test['Color']).toarray()
submission_sex = sex_cv.transform(test['SexuponOutcome']).toarray() #Main Feature
submission_breed = breed_cv.transform(test['Breed']).toarray()
submission_sparse = np.concatenate((submission_color,submission_sex,submission_breed),axis=1)
X_validation = test[selectCol]
X_validation = np.concatenate((submission_sparse,X_validation),axis=1)
X_validation = X_validation.astype(np.float)

X_validation = sc.transform(X_validation)
y_pred = classifier.predict(X_validation)
y_prob = classifier.predict_proba(X_validation)
#Convert our answer to csv
ID = test['ID']

columns = ['ID','Adoption','Died','Euthanasia','Return_to_owner','Transfer']
predict_df = pd.DataFrame(
    {'ID': ID,
     'Adoption': y_prob[:,0],
     'Died': y_prob[:,1],
     'Euthanasia': y_prob[:,2],
     'Return_to_owner': y_prob[:,3],
     'Transfer': y_prob[:,4]
     },  columns=columns)
predict_df.to_csv('animals_outcome_prediction.csv',index=False)


