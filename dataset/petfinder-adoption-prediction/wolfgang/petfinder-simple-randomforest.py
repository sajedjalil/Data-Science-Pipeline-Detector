
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier


def main():
    train = pd.read_csv("../input/train/train.csv")
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train[['Type', 'Age', 'Gender', 'Breed1', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Fee', 'VideoAmt', 'PhotoAmt', 'Quantity']].values, train['AdoptionSpeed'].values)
    
    # predict test data
    test = pd.read_csv('../input/test/test.csv')
    predictions = model.predict(test[['Type', 'Age', 'Gender', 'Breed1', 'Color1', 'Color2', 'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Fee', 'VideoAmt', 'PhotoAmt', 'Quantity']].values)
    print(predictions)
    
    submission = pd.read_csv("../input/test/sample_submission.csv")
    submission['AdoptionSpeed'] = predictions
    submission.to_csv("submission.csv", index=False)

if __name__ == '__main__':
    main()
    