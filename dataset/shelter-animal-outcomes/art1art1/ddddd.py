# -*- coding: utf-8 -*-

import pandas as pd                                     # biblioteka analizująca dane wejściowe
import numpy as np                                      # biblioteka umożliwiająca zaawansowane działania matematyczne
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

# funckja przygotowująca parametry do przetworzenia
def prepare_data(data, train):
    # stworzenie parametru "czy zwierze ma imię?"
    data['HasName'] = data['Name'].fillna(0)
    data.loc[data['HasName'] != 0,"HasName"] = 1
    data['HasName'] = data['HasName'].astype(int)

    # stworzenie parametru "typ zwierzęcia"
    data['AnimalType'] = data['AnimalType'].map({'Cat':0,'Dog':1})

    if(train):
        # stworzenie parametru "co się stało ze zwierzęciem?"
        data.drop(['AnimalID','OutcomeSubtype'],axis=1, inplace=True)
        data['OutcomeType'] = data['OutcomeType'].map({'Return_to_owner':4, 'Euthanasia':3, 'Adoption':0, 'Transfer':5, 'Died':2})

    # stworzenie parametru "rasa i wysterylizowanie"
    gender = {'Neutered Male':1, 'Spayed Female':2, 'Intact Male':3, 'Intact Female':4, 'Unknown':5, np.nan:0}
    data['SexuponOutcome'] = data['SexuponOutcome'].map(gender)

    # funkcja licząca ilość dni
    def agetodays(x):
        try:
            y = x.split()
        except:
            return None 
        if 'year' in y[1]:
            return float(y[0]) * 365
        elif 'month' in y[1]:
            return float(y[0]) * (365/12)
        elif 'week' in y[1]:
            return float(y[0]) * 7
        elif 'day' in y[1]:
            return float(y[0])

    # obliczanie parametru wieku zwierząt w dniach
    data['AgeInDays'] = data['AgeuponOutcome'].map(agetodays)
    data.loc[(data['AgeInDays'].isnull()),'AgeInDays'] = data['AgeInDays'].median()

    data['Year'] = data['DateTime'].str[:4].astype(int)
    data['Month'] = data['DateTime'].str[5:7].astype(int)
    data['Day'] = data['DateTime'].str[8:10].astype(int)
    data['Hour'] = data['DateTime'].str[11:13].astype(int)
    data['Minute'] = data['DateTime'].str[14:16].astype(int)

    # stworzenie parametru "czy ma imię + jakiej jest płci + sterylizacja"
    data['Name+Gender'] = data['HasName'] + data['SexuponOutcome']

    # stworzenie parametru "rasa + jakiej jest płci + sterylizacja"
    data['Type+Gender'] = data['AnimalType'] + data['SexuponOutcome']

    # stworzenie parametru "czy jest mieszańcem"
    data['IsMix'] = data['Breed'].str.contains('mix',case=False).astype(int)

    # zwrócenie danych
    return data.drop(['AgeuponOutcome','Name','Breed','Color','DateTime'],axis=1)

# funkcja tworząca drzewo najlepszych parametrów określających zbiór danych treningowych
def best_params(data):
    rfc = RandomForestClassifier()
    param_grid = { 
        'n_estimators': [50, 400],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
    CV_rfc.fit(data[0::,1::],data[0::,0])
    return CV_rfc.best_params_

if __name__ == "__main__":
    # ścieżki danych wejściowych
    in_file_train = '../input/train.csv'
    in_file_test = '../input/test.csv'

    # ładowanie danych treningowych i testowych z plików CSV
    print("Loading data...\n")
    pd_train = pd.read_csv(in_file_train)
    pd_test = pd.read_csv(in_file_test)

    # przetwarzanie danych wejściowych
    print("Preparing data...\n")
    pd_train = prepare_data(pd_train, True)
    pd_test = prepare_data(pd_test, False)

    pd_test.drop('ID',inplace=True,axis=1)

    train = pd_train.values
    test = pd_test.values

    # sprawdzanie najlepszych parametrów pobranych danych trenigowych
    print("Calculating best case params...\n")
    print(best_params(train))

    # przewidywanie wyników przez drzewo
    print("Predicting... \n")
    forest = RandomForestClassifier(n_estimators = 400, max_features='auto')
    forest = forest.fit(train[0::,1::],train[0::,0])
    predictions = forest.predict_proba(test)

    # przygotowanie danych wynikowych
    output = pd.DataFrame(predictions,columns=['Adoption','Died','Euthanasia','Return_to_owner','Transfer'])
    output.columns.names = ['ID']
    output.index.names = ['ID']
    output.index += 1

    # zapisanie pliku wynikowego
    print("Writing predictions.csv\n")
    #print(output)
    output.to_csv('predictions.csv')

    print("Done.\n")