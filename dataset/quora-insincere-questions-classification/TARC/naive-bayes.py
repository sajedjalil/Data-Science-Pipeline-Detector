import pandas as pd
import csv
import re
import sklearn

from nltk.corpus import stopwords

import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

stopWords = set(stopwords.words('english'))

def read_dict_file(csvfile):
    with open(csvfile) as csv_file:
        reader = csv.reader(csv_file)
        mydict = dict(reader)

    return mydict

def write_dict_file(csvfile, dict1,dict_names):
    with open(csvfile, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(dict_names)
        for key, value in dict1.items():
            writer.writerow([key, value])

#see stats about 
def number_of_classes(data):
    rows0, rows1 = 0,0
    for _,row in data.iterrows():
        c = row["target"]
        if c == 0:
            rows0 = rows0 + 1
        else:
            rows1 = rows1 + 1 

    return rows0, rows1
    
def main_stats(data, percentage):

    test_limit = int(percentage * data.shape[0])

    dict1 = {}
    dict0 = {}
    rows0, rows1 = 0,0

    for index, row in data.iloc[:test_limit].iterrows():

        words= []

        tokens = row["question_text"].split(" ")

        for w in tokens:
            if w != '':
                w.lower()
                if w not in stopWords:
                    words.append(w)

        if row["target"] == 1:
            for w in words:
                rows1 = rows1 + 1
                word = w
                word = re.sub('[^A-Za-z]', '', word)
                word = porter_stemmer.stem(word)
                word = wordnet_lemmatizer.lemmatize(word)
                try:
                    dict1[word] = dict1[word] + 1 
                except:
                    dict1[word] = 1
        else:
            for w in words:
                rows0 = rows0 + 1
                word = w
                word = re.sub('[^A-Za-z]', '', word)
                word = porter_stemmer.stem(word)
                word = wordnet_lemmatizer.lemmatize(word)
                try:
                    dict0[word] = dict0[word] + 1 
                except:
                    dict0[word] = 1

    for key,value in dict1.items():
        dict1[key] = float(value/rows1)

    for key,value in dict0.items():
        dict0[key] = float(value/rows0)


    return dict1, dict0, rows0, rows1

def test(dict1, dict0, p0, p1, test_set, missing_prob):

    correct = 0
    wrong = 0

    for index, row in test_set.iterrows():

        words= []

        tokens = row["question_text"].split(" ")

        for w in tokens:
            word = re.sub('[^A-Za-z]', '', w)
            if word != '':
                word.lower()
                word = porter_stemmer.stem(word)
                word = wordnet_lemmatizer.lemmatize(word)
                words.append(word)
                

        prob0, prob1 = 10, 1

        for word in words:
            if word not in dict0:
                dict0[word] = missing_prob
            if word not in dict1:
                dict1[word] = missing_prob            
            prob0 = prob0 * dict0[word]
            prob1 = prob1 * dict1[word]

        prob0 = prob0 * p0
        prob1 = prob1 * p1

        pred = None
        if prob0 < prob1:
            pred = 1
        else:
            pred = 0 

        if pred  == row["target"]:
            correct = correct + 1
        else:
            wrong = wrong  + 1


    return correct, wrong


def write_predictions(dict1, dict0, p0, p1, test_set, missing_prob):

    submission = {}

    for index, row in test_set.iterrows():

        words= []

        tokens = row["question_text"].split(" ")

        for w in tokens:
            word = re.sub('[^A-Za-z]', '', w)
            if word != '':
                if w not in stopWords:
                    word.lower()
                    word = porter_stemmer.stem(w)
                    words.append(word)

        prob0, prob1 = 1, 1

        for word in words:
            if word not in dict0:
                dict0[word] = p0
            if word not in dict1:
                dict1[word] = p1           
            prob0 = prob0 * dict0[word]
            prob1 = prob1 * dict1[word]

        prob0 = prob0 * p0
        prob1 = prob1 * p1

        pred = None
        
        if prob0 < prob1:
            pred = 1
        else:
            pred = 0

        submission[row["qid"]] = pred

        dict_names = ["qid","prediction"]

    write_dict_file("submission.csv", submission,dict_names)


if __name__ =="__main__":
    

    #data = pd.read_csv("questions/train.csv")
    
    csv_name_train  = "../input/train.csv"
    csv_name_test = "../input/test.csv"

    """
    dict1, dict0 = main_stats(csv_name)

    #write dict files
    write_dict_file("dict1.csv",dict1)
    write_dict_file("dict0.csv",dict0)
    """

    #get number of classes
    percentage = 1.0
    data = pd.read_csv(csv_name_train)
    dict1, dict0, rows0, rows1 = main_stats(data, percentage)
    #test_limit = int(percentage * data.shape[0])
    #write_dict_file("dict1.csv",dict1)
    #write_dict_file("dict0.csv",dict0)
    p0 = rows0/(rows0 + rows1)
    p1 = rows1/(rows0 + rows1)
    missing_prob = float(1/data.shape[0])
    data = pd.read_csv(csv_name_test)
    #print(test(dict1, dict0, p0, p1, data, missing_prob))

    write_predictions(dict1, dict0, p0, p1, data, missing_prob)
