# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 00:55:09 2017

@author: Hsemu
"""
from __future__ import print_function
from collections import defaultdict
import numpy as np
import datetime
import csv
from operator import itemgetter
import sys
import pandas as pd
from sklearn import ensemble, preprocessing
import xgboost as xgb 
from sklearn.tree import DecisionTreeClassifier
import math
from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

os.system("ls ../input")

# Load dataset
train = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sampleSubmission.csv')
weather = pd.read_csv('../input/weather.csv')

#Converting to Date Object
def getdate(datacol):
    return pd.to_datetime(datacol, format="%Y-%m-%d")

#Getting the Year
def getyear(datacol):
    return datacol['Date'].dt.year

#Getting day of Year
def getdayofyear(datacol):
    return datacol['Date'].dt.dayofyear

#Getting the Duplicated Values
def Duplicatedfeature(datacol):
    datacol['Freqcount'] = datacol.groupby(['Trap','year','day_of_year','Latitude','Longitude'])['Species'].transform(pd.Series.value_counts)
    return datacol['Freqcount']

#Getting Freq by Traps
def Duplicatedtraps(datacol):
    datacol['Freqcounttraps'] = datacol.groupby(['Trap','year'])['day_of_year'].transform(pd.Series.value_counts)
    return datacol['Freqcounttraps']

#Get Frequency of the Count of traps
def trapsfrequency(datacol):
    datacol['trapsfrequency'] = datacol.groupby(['Trap','year','day_of_year'])['Freqcounttraps'].transform(pd.Series.value_counts)
    return datacol['trapsfrequency']

#Create New feature for count of traps 
def trapsfrequencyequal(datacol,number):
    datacol['trapsfrequencyequal_'+str(number)] = np.where(datacol['Freqcounttraps']==number,datacol['trapsfrequency'], 0)
    return datacol['trapsfrequencyequal_'+str(number)]

#Create New feature for count of rows of traps atleast 2
def trapsfrequencygreater(datacol,number):
    datacol['trapsfrequencygreater_'+str(number)] = np.where(datacol['Freqcounttraps']>number,datacol['trapsfrequency'], 0)
    return datacol['trapsfrequencygreater_'+str(number)]

##Get count by currendate and check for atleast 2
def trapsfrequencybycurrentdate(datacol):
    datacol['trapsfrequencycurrentdate'] = datacol.groupby(['Date'])['Trap'].transform(pd.Series.value_counts)
    return datacol['trapsfrequencycurrentdate']

def trapsfrequencycurrentdategreater1(datacol):
    datacol['trapsfrequencycurrentdategreater1'] = np.where(datacol['trapsfrequencycurrentdate']>1,datacol['trapsfrequencycurrentdate'], 0)
    return datacol['trapsfrequencycurrentdategreater1']

def Preprocessing(train):
    train['Date']=getdate(train['Date'])
    train['year']=getyear(train).astype('int64')
    train['day_of_year']=getdayofyear(train).astype('int64')
    train['Freqcount']=Duplicatedfeature(train).astype('int64')
    train['Freqcounttraps']=Duplicatedtraps(train).astype('int64')
    train['trapsfrequency']=trapsfrequency(train).astype('int64')
    trapsfrequencyequal(train,2).astype('int64')
    trapsfrequencyequal(train,3).astype('int64')
    trapsfrequencyequal(train,4).astype('int64')
    trapsfrequencyequal(train,5).astype('int64')
    trapsfrequencyequal(train,6).astype('int64')
    train['sumoftrapsfrequencyequal']=train.trapsfrequencyequal_2+train.trapsfrequencyequal_3+train.trapsfrequencyequal_4+train.trapsfrequencyequal_5+train.trapsfrequencyequal_6
    trapsfrequencygreater(train,1).astype('int64')
    trapsfrequencybycurrentdate(train).astype('int64')
    trapsfrequencycurrentdategreater1(train).astype('int64')
    return train


train=Preprocessing(train)
test=Preprocessing(test)

# Convert categorical data to numbers
lbl = preprocessing.LabelEncoder()
lbl.fit(list(train['Species'].values) + list(test['Species'].values))
train['Species'] = lbl.transform(train['Species'].values)
test['Species'] = lbl.transform(test['Species'].values)

lbl.fit(list(train['Street'].values) + list(test['Street'].values))
train['Street'] = lbl.transform(train['Street'].values)
test['Street'] = lbl.transform(test['Street'].values)

lbl.fit(list(train['Trap'].values) + list(test['Trap'].values))
train['Trap'] = lbl.transform(train['Trap'].values)
test['Trap'] = lbl.transform(test['Trap'].values)

def precip(text):
    TRACE = 1e-3
    text = text.strip()
    if text == "M":
        return None
    if text == "-":
        return None
    if text == "T":
        return TRACE
    return float(text)

def impute_missing_weather_station_values(weather):
    # Stupid simple
    for k, v in weather.items():
        if v[0] is None:
            v[0] = v[1]
        elif v[1] is None:
            v[1] = v[0]
        for k1 in v[0]:
            if v[0][k1] is None:
                v[0][k1] = v[1][k1]
        for k1 in v[1]:
            if v[1][k1] is None:
                v[1][k1] = v[0][k1]
    
def load_weather():
    weather = {}
    for line in csv.DictReader(open("../input/weather.csv")):
        for name, converter in {"Date" : date,
                                "Tmax" : float,"Tmin" : float,"Tavg" : float,
                                "DewPoint" : float, "WetBulb" : float,
                                "PrecipTotal" : precip,"Sunrise" : precip,"Sunset" : precip,
                                "Depart" : float, "Heat" : precip,"Cool" : precip,
                                "ResultSpeed" : float,"ResultDir" : float,"AvgSpeed" : float,
                                "StnPressure" : float, "SeaLevel" : float}.items():
            x = line[name].strip()
            line[name] = converter(x) if (x != "M") else None
        station = int(line["Station"]) - 1
        assert station in [0,1]
        dt = line["Date"]
        if dt not in weather:
            weather[dt] = [None, None]
        assert weather[dt][station] is None, "duplicate weather reading {0}:{1}".format(dt, station)
        weather[dt][station] = line
    impute_missing_weather_station_values(weather)        
    return weather
    
    
def load_training():
    training=train
    return training
    
def load_testing():
    testing=test
    return testing
    
    
def closest_station(lat, longi):
    # Chicago is small enough that we can treat coordinates as rectangular.
    stations = np.array([[41.995, -87.933],
                         [41.786, -87.752]])
    loc = np.array([lat, longi])
    deltas = stations - loc[None, :]
    dist2 = (deltas**2).sum(1)
    return np.argmin(dist2)
       
def normalize(X, mean=None, std=None):
    count = X.shape[1]
    if mean is None:
        mean = np.nanmean(X, axis=0)
    for i in range(count):
        X[np.isnan(X[:,i]), i] = mean[i]
    if std is None:
        std = np.std(X, axis=0)
    for i in range(count):
        X[:,i] = (X[:,i] - mean[i]) / std[i]
    return mean, std
    
def scaled_count(record):
    SCALE = 9.0
    if "NumMosquitos" not in record:
        # This is test data
        return 1
    return int(np.ceil(record["NumMosquitos"] / SCALE))
    
    
def assemble_X(base, weather):
    X = []
    for b in base:
        date = b["Date"]
        lat, longi = b["Latitude"], b["Longitude"]
        case = [date.year, date.month, date.day, date.weekday(), lat, longi]
        # Look at a selection of past weather values
        for days_ago in [0,1,3,5,8,12]:
            day = date - datetime.timedelta(days=days_ago)
            for obs in ["Tmax","Tmin","Tavg","DewPoint","WetBulb","PrecipTotal","Depart","Sunrise","Sunset","Cool","ResultSpeed","ResultDir"]:
                station = closest_station(lat, longi)
                case.append(weather[day][station][obs])
        # Specify which mosquitos are present
        species_vector = [float(x) for x in species_map[b["Species"]]]
        case.extend(species_vector)
        # Weight each observation by the number of mosquitos seen. Test data
        # Doesn't have this column, so in that case use 1. This accidentally
        # Takes into account multiple entries that result from >50 mosquitos
        # on one day. 
        for repeat in range(scaled_count(b)):
            X.append(case)    
    X = np.asarray(X, dtype=np.float32)
    return X
    
def assemble_y(base):
    y = []
    for b in base:
        present = b["WnvPresent"]
        for repeat in range(scaled_count(b)):
            y.append(present)    
    return np.asarray(y, dtype=np.int32).reshape(-1,1)


class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))

def train():
    weather = load_weather()
    training = load_training()
    
    X = assemble_X(training, weather)
    mean, std = normalize(X)
    y = assemble_y(training)
        
    input_size = len(X[0])
    
    learning_rate = theano.shared(np.float32(0.1))
    
    net = NeuralNet(
    layers=[  
        ('input', InputLayer),
         ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', DenseLayer),
        ('dropout2', DropoutLayer),
        ('output', DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, input_size), 
    hidden1_num_units=400, 
    dropout1_p=0.4,
    hidden2_num_units=200, 
    dropout2_p=0.4,
    output_nonlinearity=sigmoid, 
    output_num_units=1, 

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=learning_rate,
    update_momentum=0.9,
    
    # Decay the learning rate
    on_epoch_finished=[
            AdjustVariable(learning_rate, target=0, half_life=4),
            ],

    # This is silly, but we don't want a stratified K-Fold here
    # To compensate we need to pass in the y_tensor_type and the loss.
    regression=True,
    y_tensor_type = T.imatrix,
    objective_loss_function = binary_crossentropy,
     
    max_epochs=60, 
    eval_size=0.1,
    verbose=1,
    )

    X, y = shuffle(X, y, random_state=123)
    net.fit(X, y)
    
    _, X_valid, _, y_valid = train_test_split(X, y)
    probas = net.predict_proba(X_valid)[:,0]
    print("ROC score", metrics.roc_auc_score(y_valid, probas))

    return net, mean, std     
    

def submit(net, mean, std):
    weather = load_weather()
    testing = load_testing()
    X = assemble_X(testing, weather) 
    normalize(X, mean, std)
    predictions = net.predict_proba(X)[:,0]    
    #
    out = csv.writer(open("sampleSubmission.csv", "w"))
    out.writerow(["Id","WnvPresent"])
    for row, p in zip(testing, predictions):
        out.writerow([row["Id"], p])


if __name__ == "__main__":
    net, mean, std = train()
    submit(net, mean, std)



    