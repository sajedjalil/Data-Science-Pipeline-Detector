import pandas as pd
import sklearn as sk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from scipy.stats.stats import pearsonr  

#############################################################################################################
#read data
train = pd.read_csv("../input/train.csv", parse_dates=["datetime"])
train["hour"] = pd.DatetimeIndex(train['datetime']).hour
train["month"] = pd.DatetimeIndex(train['datetime']).month
train["temp"] =  train.temp*9.0/5.0+32.0
train["atemp"] = train.atemp*9.0/5.0+32.0
print(train.corr())

# visualize the relationship between the features and the response using scatterplots for the total demand
fig, axs = plt.subplots(2, 3, sharey=True)
train.plot(kind='scatter', x='temp', y='count', ax=axs[0, 0], figsize=(16, 8), color='red')
train.plot(kind='scatter', x='atemp', y='count', ax=axs[0, 1], color='cyan')
train.plot(kind='scatter', x='humidity', y='count', ax=axs[0, 2], color='magenta')
train.plot(kind='scatter', x='windspeed', y='count', ax=axs[1, 0], color='yellow')
train.plot(kind='scatter', x='month', y='count', ax=axs[1, 1], color='blue')
train.plot(kind='scatter', x='hour', y='count', ax=axs[1, 2], color='green')
plt.savefig("datasetTotalExploration.png")


# visualize the relationship between the features and the response using scatterplots for the casual demand
fig, axs = plt.subplots(2, 3, sharey=True)
train.plot(kind='scatter', x='temp', y='casual', ax=axs[0, 0], figsize=(16, 8),  color='red')
train.plot(kind='scatter', x='atemp', y='casual', ax=axs[0, 1],  color='cyan')
train.plot(kind='scatter', x='humidity', y='casual', ax=axs[0, 2],  color='magenta')
train.plot(kind='scatter', x='windspeed', y='casual', ax=axs[1, 0],  color='yellow')
train.plot(kind='scatter', x='month', y='casual', ax=axs[1, 1],  color='blue')
train.plot(kind='scatter', x='hour', y='casual', ax=axs[1, 2],  color='green')
plt.savefig("datasetCasualExploration.png")

# visualize the relationship between the features and the response using scatterplots for the registered demand
fig, axs = plt.subplots(2, 3, sharey=True)
train.plot(kind='scatter', x='temp', y='registered', ax=axs[0, 0], figsize=(16, 8),  color='red') 
train.plot(kind='scatter', x='atemp', y='registered', ax=axs[0, 1],  color='cyan')
train.plot(kind='scatter', x='humidity', y='registered', ax=axs[0, 2],  color='magenta')
train.plot(kind='scatter', x='windspeed', y='registered', ax=axs[1, 0],  color='yellow')
train.plot(kind='scatter', x='month', y='registered', ax=axs[1, 1],  color='blue')
train.plot(kind='scatter', x='hour', y='registered', ax=axs[1, 2],  color='green')
plt.savefig("datasetRegisteredExploration.png")

# visualize the relationship between the features and the response using scatterplots monthly
fig, axs = plt.subplots(3,1, figsize=(10, 6), sharey=True)
#train.plot(kind='scatter', x='month', y='temp', ax=axs[0, 0], figsize=(12, 6),  color='red')
train.plot(kind='scatter', x='atemp', y='month', ax=axs[0], color='red')
train.plot(kind='scatter', x='humidity', y='month', ax=axs[1],  color='cyan')
train.plot(kind='scatter', x='windspeed', y='month', ax=axs[2],  color='magenta')
plt.savefig("datasetMonthExploration1.png")

fig, axs = plt.subplots(3,1, figsize=(10, 6), sharey=True)
train.plot(kind='scatter', x='count', y='month', ax=axs[0],  color='yellow')
train.plot(kind='scatter', x='casual', y='month', ax =axs[1],  color='blue')
train.plot(kind='scatter', x='registered', y='month', ax =axs[2],  color='green')
plt.savefig("datasetMonthExploration2.png")

#compute correlation between features
print(train.corr())
plt.figure()
plt.matshow(train.corr())
plt.colorbar()
plt.savefig("corrMatrix.png")

#compute pearson correlation for highly correlated features
corrPtemp = pearsonr(train.temp, train.atemp)
print(corrPtemp)

corrPWH = pearsonr(train.weather, train.humidity)
print(corrPWH)


#plt.figure()
#n, bins, h = plt.hist(train.temp, bins=25, histtype='stepfilled')
#plt.setp(h, facecolor='#53cfff', alpha=0.75)
#plt.xlabel("Temperature (°F)")
#plt.ylabel("Frequency")
#plt.title("Temperature Distribution in Washington DC")
#plt.savefig("temperature_distribution_in_washington_dc.png")

#print('\nSummary of train dataset:\n')
#print(train.describe())
#print('\nSummary of test dataset:\n')
#print(test.describe())