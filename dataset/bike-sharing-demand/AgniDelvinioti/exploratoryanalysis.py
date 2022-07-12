import pandas as pd
import sklearn as sk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
from scipy.stats.stats import pearsonr  

############################################################################################################
#read data
train = pd.read_csv("../input/train.csv", parse_dates=["datetime"])
train["hour"] = pd.DatetimeIndex(train['datetime']).hour
train["month"] = pd.DatetimeIndex(train['datetime']).month
train["year"] = pd.DatetimeIndex(train['datetime']).year
train["temp"] =  train.temp*9.0/5.0+32.0
train["atemp"] = train.atemp*9.0/5.0+32.0

############################################################################################################
#compute correlation between features
feature_cols = ['hour', 'month', 'temp', 'atemp', 'humidity', 'windspeed', 'count', 'registered', 'casual']
trainFI = train[feature_cols]

print(trainFI.corr())
plt.figure()
plt.matshow(trainFI.corr())
plt.colorbar()
plt.savefig("corrMatrix.png")

#compute pearson correlation for highly correlated features
corrPtemp = pearsonr(train.temp, train.atemp)
print(corrPtemp)

corrPWH = pearsonr(train.weather, train.humidity)
print(corrPWH)

############################################################################################################
#exploratory analysis
import seaborn as sb; 
sb.set(style="ticks", color_codes=True)
sb_plot = sb.pairplot(train, hue="workingday", palette="husl",x_vars=['hour', 'month', 'atemp', 'temp', 'humidity', 'windspeed'], y_vars=["count", "registered","casual"])
sb_plot.savefig("exploratoryAnalysisGeneral.png")

sb_plot = sb.pairplot(train, hue="workingday", palette="husl",y_vars=['atemp', 'humidity', 'windspeed'], x_vars=["month"])
sb_plot.savefig("exploratoryAnalysisMonth.png")
