import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

train = pd.read_csv("../input/train.csv", parse_dates=["datetime"])
train["hour"] = pd.DatetimeIndex(train['datetime']).hour
train["month"] = pd.DatetimeIndex(train['datetime']).month
train["temp"] =  train.temp*9.0/5.0+32.0
train["temp_jittered"] = train.temp + np.random.randn(len(train))
train["hour_jittered"] = train.hour + np.random.randn(len(train))
train["atemp"] = train.atemp*9.0/5.0+32.0
train["count"] = np.float64(train["count"])
print(train.head())
print("Train Shape: %d" % train.shape[0])

plt.figure()
n, bins, h = plt.hist(train.temp, bins=25, histtype='stepfilled')
plt.setp(h, facecolor='#53cfff')
plt.xlabel("Temperature (°F)")
plt.ylabel("Frequency")
plt.title("Temperature Distribution in Washington DC")
plt.savefig("1_hist.png")

weekend = train[(train.workingday==0) & (train.holiday==0)]
print("Weekend Samples: %d" % weekend.shape[0])
weekend_noon = weekend[weekend.hour==12]
print("Weekend at noon samples: %d" % weekend_noon.shape[0])

plt.figure()
weekend_noon.plot(kind='scatter', x='humidity', y='count', c=weekend_noon.temp)
plt.savefig("2_scatter.png")

plt.figure()
train.plot(kind='scatter', x='temp_jittered', y='hour_jittered', c='month')
plt.savefig("3_scatter.png")

plt.figure()
sns.regplot("hour_jittered", "temp_jittered", data=train)
plt.savefig("4_scatter.png")

plt.figure()
sns.pairplot(data=weekend_noon[["temp", "month", "humidity", "count"]], hue="count")
plt.savefig("5_pair.png")