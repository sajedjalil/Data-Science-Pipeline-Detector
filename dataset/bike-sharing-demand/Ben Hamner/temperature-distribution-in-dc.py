import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", parse_dates=["datetime"])
train["hour"] = pd.DatetimeIndex(train['datetime']).hour
train["temp"] =  train.temp*9.0/5.0+32.0
train["atemp"] = train.atemp*9.0/5.0+32.0

plt.figure()
n, bins, h = plt.hist(train.temp, bins=25, histtype='stepfilled')
plt.setp(h, facecolor='#53cfff', alpha=0.75)
plt.xlabel("Temperature (°F)")
plt.ylabel("Frequency")
plt.title("Temperature Distribution in Washington DC")
plt.savefig("temperature_distribution_in_washington_dc.png")
