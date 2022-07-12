import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pandas as pd

train = pd.read_csv("../input/train.csv", parse_dates=["datetime"])
train["hour"] = pd.DatetimeIndex(train['datetime']).hour


plt.figure()
n, bins, h = plt.hist(train.temp, bins=25, histtype='stepfilled')
plt.setp(h, facecolor='#53cfff', alpha=0.75)
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")
plt.title("Temperature Distribution in Washington DC")
plt.savefig("temperature_distribution_in_washington_dc.png")
print (n)
print (h)
