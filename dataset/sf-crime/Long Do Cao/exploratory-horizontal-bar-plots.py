import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile


###LOAD DATA
dir="../input/"
z = zipfile.ZipFile(dir+"train.csv.zip")
df = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])




###PROCESSING
categories = ["Category","DayOfWeek","PdDistrict"]  #plot only these categories
for cat in categories:
    grouped = df.groupby(cat)
    count = grouped.count()


### PLOT
    plt.figure()
    count.sort(columns="Dates",ascending=1)["Dates"].plot(kind="barh") #take arbitrarly the first column for plot
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig("barh_"+cat)

