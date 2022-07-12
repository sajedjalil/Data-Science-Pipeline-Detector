import pandas
import numpy
import os
from datetime import datetime
import re
import matplotlib
import matplotlib.pyplot as plt
import zipfile

#replace with ur own location of file
z = zipfile.ZipFile('../input/train.csv.zip')
crime = pandas.read_csv(z.open("train.csv"))

targets = crime["Category"].unique().tolist()
#days = crime["DayOfWeek"].unique()
#print targets

crime_dict = {}
for crimes in targets:
    crime_dict[crimes] = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#print crime_dict
    
daydict = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}
    
for index,row in crime.iterrows():
    crime_dict[row["Category"]][daydict[row["DayOfWeek"]]]+= 1.0

final = []
total = 8780.50
for crimes in crime_dict:
    for i in range (0,7):
        crime_dict[crimes][i] /= total
    final.append(crime_dict[crimes])

#use following code to generate plot using pandas.plot alone
##chart = pandas.DataFrame(final).transpose()
##chart.plot(kind = 'barh',stacked= True,legend = False)
##matplotlib.pyplot.show()
#print crime_dict

# following code and related functions enable you to plot similar graph using dictionary and matplotlib
ind = numpy.arange(7)
days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
width = 0.30
count = 0
colors = ['b','g','r','c','m','y','k']
for crimes in crime_dict:
    if count == 0:
        plt.barh(ind,crime_dict[crimes],color = colors[count%7],align = 'center')
        prev = numpy.array(crime_dict[crimes])
    else:
        plt.barh(ind,crime_dict[crimes],color = colors[count%7],left = prev,align = 'center')
        prev = prev+numpy.array(crime_dict[crimes])
    count+=1
    
plt.yticks(ind,days)
plt.xlabel("Percentage")
plt.ylabel("day")
plt.title('Distribution of crimes according to day of week')
plt.savefig("DayvsCrime.png")
         