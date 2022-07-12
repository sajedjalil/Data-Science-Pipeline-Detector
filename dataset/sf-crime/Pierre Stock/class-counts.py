from pylab import *
from collections import Counter

#read the training dataset
from pandas import read_csv
df = read_csv("../input/train.csv")

#get the frequencies of the crime categories
categories=Counter(df.Category).most_common()
keys=[e[0] for e in categories]
values=[e[1] for e in categories]
 
#plot the histogram
pos = arange(len(keys))[::-1]+.5    # the bar centers on the y axis
plt.figure(figsize=(23,10))
barh(pos,values, align='center')
yticks(pos, keys)
plt.tick_params(axis='x',which='both',top='off')  
plt.tick_params(axis='y',which='both',right='off') 
plt.tick_params(axis='y',which='both',left='off') 
title('Training set class counts', fontsize=16)
plt.savefig("classs_counts.pdf",format="pdf")