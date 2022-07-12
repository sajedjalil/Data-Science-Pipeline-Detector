#load libraries
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import zipfile
#%matplotlib inline

#load the dataset
z=zipfile.ZipFile('../input/train.csv.zip')
sf_df=pd.read_csv(z.open('train.csv'))
#sf_df = pd.read_csv("train.csv")
sf_df.info()
sf_df.columns#columns used
sf_df.head(5)

#############################
#############EDA#############
#############################

#frequency count for Category
sf_df.Category.value_counts().plot(kind="bar")
#most occurring crime= LARCENY/THEFT

#Cross-tabulate Category and PdDistrict
sf_df_crosstab = pd.crosstab(sf_df.PdDistrict,sf_df.Category,margins=True)
del sf_df_crosstab['All']#delete All column
sf_df_crosstab = sf_df_crosstab.ix[:-1]#delete last row (All)

#but we need to visualize it more clearly. build a heatmap
#set the labels
column_labels = list(sf_df_crosstab.columns.values)
row_labels = sf_df_crosstab.index.values.tolist()
#plot to a heatmap using matplotlib - (cannot install seaborn library)
fig,ax = plt.subplots()
heatmap = ax.pcolor(sf_df_crosstab,cmap=plt.cm.Blues)
#format
fig = plt.gcf()
fig.set_size_inches(15,5)
#turn off the frame
ax.set_frame_on(False)
# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(sf_df_crosstab.shape[0])+0.5, minor=False)
ax.set_xticks(np.arange(sf_df_crosstab.shape[1])+0.5, minor=False)
# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(column_labels, minor=False)
ax.set_yticklabels(row_labels, minor=False)
#rotate
plt.xticks(rotation=90)
#remove gridlines
ax.grid(False)
# Turn off all the ticks
ax = plt.gca()
for t in ax.xaxis.get_major_ticks(): 
    t.tick1On = False 
    t.tick2On = False 
for t in ax.yaxis.get_major_ticks(): 
    t.tick1On = False 
    t.tick2On = False  
plt.show()    
####
    
##TIME/DATE
#12 years worth of crime data
sf_df["Dates"].min()#start date: 2003-01-06 00:01:00
sf_df["Dates"].max()#end date: 2015-05-13 23:53:00
#extract year and month from "Dates" column and build into new columns
sf_df["Dates"] = pd.to_datetime(sf_df["Dates"])#convert col to datetime format
sf_df["Year"],sf_df["Month"] = sf_df['Dates'].apply(lambda x: x.year), sf_df['Dates'].apply(lambda x: x.month)
sf_df.head()

#Cross-tabulate Category and Year
sf_df_crosstab_dt = pd.crosstab(sf_df.Category,sf_df.Year,margins=True)
del sf_df_crosstab_dt['All']#delete All column
sf_df_crosstab_dt = sf_df_crosstab_dt.ix[:-1]#delete last row (All)

#but we need to visualize it more clearly. build a heatmap
#set the labels
column_labels_dt = list(sf_df_crosstab_dt.columns.values)
row_labels_dt = sf_df_crosstab_dt.index.values.tolist()
#plot to a heatmap using matplotlib - (cannot install seaborn library)
fig,ax = plt.subplots()
heatmap = ax.pcolor(sf_df_crosstab_dt,cmap=plt.cm.Blues)
#format
fig = plt.gcf()
fig.set_size_inches(5,10)
#turn off the frame
ax.set_frame_on(False)
# put the major ticks at the middle of each cell
ax.set_yticks(np.arange(sf_df_crosstab_dt.shape[0])+0.5, minor=False)
ax.set_xticks(np.arange(sf_df_crosstab_dt.shape[1])+0.5, minor=False)
# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticklabels(column_labels_dt, minor=False)
ax.set_yticklabels(row_labels_dt, minor=False)
#rotate
plt.xticks(rotation=90)
#remove gridlines
ax.grid(False)
# Turn off all the ticks
ax = plt.gca()
for t in ax.xaxis.get_major_ticks(): 
    t.tick1On = False 
    t.tick2On = False 
for t in ax.yaxis.get_major_ticks(): 
    t.tick1On = False 
    t.tick2On = False
plt.show()    