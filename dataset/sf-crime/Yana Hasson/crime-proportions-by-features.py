# import libraries
import numpy as np
import scipy as scp
import matplotlib as mpl
import sklearn as skl
import matplotlib.pyplot as plt
import pandas
import re

# extract year, day and hour and create "Year", "Month", "Hour" columns in train_data_frame
train_data_frame = pandas.read_csv("../input/train.csv", sep=",", quotechar='"')
test = pandas.read_csv("../input/test.csv", sep=",", quotechar='"')


def parse_date(date):
    """
    Extracts Year, Month, and Hour out of initial Dates 
    :param date: Date as is formated in the training set column Dates
    :return: iterator (map object) ??
    """
    mo = re.search(r'^([0-9]{4})-([0-9]{2})-[0-9]{2}\s+([0-9]{2}):[0-9]{2}:[0-9]{2}$', date)
    return map(int, (mo.group(1), mo.group(2), mo.group(3)))
    
# Extract 'Year', 'Month' and 'Hour' columns for later use
train_data_frame['Year'], train_data_frame['Month'], train_data_frame['Hour'] = zip(*train_data_frame.loc[:, 'Dates'].map(parse_date))
test['Year'], test['Month'], test['Hour'] = zip(*test.loc[:, 'Dates'].map(parse_date))

# clean data : delete data outliers from train_data_frame
train_data_frame_ret = train_data_frame[train_data_frame.Y < 38]
print(train_data_frame_ret.shape)
def proportionCrimeCategory(data, discreteParam):
    """
    Extracts proportion of Crimes in the data that  
    :param data: input dataframe
    :param discreteParam: name of column of dataframe 
    :return normedtable: Series
            column 1 : index of month
            column 2 : sum of crimes during this month/total numbers of crimes
    """
    by_param = data.groupby([discreteParam, 'Category'])
    #apllying .size() allows to extract number of instances for each Crime Category in each discreteParam
    table = by_param.size()
    #puts it as a 2D table with number of occurences per discreteParam and Crime Category
    d2table = table.unstack()
    #d2table.sum(1) returns the number of crimes in all crime categories per discreteParam
    #1 is for the number of the axe on which the sum is done. Here :Ccrime Category
    normedtable = d2table.div(d2table.sum(1), axis=0)
    return normedtable

discreteParamList = ['DayOfWeek', 'PdDistrict', 'Year','Month','Hour']
fig1, axes1 = plt.subplots(len(discreteParamList),1) #creates a 3x1 blank plot
for i in range(len(discreteParamList)): #now we fill in the subplots
    param = discreteParamList[i]
    table = proportionCrimeCategory(train_data_frame_ret, param)

    ax = axes1[i]
    #create plot title
    ax.set_title("Categrories of crime by %s" % discreteParamList[i])
    ax.title.set_fontsize(20)
    #choose colormap (memo : looks ok : none, prism_r)
    #stacked = true, allows to stack the different categories into one bar
    table.plot(kind='barh', stacked=True, ax=axes1[i],figsize=(20,30), color=['lightblue','blue','pink','magenta', 'purple', 'red', 'darkred', 'white', 'black','turquoise'])
    #Puts legend only on second graph and pushes the legend out of plot
    if i==1:
        #how to move legends : http://matplotlib.org/examples/pylab_examples/legend_demo3.html
        ax.legend(bbox_to_anchor=(1.2, 2))

    else:
        ax.legend_.remove()

plt.savefig("proportions.pdf",format="pdf")
