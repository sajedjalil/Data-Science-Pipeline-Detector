#script to generate itemsets

#cjaas  July 2015


import csv
import os
from datetime import datetime
import numpy as np
#from progressbar import ProgressBar
import pylab as pl
import matplotlib.pyplot as plt

def FindCategory(cat):
    if cat == 1:
        return "ARSON"
    elif cat == 2:
        return "ASSAULT"
    elif cat == 3:
        return "BAD CHECKS"
    elif cat == 4:
        return "BRIBERY"
    elif cat == 5:
        return "BURGLARY"
    elif cat == 6:
        return "DISORDERLY CONDUCT"
    elif cat == 7:
        return "DRIVING UNDER THE INFLUENCE"
    elif cat == 8:
        return "DRUG/NARCOTIC"
    elif cat == 9:
        return "DRUNKENNESS"
    elif cat == 10:
        return "EMBEZZLEMENT"
    elif cat == 11:
        return "EXTORTION"
    elif cat == 12:
        return "FAMILY OFFENSES"
    elif cat == 13:
        return "FORGERY/COUNTERFEITING"
    elif cat == 14:
        return "FRAUD"
    elif cat == 15:
        return "GAMBLING"
    elif cat == 16:
        return "KIDNAPPING"
    elif cat == 17:
        return "LARCENY/THEFT"
    elif cat == 18:
        return "LIQUOR LAWS"
    elif cat == 19:
        return "LOITERING"
    elif cat == 20:
        return "MISSING PERSON"
    elif cat == 21:
        return "NON-CRIMINAL"
    elif cat == 22:
        return "OTHER OFFENSES"
    elif cat == 23:
        return "PORNOGRAPHY/OBSCENE MAT"
    elif cat == 24:
        return "PROSTITUTION"
    elif cat == 25:
        return "RECOVERED VEHICLE"
    elif cat == 26:
        return "ROBBERY"
    elif cat == 27:
        return "RUNAWAY"
    elif cat == 28:
        return "SECONDARY CODES"
    elif cat == 29:
        return "SEX OFFENSES NON FORCIBLE"
    elif cat == 30:
        return "SEX OFFENSES FORCIBLE"
    elif cat == 31:
        return "STOLEN PROPERTY"
    elif cat == 32:
        return "SUICIDE"
    elif cat == 33:
        return "SUSPICIOUS OCC"
    elif cat == 34:
        return "TREA"
    elif cat == 35:
        return "TRESPASS"
    elif cat == 36:
        return "VANDALISM"
    elif cat == 37 :
        return"VEHICLE THEFT"
    elif cat == 38:
        return "WARRANTS"
    elif cat == 39:
        return "WEAPON LAWS"
    else:
        return 0

def EnumCategory(cat):
    if str(cat)== "ARSON":
        return 1
    elif str(cat)== "ASSAULT":
        return 2
    elif str(cat)== "BAD CHECKS":
        return 3
    elif str(cat)== "BRIBERY":
        return 4
    elif str(cat)== "BURGLARY":
        return 5
    elif str(cat)== "DISORDERLY CONDUCT":
        return 6
    elif str(cat)== "DRIVING UNDER THE INFLUENCE":
        return 7
    elif str(cat)== "DRUG/NARCOTIC":
        return 8
    elif str(cat)== "DRUNKENNESS":
        return 9
    elif str(cat)== "EMBEZZLEMENT":
        return 10
    elif str(cat)== "EXTORTION":
        return 11
    elif str(cat)== "FAMILY OFFENSES":
        return 12
    elif str(cat)== "FORGERY/COUNTERFEITING":
        return 13
    elif str(cat)== "FRAUD":
        return 14
    elif str(cat)== "GAMBLING":
        return 15
    elif str(cat)== "KIDNAPPING":
        return 16
    elif str(cat)== "LARCENY/THEFT":
        return 17
    elif str(cat)== "LIQUOR LAWS":
        return 18
    elif str(cat)== "LOITERING":
        return 19
    elif str(cat)== "MISSING PERSON":
        return 20
    elif str(cat)== "NON-CRIMINAL":
        return 21
    elif str(cat)== "OTHER OFFENSES":
        return 22
    elif str(cat)== "PORNOGRAPHY/OBSCENE MAT":
        return 23
    elif str(cat)== "PROSTITUTION":
        return 24
    elif str(cat)== "RECOVERED VEHICLE":
        return 25
    elif str(cat)== "ROBBERY":
        return 26
    elif str(cat)== "RUNAWAY":
        return 27
    elif str(cat)== "SECONDARY CODES":
        return 28
    elif str(cat)== "SEX OFFENSES NON FORCIBLE":
        return 29
    elif str(cat)== "SEX OFFENSES FORCIBLE":
        return 30
    elif str(cat)== "STOLEN PROPERTY":
        return 31
    elif str(cat)== "SUICIDE":
        return 32
    elif str(cat)== "SUSPICIOUS OCC":
        return 33
    elif str(cat)== "TREA":
        return 34
    elif str(cat)== "TRESPASS":
        return 35
    elif str(cat)== "VANDALISM":
        return 36
    elif str(cat)== "VEHICLE THEFT":
        return 37
    elif str(cat)== "WARRANTS":
        return 38
    elif str(cat)== "WEAPON LAWS":
        return 39
    else:
        return 0



#### ------------- Part 1: Generate Itemsets ------------- ####

#read in training data
training_file =  "../input/train.csv"
train_data = []
with open(training_file, 'rt') as tF:
    train_reader = csv.reader(tF, delimiter=',', quotechar='"')
    next(train_reader, None)   #skip header
    for line in train_reader:  #iterate over incidents
        train_data.append(line)


#make sure training data is sorted by date, time and address
train_data_np = np.array(train_data)
train_data_np = sorted(train_data_np,key=lambda x: x[0])
train_data_np = sorted(train_data_np,key=lambda x: x[6])
train_data = train_data_np


#initialise variables for generating itemsets from training data
item_sets = []         #to hold list of itemsets
item_sets.append([])   #append empty list of itemsets for first itemset
previous_address = ''  #temp variable to compare addresses of current and previous incident
previous_date = ''     #temp variable to compare datetime of current and previous incident
itemset_index = -1     #start at itemset 0, so initialise to -1

#iterate over incidents in training data to mine the itemsets in the data
#print 'Generating itemsets...'
#pbar = ProgressBar()
for line in train_data:
    date = datetime.strptime(line[0], '%Y-%m-%d %H:%M:%S')
    address = line[6]
    category = EnumCategory(line[1])
    #check if current incident happened at same time and place as the previous
    if (address == previous_address and previous_date == date):
        item_sets[itemset_index].append(category)  #add this incident to the same itemset as previous incident
    else:
        item_sets[itemset_index] = sorted(item_sets[itemset_index])  #sort previous itemset
        item_sets.append([])   #append new list to hold next itemset
        itemset_index += 1     #increment itemset index
        item_sets[itemset_index].append(category)   #append this latest crime to the new itemset
    previous_address = address
    previous_date = date


output_itemsets = sorted(item_sets)

#remove any empty itemsets
for itemset in item_sets:
    if(len(itemset) < 1):
        output_itemsets.remove(itemset)


#### ------------- Part 2: Investigate Itemsets ------------- ####
# Aims to answer the questions:
# 1. "If an incident itemset is of length n, how likely is it that it contains an incident of category q?"
# 2. "If an incident is of category q, how likely is it that that incident was contained in an itemset of length N?"

#set parameters
n_categories = 39       #number of crime categories
max_size_itemsets = 16   #maximum size of itemsets investigated (longest itemset in training data contains 16 items)

#initialise lists
total_count_itemsets = [0] * (max_size_itemsets + 1)  #count number of itemsets of particular lengths [(0), 1, 2, ..., maxSizeItemsets-1]
total_count_categories = [0] *(n_categories + 1)   #count total number of incidents of category [(0), 1, 2, ..., n_categories]
actual_items = []                   #list to hold the actual itemsets

#extend actual_items to hold one list of itemsets for each length of itemset [(0), 1, 2, ..., maxSizeItemsets-1]
for i in range(0,max_size_itemsets + 1):
    actual_items.append([])

#iterate over itemsets
for itemset in output_itemsets:
    n_items = len(itemset)
    total_count_itemsets[n_items] += 1
    actual_items[n_items].append(itemset)

#for each length of itemset, maintain a list of counts of incident categories [(0), 1, 2, ..., n_categories]
counts_categories = []
for i in range(0,max_size_itemsets + 1):
    counts_categories.append([0]*(n_categories+1))  #initialise count for each crime category

#for each incident category, maintain a list of counts of itemset length [(0), 1, 2, ..., max_size_itemset-1]
counts_itemsets = []
for i in range(0,n_categories+1):
    counts_itemsets.append([0]*(max_size_itemsets + 1))  #initialise count for each length of itemset

#populate counts
for number_items in range(0,max_size_itemsets + 1):
    for itemset in actual_items[number_items]:
        unique_entries = set(itemset)
        for category in unique_entries:
            counts_categories[number_items][int(category)] += 1
        for category in itemset:
            counts_itemsets[int(category)][number_items] += 1
            total_count_categories[int(category)] += 1


#### ------------- Part 3: Plot Results ------------- ####
# First answer question:
#   "If an incident itemset is of length L, how likely is it that it contains an incident of category q?"

#plot bar chart for each length of incident itemset
#print "Plotting results for each length of itemset..."
for i_length in range(1,max_size_itemsets+1):
    # Make a square figure and axes
    fig = pl.figure()
    ax = pl.subplot(111)

    #x-labels of bar chart = crime categories
    labels = ["ARSON", "ASSAULT", "BAD CHECKS", "BRIBERY", "BURGLARY", "DISORDERLY CONDUCT", \
            "DRIVING UNDER THE INFLUENCE", "DRUG/NARCOTIC","DRUNKENNESS", "EMBEZZLEMENT","EXTORTION", \
            "FAMILY OFFENSES","FORGERY/COUNTERFEITING","FRAUD", "GAMBLING", "KIDNAPPING", "LARCENY/THEFT", \
             "LIQUOR LAWS", "LOITERING", "MISSING PERSON", "NON-CRIMINAL", "OTHER OFFENSES", \
            "PORNOGRAPHY/OBSCENE MAT", "PROSTITUTION", "RECOVERED VEHICLE", "ROBBERY", "RUNAWAY", "SECONDARY CODES", \
            "SEX OFFENSES NON FORCIBLE", "SEX OFFENSES FORCIBLE", "STOLEN PROPERTY", "SUICIDE", "SUSPICIOUS OCC", \
            "TREA", "TRESPASS", "VANDALISM", "VEHICLE THEFT", "WARRANTS", "WEAPON LAWS"]


    n_labels = range(1,40)


    #find number of iLength-length itemsets containing at least one incident of each given crime category
    incidents = counts_categories[i_length][1:]

    #normalise by total number of itemsets of length iLength
    if(total_count_itemsets[i_length] > 0):
        for i_category in range(0,n_categories):
            incidents[i_category] = float(incidents[i_category]) / float(total_count_itemsets[i_length])

    #construct bar chart
    ax.bar(n_labels, incidents, width=1, align='center')
    #pl.title('Incidents in Itemsets of Length '+ str(i_length), bbox={'facecolor':'0.8', 'pad':5})
    pl.xlim([1,n_categories+1])
    pl.ylim([0.00001,1])
    pl.yscale('log')
    pl.xlabel('Incident category')
    pl.ylabel('Fraction of ' + str(i_length) +'-itemsets containing' + '\n' + ' at least one incident of given category')
    plt.xticks(n_labels, labels, rotation='vertical')
    plt.gcf().subplots_adjust(bottom=0.55)

    #save plot to file
    figName = 'distr_itemset_length_' + str(i_length) + '.png'
    pl.savefig(figName)
    pl.close()

# Secondly answer question:
#   "If an incident is of category q, how likely is it that it was part of an L-itemset?"

#plot bar chart for each length of incident itemset
#print "Plotting results for each incident category..."
for i_category in range(1,n_categories+1):
    # Make a square figure and axes
    fig = pl.figure()
    ax = pl.subplot(111)

    #x-labels of bar chart = length of itemsets
    labels = range(0,max_size_itemsets+1)

    #find number of iLength-length itemsets containing at least one incident of each given crime category
    incidents = counts_itemsets[i_category][:]

    #normalise by total number of itemsets of length iLength
    if(total_count_categories[i_category] > 0):
        for i_length in range(0,max_size_itemsets+1):
            incidents[i_length] = float(incidents[i_length]) / float(total_count_categories[i_category])

    #construct barchart
    ax.bar(labels, incidents, width=1, align='center')
    #pl.title('Incidents in Itemsets of Length '+ str(i_length), bbox={'facecolor':'0.8', 'pad':5})
    pl.xlim([1,max_size_itemsets+1])
    pl.ylim([0.00001,1])
    pl.yscale('log')
    pl.xlabel('Itemset length, L')
    pl.ylabel('Fraction of ' + str(FindCategory(i_category)) + ' incidents' + '\n' + 'being contained in an L-itemset')

    #save plot to file
    figName =  'distr_category_' + str(i_category) + '.png'
    pl.savefig(figName)
    pl.close()