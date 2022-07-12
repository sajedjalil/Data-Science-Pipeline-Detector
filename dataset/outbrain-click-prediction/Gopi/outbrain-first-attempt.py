# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# get clicks_train, clicks_test, & events csv files as a DataFrame
clicks_train = pd.read_csv('../input/clicks_train.csv')
clicks_test  = pd.read_csv('../input/clicks_test.csv')
events_df    = pd.read_csv('../input/events.csv', usecols=['uuid', 'platform', 'geo_location'])
# ads_df          = pd.read_csv('../input/promoted_content.csv')
# documents_df    = pd.read_csv('../input/documents_meta.csv')
# categories_df   = pd.read_csv('../input/documents_categories.csv')


# preview the data
#clicks_train.head()

#clicks_train.info()
#print("----------------------------")
#clicks_test.info()
# Ads

# What's the frequency Vs the mean of each ad

# The Frequency(Count of occurrence for each value)
ads_freq = clicks_train['ad_id'].value_counts()

# The mean(The average of clicks)
ads_clicked = clicks_train[clicks_train['clicked'] == 1]['ad_id'].value_counts()
ads_average = ads_clicked.values / ads_freq[ads_clicked.index]
# Given the number of clicks for each ad
# We can show the important, the max and the min values
# This gives a clue about how values(count of clicks) is distributed
# For me, I would guess probably it's normally distributed, but, let's see

# Plot max, min values, & 2nd, 3rd quartile
#fig, (axis1) = plt.subplots(1,1,figsize=(12,5))
#sns.boxplot([ads_clicked], ax=axis1)
#axis1.set(xlabel='Frequency for count of clicks')
# Huummm, It seems most of the ad clicks lies between 1 and 10000, 
# and few of them lies after 10000, Isn't it?
# But, this doesn't clearly show the frequency for ad clicks
# So, Let's get deeper ...

# Plot frequency for clicks on ads

# And because there are many values(small) that just appeared a few times, 
# and few values(large) that appeared so much,
# Thus, we had to use Log to show all of them.
#fig, (axis1, axis2) = plt.subplots(2,1,figsize=(12,8))
#ads_clicked.plot(kind='hist',bins=50,log=True,ax=axis1)
#axis1.set(ylabel='Log10(Frequency)', xlabel='Count of Clicks')

# Plot the average of clicks and the standard deviation
# This is a huge std!. According to the empirical rule (given mean=66 & std=578):
# 68% of the clicks where between 66 - 578 and 66 + 578 clicks.
# 95% of the clicks where between 66 - 2 X (578) and 66 + 2 X (578) clicks.
# 98% of the clicks where between 66 - 3 X (578) and 66 + 3 X (578) clicks.
#Series(ads_clicked.mean()).plot(yerr=ads_clicked.std(),kind='bar',legend=False, ax=axis2)
# Now, we can also dive deeper, 
# and see the the percentage of ads that were clicked less than(or equal) X times?

#ads_perc = Series()
#for i in [2, 10, 50, 100, 1000, 5000]:
#    ads_perc[str(i)] = round((ads_clicked.values <= i).mean() * 100, 2)

#ax = ads_perc.plot(kind='bar', figsize=(12,3), colormap="summer")
#ax = ax.set(ylabel='Percentage', xlabel='Count of Clicks')
# Finally, it's time to show the actual clicks Vs views
# The frequency for count of clicks stops at almost 45000 click, 
# while there is a good presence for the freqeuncy for count of views after.

#fig, (axis1) = plt.subplots(1,1,figsize=(12,5))

#ads_clicked.name = 'Frequency for count of clicks'
#ads_freq.name =  'Frequency for count of views'

#ads_clicked.plot(kind='hist',bins=50,normed=True,log=True,color='indianred',alpha=0.5,legend=True)
#ads_freq.plot(kind='hist',bins=50,normed=True,log=True,alpha=0.5,legend=True)
#axis1.set(ylabel='Log10(Frequency)', xlabel='Count of Clicks')
# Users
# How about users? What's the frequency for user clicks? 
# Is it going to be just like clicks on ad?; 
# where we have many values(small) appeared a few times, and few values(large) that appeared so much
#users_freq = events_df['uuid'].value_counts()
# Plot frequency for user clicks

#fig, (axis1, axis2) = plt.subplots(2,1,figsize=(12,8))

# Same thing here; many values(small) appeared a few times, 
# and few large values(large) that appeared so much
#users_freq.plot(kind='hist',log=True,colormap="Set2",bins=50,ax=axis1)
#axis1.set(ylabel='Log10(Frequency)', xlabel='Count of Clicks')

# What's the percentage of users who clicked on ads less than(or equal) X times?
#users_perc = Series()
#for i in [2, 3, 5, 10, 50]:
#    users_perc[str(i)] = round((users_freq.values <= i).mean() * 100, 2)

#users_perc.plot(kind='bar',colormap="Set2",ax=axis2)
# Locations
# Grap the country from location(country>state>DMA)
#events_df['geo_location'] = events_df['geo_location'].apply(lambda x: str(x).split(">")[0])

# How many times each country participated in ad clicks?
# Limit the answer to only countries with more than(or equal) 100000 participation
#location_freq = events_df['geo_location'].value_counts()
#location_sum  = location_freq.sum()
#location_freq = location_freq[location_freq >= 100000]

#fig, (axis1) = plt.subplots(1,figsize=(12,5))
#location_freq.plot(kind='bar',colormap="Set3",ax=axis1)

#for p in axis1.patches:
#        axis1.annotate('%{:.2f}'.format(p.get_height() * 100 / location_sum), (p.get_x()+0.1, p.get_height()+100000))
        # Platform
# Just a quick look at different platforms, and see which is more impactful than the other

# Make sure all values are consistent; no 1(int) & "1"(str) at the same time!
#events_df["platform"] = events_df["platform"].map({1: "1", 2: "2", 3: "3"})
#events_df["platform"] = events_df["platform"].astype(str)

# Remove all NaN values
#platform_freq = events_df["platform"].value_counts()
#del platform_freq['nan']
#platform_sum = platform_freq.sum()
# Plot count(frequency) for every platform
#fig, (axis1) = plt.subplots(1,figsize=(12,5))
#platform_freq.plot(kind='bar',colormap="Set3",ax=axis1)

#for p in axis1.patches:
#        axis1.annotate('%{:.2f}'.format(p.get_height() * 100 / platform_sum), (p.get_x()+0.1, p.get_height()+100000))
        # Predictions
# The requirements: For every set of recommendation, sort the ads according to their likelyhood of being clicked
# The solution: For every set of recommendation, we are going to sort the ads based on one of the following:
    # 1. Count of ad Views(clicked or not clicked)
    # 2. Count of ad Clicks
    # 3. Average of ad Clicks = Count of ad Clicks / Count of ad Views
    # 4. Adjusted Average(with constant) = Count of ad Clicks / (Count of ad Views + constant)
    # 5. Adjusted Average(using power) = Count of ad Clicks^2 / Count of ad Views
    # 6. Log10(Count of ad Clicks)
    # 7. Probability Density Function F(ad) = (1/ Max - Min) X (Clicks - Min) — Max & Min for number of ad Clicks 
    # 8. Probability Density Function F(ad) = (1/ Max - Min) X (Clicks - Min) — Max & Min for number of ad Clicks in the current set of recommendation
    # 9. Calculate the zscore Count of ad Clicks
    # .... 
    
# The first solution can be misleading, as an ad can be have huge number of views but few clicks.
# Solutions 2, 6, 7, 8, 9, are almost the same. They depdend on the Count of Clicks for each ad.
# The 3rd Solution is reasonable, but, it can be tricky when you have an ad with views=2 and clicks=2,
    # and, another ad with views=1000 and clicks=800, 
    # so, the probability for the first add to be clicked is 100%, while the second is 80%, 
    # although the second ad has much higher number of clicks.
# The 4th and 5th Solutions are almost the same, and they solve the problem of the 3rd solution.
    # The 4th solution penalizes ads with small number of clicks by adding a fixed constant(usually the average of number of views)
    # The 5th solution powers the count of clicks, which in turn rewards the ads with large number of clicks.
    # NOTE: The constant can be tuned to improve the score.
    
# We will go with the 4th Solution, and see what we will get.

# First, clear up memory!
import gc
try: del clicks_train,clicks_test,events_df
except: pass;
gc.collect()

# Submission
constant = int(ads_freq.mean() + 578)
ads_adj_average = ads_clicked.values / ( ads_freq[ads_clicked.index] + constant ) 
# ads_adj_average = (ads_clicked.values**2) / ads_freq[ads_clicked.index] 

def get_score(ad):
    if ad not in ads_adj_average:
        return 0
    return ads_adj_average[ad] 

def solve(ads):
    # convert to int so we can sort
    ads = map(int, ads.split())
    # sort according to get_score function
    ads = sorted(ads, key=get_score, reverse=True) 
    # convert back to string so we can join by " "
    return " ".join(map(str, ads)) 
   
# Q: Why we are using sample_submission.csv file instead of the clicks_test.csv file?
# A: The sample_submission.csv files contains the same data as in clicks_test.csv, 
# but grouped by display_id, where each display_id has the ad ids separated by space.

submission = pd.read_csv("../input/sample_submission.csv") 
submission['ad_id'] = submission['ad_id'].apply(lambda ads: solve(ads))

submission.to_csv("outbrain.csv", index=False)