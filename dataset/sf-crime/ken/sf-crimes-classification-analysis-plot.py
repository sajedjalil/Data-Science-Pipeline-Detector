import pandas as pd
import zipfile
import matplotlib.pyplot as pl

z = zipfile.ZipFile('../input/train.csv.zip')
print(z.namelist())

train = pd.read_csv(z.open('train.csv'), parse_dates=['Dates'])


print(train.head())

# plot of the category counts
print(train.Category.value_counts())
train.Category.value_counts().plot(kind='barh', figsize=(14,7))
pl.savefig('category_counts.png')

# plot of the DayofWeek counts
print(train.DayOfWeek.value_counts())
train.DayOfWeek.value_counts().plot(kind='barh', figsize=(8,8))
pl.savefig('Dayofweek_counts.png')

# plot of the Resolution counts
print(train.Resolution.value_counts())
train.Resolution.value_counts().plot(kind='barh', figsize=(14,7))
pl.savefig('Resolution_counts.png')


#When the dates and address of two events are identify, we reconize them as multi-label event,so we groupby the Dates and Address
#use 'count()' to figure out how many categories each  (multi-label) event have.
train['multi']=1
count_multi_category = train[['Dates','Address','multi']].groupby(['Dates','Address']).count().reset_index()

#the detail descrbtion such as mean of the count of multi-category event
print(count_multi_category.multi.describe())

#Plot of the multi-category counts
print(count_multi_category.multi.value_counts())
count_multi_category.multi.value_counts().plot(kind='barh', title='counts of each type of events (group by the number of categories that it involved)', figsize=(14,7))  
pl.savefig('multi_category_counts.png')  