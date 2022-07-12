"""This is an attempt to visualize both the temporal and spatial distributions of crimes"""

__author__='Lian L'
__version__='0.3'
__date__='23/07/2015'


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('ggplot')
train_data=pd.read_csv('../input/train.csv')


# We can first visualize the distribution of crimes over a day (by hours) in different months:
import re
def parse_date(date):
    mo=re.search(r'^([0-9]{4})-([0-9]{2})-[0-9]{2}\s+([0-9]{2}):[0-9]{2}:[0-9]{2}$',date)
    return map(int,(mo.group(1),mo.group(2),mo.group(3)))
# Extract 'Year', 'Month' and 'Hour' columns for later use
month_dict={1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
train_data['Year'],train_data['Month'],train_data['Hour']=zip(*train_data.loc[:,'Dates'].map(parse_date))

data_month_hour=pd.crosstab(train_data['Hour'],train_data['Month'])
axhandles=data_month_hour.plot(kind='bar',subplots=True,layout=(4,3),figsize=(16,12),sharex=True,sharey=True,xticks=range(0,24,4),rot=0)
# Note here the subplots are based on columns, each column a new subplot
i=1
for axrow in axhandles:
    for ax in axrow:
        ax.set_xticklabels(range(0,24,4))
        ax.legend([month_dict[i]],loc='best')
        # Note here the argument has to be a list or a tuple, e.g. (month_dict[i],).
        # From Matplotlib official documentation: To make a legend for lines which already exist on the axes (via plot for instance),
        #    simply call this function with an ITERABLE of strings, one for each legend item.
        ax.set_title("")
        i+=1
plt.suptitle('Distribution of Crimes by Hour',size=20)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('Distribution_of_Crimes_by_Hour.png')


# We can also have a look at the distribution over days in a week
day_of_week=train_data['DayOfWeek'].value_counts()
day_of_week=day_of_week.reindex(index=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
# Make the DataFrame or Series conform to the new index order
fig,ax=plt.subplots()
day_of_week.plot(kind='bar',ax=ax,title='Discribution of Crimes by Day in Week',rot=0)
fig.savefig('Distribution_of_Crimes_by_Day_in_Week.png')


# Next, let's explore the spatial distribution of crimes. We can create a simple crosstab table and look at the count distribution
# Let's try to look at the city-wide 10 most common crimes in SF and a breakdown by district

ten_most_common=train_data[train_data['Category'].isin(train_data['Category'].value_counts().head(10).index)]

ten_most_crime_by_district=pd.crosstab(ten_most_common['PdDistrict'],ten_most_common['Category'])
ten_most_crime_by_district.plot(kind='barh',figsize=(16,10),stacked=True,colormap='Greens',title='Disbribution of the City-wide Ten Most Common Crimes in Each District')
plt.savefig('Disbribution_of_the_City-wide_Ten_Most_Common_Crimes_in_Each_District.png')

# Now let's look at the crime trend by year for each district. We have three features: 'Year', 'Category' and 'PdDistrict'. For each 'PdDistrict' and 'Year', we would like to see the crime composition instead of simply counts, since data are incomplete for 2015
freq_by_d_c=pd.pivot_table(train_data[['PdDistrict','Category','Year','Dates']],values='Dates',columns=('Year'),index=('PdDistrict','Category'),aggfunc='count')
freq_by_d_c=freq_by_d_c.fillna(0).apply(lambda x: x/np.sum(x))
freq_by_d_c=freq_by_d_c.stack()
freq_by_d_c=freq_by_d_c.reset_index()
freq_by_d_c=freq_by_d_c.rename(columns={0:'Fraction'})
most_common=freq_by_d_c[freq_by_d_c['Category'].isin(train_data['Category'].value_counts().head(10).index)]

by_hour_for_months=sns.FacetGrid(most_common, hue='Category', col='PdDistrict', col_wrap=5, sharex=True, sharey=False, size=4,\
                        aspect=0.9, palette='rainbow')
by_hour_for_months=by_hour_for_months.map(plt.plot,'Year','Fraction').add_legend()
plt.savefig('Crime_Trend_in_Each_District.png')

# We want to create a scatterplot of crime occurences for the whole city
# Borrowing the map and information from Ben's script
SF_map= np.loadtxt("../input/sf_map_copyright_openstreetmap_contributors.txt")
# Supplied map bounding box:
#    ll.lon     ll.lat   ur.lon     ur.lat
#    -122.52469 37.69862 -122.33663 37.82986
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
asp = SF_map.shape[0] * 1.0 / SF_map.shape[1]
fig = plt.figure(figsize=(16,16))
plt.imshow(SF_map,cmap='gray',extent=lon_lat_box,aspect=1/asp)
ax=plt.gca()
# Discard some entries with erratic position coordinates
train_data[train_data['Y']<40].plot(x='X',y='Y',ax=ax,kind='scatter',marker='o',s=2,color='green',alpha=0.01)
ax.set_axis_off()
plt.savefig('TotalCrimeonMap.png')