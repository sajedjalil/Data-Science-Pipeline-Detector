import numpy as np
import pandas as pd
import pylab as plt

df = pd.DataFrame.from_csv('../input/train.csv', index_col=False)
df['street_corner'] = df['Address'].apply(lambda x: 1 if '/' in x else 0)

crime_categories = df['Category'].unique()
n_crime_categories = crime_categories.shape[0]
street_corner_percentages = np.zeros((2,n_crime_categories+1))

for n in np.arange(n_crime_categories):
	if 0 in df['street_corner'].loc[df['Category'] == crime_categories[n]].value_counts().keys():
		street_corner_percentages[0,n] = df['street_corner'].loc[df['Category'] == crime_categories[n]].value_counts()[0]	
	if 1 in df['street_corner'].loc[df['Category'] == crime_categories[n]].value_counts().keys():
		street_corner_percentages[1,n] = df['street_corner'].loc[df['Category'] == crime_categories[n]].value_counts()[1]	
street_corner_percentages[:,-1] = np.sum(street_corner_percentages, axis=1)
street_corner_percentages = 100.0*(street_corner_percentages/np.sum(street_corner_percentages, axis=0))

# Plot the bar chart of percentages of crimes that occured on street corners
ind = np.arange(n_crime_categories)
width = 1.0

fig, axarr = plt.subplots(2,1)
rects2 = axarr[0].bar(ind, street_corner_percentages[1,:-1], width, color='y')

rects3 = axarr[1].bar(ind, street_corner_percentages[1,:-1]-street_corner_percentages[1,-1], width, color='y')

axarr[0].set_ylabel('Crimes that occurred \n on street corner (in %)')
axarr[1].set_ylabel('Difference from \n Average (in %)')
axarr[0].set_title('Percentage of crimes commited on a street corner, \n brokendown according to the category of the crime.')
axarr[0].set_xticks(ind+0.5*width)
axarr[1].set_xticks(ind+0.5*width)

fig.subplots_adjust(hspace=.5)

crimes_list = list(crime_categories)
axarr[1].set_xticklabels(crimes_list, rotation='vertical')
axarr[0].get_xaxis().set_ticks([])
plt.subplots_adjust(bottom=0.45)
plt.savefig('breakdown_of_crimes_by_whether_they_occured_on_a_street_corner.png')