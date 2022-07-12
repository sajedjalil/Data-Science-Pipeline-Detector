"""
This script is a sample of how to merge multiple datasets into a single dataset.

@author: AV
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

#==============================================================================
# merge gender_age_train.csv and events.csv
#==============================================================================
def merge_genderage_events(isTrain = True):
	
	
	if isTrain:
		# Get the category of apps installed per device_id
		df_genderage = pd.read_csv("../input/gender_age_train.csv")
	else:
		df_genderage = pd.read_csv("../input/gender_age_test.csv")
	# Get only those rows that match device_id in train data
	df_events = pd.read_csv("../input/events.csv")
	df_events = df_events.drop_duplicates(subset="device_id")
	df = pd.merge(df_genderage, df_events, on="device_id")
	del df_genderage, df_events
	
	return df

#==============================================================================
# Merge app_labels.csv and label_categories.csv
#==============================================================================
def merge_category_app():
	df_applabels = pd.read_csv("../input/app_labels.csv")
	df_category = pd.read_csv("../input/label_categories.csv")
	df_category = pd.merge(df_applabels, df_category, on="label_id")
	del df_applabels
	    
	return df_category.drop_duplicates(subset="app_id")


#==============================================================================
# Merge events.csv with others
#==============================================================================
def merge_events_genderage(df_genderage):
	# read file in chunks
	chunks = pd.read_csv("../input/app_events.csv", chunksize=2e6)
	df_tmp = pd.DataFrame()
	for chunk in chunks:
		#chunk = chunk.drop_duplicates(subset="event_id")
		tmp = pd.merge(df_genderage, chunk, on="event_id")
		df_tmp = df_tmp.append(tmp)
		
	return df_tmp

#==============================================================================
# Merge category dataframe
#==============================================================================
def merge_category_genderage(df_category, df_genderage):
	return pd.merge(df_genderage, df_category, on="app_id")	

#==============================================================================
# Process train data
#==============================================================================
def process_training_data():
	#sys.modules[__name__].__dict__.clear()
	print ("Merging gender_age_train.csv and events.csv...")
	df1 = merge_genderage_events(True)
	print("Merging app_labels.csv and label_categories.csv...")
	df2 = merge_category_app()
	print(df1.shape, df2.shape)
	
	df3 = merge_events_genderage(df1)
	df4  = merge_category_genderage(df2, df3)
	
	df = df4[['device_id','gender','age','group','is_installed','is_active','category']]
	
	df.to_csv('train.csv', index=False)

#==============================================================================
# process test data
#==============================================================================
def process_test_data():
	print ("Merging gender_age_test.csv and events.csv...")
	df1 = merge_genderage_events(False)
	print("Merging app_labels.csv and label_categories.csv...")
	df2 = merge_category_app()
	print(df1.shape, df2.shape)
	
	df3 = merge_events_genderage(df1)
	df4  = merge_category_genderage(df2, df3)
	
	df = df4[['device_id','is_installed','is_active','category']]
	
	df.to_csv('test.csv', index=False)

#==============================================================================
# Main 
#==============================================================================
if __name__ == '__main__':
	process_training_data()
	process_test_data()