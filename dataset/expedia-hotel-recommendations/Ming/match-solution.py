# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


'''

using original train as the training data
'''

from collections import defaultdict


def get_data_dict_key(filename):
	data_dict = defaultdict(lambda: defaultdict(str) )

	with open(filename,'r') as fin:
		line = fin.readline().strip()
		arr = line.split(',')
		
		for elem in arr:
			data_dict[elem] = None

	fin.close()
	return data_dict







import numpy as np 


def apk(actual, predicted, k =5):

	if(len(predicted)> k):
		predicted = predicted[:k]

	score = 0.0
	num_hits = 0.0

	for i,p in enumerate(predicted):
		if p in actual and p not in predicted[:i]:
			num_hits += 1
			score += num_hits / (i + 1.0)
	return score /min( len(actual), k) 


def Get_evaluation_score(actual, predicted, k = 5):

	return np.mean( [ apk(a,p,k) for a,p in zip(actual,predicted)] )


def Get_pure_evaluation_score(actual,predicted,k = 5):
	pure_actual = []
	pure_predicted = []
	for i in range(len(actual)):
		if(predicted[i] != []):
			pure_actual.append(actual[i])
			pure_predicted.append(predicted[i])

	index_list = []
	for i in range(len(predicted)):
		if (predicted[i] !=[]):
			#print('predicted list item is:',predicted[i], 'index is:', i)
			if int( apk(actual[i],predicted[i]) ) == 0:
				index_list.append(i)

	#print('index_list is:', index_list)

	return pure_actual, pure_predicted, index_list,  Get_evaluation_score(pure_actual, pure_predicted)





from collections import defaultdict
from heapq import nlargest
from operator import itemgetter

import time
import os

from datetime import datetime 

max_line_validation = 10

def get_match_array_dict(filename_train):
	train_data_dict = get_data_dict_key(filename_train)
	top_hotel_cluster_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))


	with open(filename_train,'r') as fin:
		line = fin.readline().strip()
		col_name_list = line.split(',')

		total_line = 0
		while 1:
			line = fin.readline().strip()
			total_line += 1

			if(line == ''):
				break
# 			if(total_line > 1000000):
# 				break

			if(total_line% 1000000 == 0):
				print('match array list', 'reading {} lines....'.format(total_line))

			arr = line.split(',')

			for i, elem in enumerate(arr):
				train_data_dict[ col_name_list[i] ] = elem

			######### srch_destination_id

			
			hotel_cluster = train_data_dict['hotel_cluster']
			score_1 = 18 if int(train_data_dict['is_booking']) else 3
			score_2 = 5 if int(train_data_dict['is_booking']) else 1

# 			if train_data_dict['user_id'] !='' and train_data_dict['srch_destination_id'] !='':
# 				key = (train_data_dict['user_id'],train_data_dict['srch_destination_id'])
# 				top_hotel_cluster_dict['user_id-srch_destination_id'][key][hotel_cluster] += score_1

# 			if train_data_dict['srch_destination_id'] !=  '':
# 				key = train_data_dict['srch_destination_id']
# 				top_hotel_cluster_dict['srch_destination_id'][key][hotel_cluster] += score_1


			if train_data_dict['user_location_city'] !='' and train_data_dict['orig_destination_distance'] != '':
				key = (train_data_dict['user_location_city'], train_data_dict['orig_destination_distance'])	
				top_hotel_cluster_dict['user_location_city-orig_destination_distance'][key][hotel_cluster] += 1


# 			if train_data_dict['user_id'] !='' and train_data_dict['orig_destination_distance'] != '':
# 				key = (train_data_dict['user_id'], train_data_dict['orig_destination_distance'])	
# 				top_hotel_cluster_dict['user_id-orig_destination_distance'][key][hotel_cluster] += 1				


# 			if train_data_dict['user_location_city'] !='' and train_data_dict['orig_destination_distance'] != '' and train_data_dict['srch_destination_id'] !='':
# 				key = (train_data_dict['user_location_city'], train_data_dict['orig_destination_distance'],train_data_dict['srch_destination_id'])	
# 				top_hotel_cluster_dict['user_location_city-orig_destination_distance-srch_destination_id'][key][hotel_cluster] += 1


# 			if train_data_dict['hotel_country']!= '':
# 				key = train_data_dict['hotel_country']
# 				top_hotel_cluster_dict['hotel_country'][key][hotel_cluster] +=score_2

		fin.close()
			
	return train_data_dict, top_hotel_cluster_dict

def get_predict_list(filename_test,match_column_top_hotel =['user_location_city-orig_destination_distance']):
	def generate_key(arr, match_column_data_dict):
		index_list = []
		
		match_column_items = match_column_data_dict.split('-')

		for i in range(len(match_column_items)):
			#print('col name list is: ', col_name_list)
			index_in_arr = col_name_list.index(match_column_items[i])
			#print('index in array is: ', index_in_arr)
			index = arr[     index_in_arr     ]

			index_list.append(index)

		if len(index_list)>1:
			return tuple(index_list)
		else:
			return index_list[0]



	#print('max line hear is:',max_line_validation)
	predict_list = defaultdict( list )

	total_line = 0
	with open(filename_test,'r') as fin:
		col_name_list = fin.readline().strip().split(',')
		while 1:
			line = fin.readline().strip()
			total_line += 1

			if(line == ''):
				break
# 			if(total_line> max_line_validation):
# 				break

			if(total_line % 1000000 == 0):
				print('get predict list', 'reding {} lines...'.format(total_line))
			arr  = line.split(',')

			#print('match column top hotel is', match_column_top_hotel)
			for match_column_top_hotel_item in match_column_top_hotel:
				index_tuple = generate_key(arr,match_column_top_hotel_item)

				#print('index of predict is:', index_tuple)
				if(index_tuple in top_hotel_cluster_dict[match_column_top_hotel_item]):
					d = top_hotel_cluster_dict[match_column_top_hotel_item][index_tuple]

					topitems = nlargest(5, d.items(),key = itemgetter(1))
					
					predict_list[match_column_top_hotel_item].append([int(item[0]) for item in topitems])
				else:
					predict_list[match_column_top_hotel_item].append([])

		#print('predict list keys are ',predict_list.keys())
		list_length = len( predict_list[ list(predict_list.keys())[0]    ] )
		
		predict_list['total'] = []
		for i in range(list_length):
			predict_total_item = []
			for key in match_column_top_hotel:
				predict_total_item += predict_list[key][i]

			predict_list['total'].append(predict_total_item)	

		fin.close()
	
	return predict_list


def get_actual(filename_test):
	actual = []

	total_line = 0
	with open(filename_test,'r') as fin:
		col_name_list = fin.readline().strip().split(',')
		while 1:
			line = fin.readline().strip()
			total_line += 1

			if(line == ''):
				break
			# if(total_line> max_line_validation):
			# 	break

			if(total_line % 1000000 == 0):
				print('get actual', 'reding {} lines...'.format(total_line))


			arr  = line.split(',')
			hotel_cluster = int(arr[23])

			actual.append([hotel_cluster])
		fin.close()

	return actual

def get_submission( predicted ):
	now = datetime.now()
	filename_submission = os.path.join('submission_' + str(now.strftime("%Y_%m_%d_%H_%M")) + '.csv')

	fin = open(filename_test, 'r')
	col_name = fin.readline().strip().split(',')

	assert col_name[0] == 'id'

	print('generating id list...')
	total_line = 0
	id_list = []
	while 1:
		line = fin.readline().strip()
		
		total_line += 1
		if(line == ''):
			break

# 		if(total_line>max_line_validation):
# 			break

		if(total_line % 1000000 == 0):
			print('get submission', 'reding {} lines...'.format(total_line))
		arr = line.split(',')

		id_list.append(arr[0])




	fout = open(filename_submission,'w')
	fout.write('id,hotel_cluster\n')

	total_line = 0

	for i,row in enumerate(predicted):
		fout.write('{},'.format(i))
		for item in row:
			fout.write('{} '.format(item))
		fout.write('\n')

		total_line += 1
		# if(total_line > 1000):
		# 	break


	fin.close()
	fout.close()


if __name__ == '__main__':

	start_time = time.time()


	validate = 0
	#filename_train = os.path.join('data','validation','train_sub.csv')
	#filename_test = os.path.join('data','validation','validation.csv')

	filename_train = os.path.join('../input','train.csv')
	filename_test = os.path.join('../input','test.csv')


	#filename_submission = os.path.join('submission ')
	train_data_dict, top_hotel_cluster_dict = get_match_array_dict(filename_train)


	predict_list = get_predict_list(filename_test)

	if validate ==1:
		actual = get_actual(filename_test)
		pure_actual, pure_predicted, not_hit_index, evaluation_score =  Get_pure_evaluation_score(actual, predict_list['total'])
		print('evaluation_score is ', evaluation_score )
	else:
		get_submission(predict_list['total'])

	#print(actual[not_hit_index[0]], predict_list['total'][not_hit_index[0]], ' index is:', not_hit_index[0])


	

	print('running time is {} seconds....'.format(time.time() - start_time))
	#print(train_data_dict)
	#print( top_hotel_cluster_dict['srch_destination_id'] )
	#print(predict_list)