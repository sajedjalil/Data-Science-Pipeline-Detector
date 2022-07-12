import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

import pickle
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Uncomment this section for kaggle Execution
import riiideducation
base_url = '/kaggle/input/riiid-test-answer-prediction/'

#base_url = './input/'

tag_keys = []
# This is a framework for RIIID competation

# Entrypoint for training model
# [TODO] Change this function to switch different methods
# Put the actual logic into another function instead. Only call here.
def test_model(test_df, gbm):
#	return sample_method_always_predict_1(test_df)
	return lgbm_compressed(gbm, test_df)

def lgbm_compressed(gbm, raw_test_df):
	raw_test_df = raw_test_df.loc[raw_test_df['content_type_id'] == 0, :]
	test_x = init_dict(raw_test_df[['content_id']])
		
	for idx, row in raw_test_df.iterrows():
		user_id = int(row['user_id'])
		select_feature(idx, row, user_id, test_x)
		user_status[user_id]['last_ts'] = max(user_status[user_id]['last_ts'], row['timestamp'])
		
	y_pred = gbm.predict(pd.DataFrame(data=test_x))
    #rounding the values
	y_pred=y_pred.round(0)
	#converting from float to integer
	y_pred=y_pred.astype(int)
    
	prediction_result = raw_test_df.loc[:, ['row_id']]
	prediction_result['answered_correctly'] = y_pred
	return prediction_result
	
# A sample method, always return 1 for prediction
def sample_method_always_predict_1(test_df):
	test_df['answered_correctly'] = 1
	prediction_result = test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']]
	return prediction_result

# for viewing all files
def checkInputFiles():
	for dirname, _, filenames in os.walk('/kaggle/input'):
		for filename in filenames:
			print(os.path.join(dirname, filename))

def insert_to_key_set(key, value, dictionary):
	if not (key in dictionary):
		dictionary[key]=set()
	dictionary[key].add(value)

	
def readline(f):
	x = f.readline().strip().split(',')
	return x

lecture_keys = []
def read_lectures():
	lectures = pd.read_csv(base_url+'lectures.csv')
	for lec_id in lectures['lecture_id']:
		lecture_keys.append(lec_id)
		
def read_questions():
	questions = pd.read_csv(base_url+'questions.csv')
	print (questions.head())
	tag_dict = {}
	
	question_dict = {}
	
	tags = {}
	for i,q in questions.iterrows():
		#print (q['tags'])
		tags[q['question_id']] = []
		try:
			tag = 0
			raw_tags = q['tags'].strip().split(' ')
			for t in raw_tags:
				if not t in tag_dict:
					tag_dict[t] = len(tag_dict)+1
				tag = tag_dict[t]	
				tags[q['question_id']].append(tag)
		except AttributeError:
			tag = 0
			tags[q['question_id']].append(tag)
		#seed = 4294967295
		#for x in range(6):
		#	newtag = (tag & seed)
		#	tag = tag >> 32
		#	result[x].append(newtag)
		
	#for x in range(6):
	#questions['tag'+str(x)]=result[x]
	
	print(tag_dict)
	for tag in range(1,1+len(tag_dict)):
		questions['tag_'+str(tag)] = 0
		tag_keys.append('tag_'+str(tag))
	for i,q in questions.iterrows():
		for tag in tags[q['question_id']]:
			if tag > 0:
				questions.at[i, 'tag_'+str(tag)]=1
			# 	print(q[0:10])
		
	print (len(tag_dict))
	#print (questions.head())
	print (questions.iloc[0:10,0:10])
	return questions
	
def init_dict(df):
	
	keys = ('question_avg_correctness', 'user', 'part_correctness', 'time_from_first', 'time_from_last', 'total', 'part', 'lectures', 'prior_question_elapsed_time')
	data_x = {}
	length = len(df)
	for key in keys:
		data_x[key] = [0] * length
	for key in tag_keys:
		data_x[key] = [0] * length
		
	for key in lecture_keys:
		data_x[key] = [0] * length
	return data_x
def select_feature(index, row, user_id, train_data_x):

	if user_id not in user_status:
		user_status[user_id] = {'first_ts':row['timestamp'], 'last_ts':row['timestamp'], 'correct':0, 'wrong':0, 'lectures':set()}

	question_id = row['content_id']
	
	try:
		train_data_x['question_avg_correctness'][index]=question_correct.loc[question_id][0]
	except KeyError:
		train_data_x['question_avg_correctness'][index]=0.5
	train_data_x['user'][index]=user_id
	
	part = questions[questions['question_id'] == question_id]['part'].iloc[0]
	train_data_x['part'][index]=part
	try:
		train_data_x['part_correctness'][index]=user_correct.loc[(user_id, part)][0]
	except KeyError:
		train_data_x['part_correctness'][index]=user_correct_default.loc[part][0]
	
	
	q = questions[questions['question_id'] == question_id]
	for key in tag_keys:
		item = q[key].iloc[0]
		train_data_x[key][index]=item
	
	for lec_id in lecture_keys:
		try:
			if (lec_id in user_status[user_id]['lectures']):
				train_data_x[lec_id][index]=1
				continue
		except KeyError:
			pass
		train_data_x[lec_id][index]=0
		
	train_data_x['lectures'][index]=len(user_status[user_id]['lectures'])

	train_data_x['time_from_first'][index]=row['timestamp']/1000
	
	train_data_x['prior_question_elapsed_time'][index]=row['prior_question_elapsed_time']/1000
	train_data_x['time_from_last'][index]=(row['timestamp']-user_status[user_id]['last_ts'])/1000
	train_data_x['total'][index]=user_status[user_id]['correct'] + user_status[user_id]['wrong']
#	try:
#		train_data_x['correctness'][index]=1.0* user_status[user_id]['correct']/(user_status[user_id]['correct'] + user_status[user_id]['wrong'])
#	except:
#		train_data_x['correctness'][index]=0.5
		
global question_correct
question_correct = pd.DataFrame()
def read_train_data(questions):
	
	num_of_rows = 10**5
	raw_train_data = pd.read_csv(base_url+'train.csv', nrows=num_of_rows)
	
	
	train_data_y = []
	last_row = None
	first_timestamp = 0
	last_timestamp = 0
	
	print('start groupby')
	group_by_questions = raw_train_data[raw_train_data['content_type_id']==0][['content_id','answered_correctly']].groupby(['content_id'])

	global question_correct
	question_correct = group_by_questions.sum() / group_by_questions.count()
	
	global user_correct, user_correct_default
	part_map = questions[['question_id','part']].to_dict()['part']
	data = raw_train_data[raw_train_data['content_type_id']==0][['user_id','content_id','answered_correctly']]
	data['part'] = data['content_id'].map(part_map)
	group_by_parts = data[['user_id','part','answered_correctly']].groupby(['user_id','part'])
	user_correct = group_by_parts.sum()/group_by_parts.count()
	user_correct_default = user_correct.groupby(['part']).mean()
	
	print('finish groupby')
	
	train_data_x = init_dict(data[['content_id']])
	
	num_of_question = 0
	for idx, row in raw_train_data.iterrows():
		#print (row)
		if (idx % 1000 == 0):
			print(idx, 'rows processed')
		if (last_row is None) or (last_row['user_id'] != row['user_id']):
			user_id = int(row['user_id'])
			if user_id not in user_status:
				user_status[user_id] = {'first_ts':row['timestamp'], 'last_ts':row['timestamp'], 'correct':0, 'wrong':0, 'lectures':set()}
				
		else:
			user_status[user_id]['last_ts'] = last_row['timestamp']
		
		# is a question?
		if row['content_type_id'] == 0:
			train_data_y.append(row['answered_correctly'])
			if (row['answered_correctly'] ==1):
				user_status[user_id]['correct'] +=1
			else:
				user_status[user_id]['wrong'] +=1
			
			select_feature(num_of_question, row, user_id, train_data_x)
			num_of_question+=1
		else:
			user_status[user_id]['lectures'].add(row['content_id'])
		last_row = row
	
	
	#print(train_data_x)
	result = pd.DataFrame(data=train_data_x)
	
	#result['answer'] = train_data_y
	
	numoftrain = int(num_of_rows*0.9)
	train_x = result.iloc[0:numoftrain, :]
	test_x = result.iloc[numoftrain:,:]
	
	train_y = train_data_y[0:numoftrain]
	test_y = train_data_y[numoftrain:]
	
	#for i in range(0,len(train_x.columns),5):
	step = 6
	for i in range(0,30,step):
		print (train_x.iloc[0:10,i:i+step])
	#print (train_x[train_x[6808]>0].iloc[0:15,0:20])
	#print (train_x[train_x[6808]>0].iloc[0:15,270:290])
	train_data = lgb.Dataset(train_x, label=train_y)
	validate_data = lgb.Dataset(test_x, label = test_y)
	
	#with open('train_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
	#	pickle.dump(train_data, f)
	#	print ('train_data success stored')
		
	#with open('validate_data.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
	#	pickle.dump(validate_data, f)
	#	print ('validate_data success stored')
		
	return (train_data,validate_data)

def train(train_data, validation_data):
	# 参数
	params = {
		'learning_rate': 0.03,
		'lambda_l1': 0.05,
		'lambda_l2': 0.05,
		'max_depth': -1,
		'objective': 'binary',  # 目标函数
		#'metric': 'auc',
		'feature_fraction':0.75,
		'bagging_freq':10,
		'bagging_fraction':0.80
	}
	
	#print(train_data.data)
	#return
	
	gbm = lgb.train(params, train_data, valid_sets=[validation_data], num_boost_round=400, early_stopping_rounds=10, verbose_eval=50)
	return gbm
# main process
def read_data():
	# Training data is in the competition dataset as 
	# nrows can be changed within the limit of memory usage
	questions = pd.read_csv(base_url+'questions.csv')
	print (questions.head())
	
	lectures = pd.read_csv(base_url+'lectures.csv')
	print (lectures.head())
	
	tag_dict = {'lectures':{}, 'questions':{}}
	for index, lecture in lectures.iterrows():
		lecture_tags = tag_dict['lectures']
		insert_to_key_set(lecture['tag'], lecture['lecture_id'], lecture_tags)
	
	question_correctness = {}
	tags_by_question = {}
	for index,question in questions.iterrows():
		question_correctness[question['question_id']] = {}
		question_correctness[question['question_id']][True]={}
		question_correctness[question['question_id']][False]={}
		
		for tag_id in str(question['tags']).split(' '):
			try:
				insert_to_key_set(int(tag_id), question['question_id'], tag_dict['questions'])
				insert_to_key_set(question['question_id'], int(tag_id), tags_by_question)
			except ValueError:
				pass
	
	#print (tag_dict['lectures'])
	#print (tag_dict['questions'])
	#print (tags_by_question)
	#return
	#train_data = dt.fread('/kaggle/input/riiid-test-answer-prediction/train.csv').to_pandas()
	#for train_data in pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',low_memory=False,
	#					   chunksize=10**4, 
	#					   dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',
	#							  'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 
	#							 'prior_question_had_explanation': 'boolean',
	#						}):
	
	f=open(base_url+'train.csv')
	headline = readline(f)
	header = {}
	for i in range(len(headline)):
		header[headline[i]] = i
	
	line = readline(f)
	cnt = 0
	while (line):
		last_userid = 0
		try:
			data = {}
			for i in range(len(line)):
				data[headline[i]] = line[i]
			if last_userid != data['user_id']:
				print (data['user_id'])
				tags_covered = set()
				last_userid = data['user_id']
				cnt += 1
				
			#print("type",int(data['content_type_id']))
			if int(data['content_type_id']) == 1: # lecture
				lecture_id = int(data['content_id'])
				tag = lectures[lectures['lecture_id'] == lecture_id]['tag'].iloc[0]
				
				tags_covered.add(tag)
				print ("Lecture-",lecture_id,"tags",tag, lectures['lecture_id'])
			else:
				question_id = int(data['content_id'])
				correct = (int(data['answered_correctly']) == 1)
				
				if question_id in tags_by_question:

					question_tags = tags_by_question[question_id]
					print (question_tags, tags_covered)
					lecture_tags_count = len(question_tags.intersection(tags_covered))
				else:
					lecture_tags_count = 0
				if not question_id in question_correctness:
					question_correctness[question_id]={True:{}, False:{}}
				if not lecture_tags_count in question_correctness[question_id][correct]:
					question_correctness[question_id][correct][lecture_tags_count] = 1
				else:
					question_correctness[question_id][correct][lecture_tags_count] += 1
				#print("Question-",question_correctness[question_id])
		except Exception as e:
			print(e)
			print(data)
			break
		if (cnt ==100):
			break
		#print (train_data.head(1)[['row_id']])
		line = readline(f) 
		#break
		
	#print(tags_by_question)
	#print(question_correctness)
	# obj0, obj1, obj2 are created here...
	return
	# Saving the objects:
	with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
		pickle.dump(question_correctness, f)
		print ('success stored')
	return question_correctness
def load():
	with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
		question_correctness = pickle.load(f)
	#print ('loading success', question_correctness)
	
	for q in question_correctness:
		print(q,question_correctness[q])
		#break
	
def main2():
	iris = datasets.load_iris()
	X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
	print(X_train)

def main():
	
	train_data, validate_data = read_train_data(questions)
	gbm = train(train_data, validate_data)
	test(gbm)
	#read_data()
	#df = read_data()
	#load()

def test(model):
	env = riiideducation.make_env()
	iter_test = env.iter_test()
	batch = 0
	results = []
	for (test_df, sample_prediction_df) in iter_test:
		result = test_model(test_df, model)
		env.predict(result)
		batch += 1
		
		# print out the process
		print ("Batch "+ str(batch)+" completed.")
		
		# save the result
		results.append(result)
		 
	# combine all prediction results into a single pd.dataframe
	output = pd.concat(results)
	
	# save prediction result. file name must be "submission.csv"
	output.to_csv('submission.csv', index=False)

# run run run
user_status = {}
read_lectures()
questions = read_questions()
main()