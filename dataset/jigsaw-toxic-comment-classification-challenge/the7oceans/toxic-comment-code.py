import pandas as pd
import numpy as np
import csv
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.svm import *
from nltk.stem.porter import *
from sklearn.naive_bayes import *
from nltk.corpus import stopwords


# LOAD TRAIN DATA 
spam_data = pd.read_csv('../input/train.csv',nrows = 70000)
#spam_data = pd.read_csv('../input/train.csv', nrows = 10000)

#-----------------------------
#LOAD SUMBISSION DATA
#answer_data = pd.read_csv('../input/test.csv',nrows = 1000)
answer_data = pd.read_csv('../input/test.csv')

# PULL XDF VALUES
xdf1 = answer_data['comment_text']

# CREATE SUBMISSION DATAFRAME
submissiondf = answer_data['id']

#-------------------------------------------

# TARGET NAMES
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# DEFINE AND XDF
xdf = spam_data['comment_text']
xdf1 = answer_data['comment_text']

# ENCODE AND DECODE XDF
xdf = xdf.str.encode('utf8', 'ignore').str.lower()
xdf = xdf.str.decode('ascii', 'ignore')
xdf = xdf.str.split()

xdf1 = xdf1.str.encode('utf8', 'ignore').str.lower()
xdf1 = xdf1.str.decode('ascii', 'ignore')
xdf1 = xdf1.str.split()

# REMOVE STOP WORDS 
# NO NEED TO DO IT IN ANSWER DATA
stop = stopwords.words('english')

xdf = xdf.apply(lambda x: [item for item in x if item not in stop])

# STEM THE DATA FRAME
Porter = nltk.WordNetLemmatizer()

xdf = xdf.apply(lambda row: [Porter.lemmatize(line) for line in row])

xdf1 = xdf1.apply(lambda row: [Porter.lemmatize(line) for line in row])

# CONVERT LIST INTO STRING IN THE DATAFRAME
xdf = xdf.apply(' '.join)
xdf1 = xdf1.apply(' '.join)

# CONVERT COMMENTS TO FEATURES
vect = CountVectorizer(min_df = 5,ngram_range = (1,5)).fit(xdf)

# CONVERT TRAIN DATA TO VECT
xdf = vect.transform(xdf)
xdf1 = vect.transform(xdf1)

# TRAIN ALGORTHIM FOR EACH CLASS
for target_name in class_names:

	# DEFINE XDF & YDF
	xdf = xdf
	ydf = spam_data[target_name]
			
	# TRAIN AND TEST
	x_train, x_test, y_train, y_test = train_test_split(xdf,ydf, random_state=0)

	# CLASSIFIER
	clf = LogisticRegression(C=200).fit(x_train, y_train)
	#clf = BernoulliNB(alpha = .001).fit(x_train, y_train)

	# COMPUTE AUC
	predict_labels = clf.predict(x_test)

	auc = roc_auc_score(y_test,predict_labels)
	
	print ('\n','-----------------------------------------')
	print (target_name)
	print (auc)
	print ('')

	#---------------------------------------------------------

	# GET FEATURE NAMES INTO A LIST
	feature_names = np.array(vect.get_feature_names())

	# GET COEFS INTO A LIST
	coef = clf.coef_.T

	coef = coef.tolist()

	# CREATE COEF AND FEATURE DATA
	line = 	{'1-Feature_Names': feature_names,
			 '2-Coef_Values': coef,	
			}
		
	featuredf = pd.DataFrame(data = line)

	featuredf = featuredf.sort_values('2-Coef_Values',ascending=False)

	# PULL TOP AND BOTTOM 10

	feature_names_c = featuredf['1-Feature_Names'].reset_index().drop('index',axis = 1)
		
	top_10_features = feature_names_c.iloc[:10].values.tolist()

	print (top_10_features)

	#---------------------------------------------------------------------
	# PRINT SUBMISSION

	# TARGET & PROBABILITY
	probability = clf.predict_proba(xdf1)[:,1]
	target = clf.predict(xdf1)

	# ROUND NUMBERS TO MAKE IT EAST TO READ
	#probability = np.round_(probability,4)

	# CREATE DATAFRAME TO EXPORT
	line = 	{target_name : probability ,
			#'target' : target,
			}
			
	newdf = pd.DataFrame(data = line)
	
	submissiondf = pd.concat([submissiondf,newdf],axis = 1)
	
submissiondf = submissiondf.set_index('id')
	
# EXPORT TO EXCEL
submissiondf.to_csv('Submission.csv')

print (submissiondf.head())