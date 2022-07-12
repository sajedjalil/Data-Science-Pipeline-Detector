# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#function to clean data
def clean_text( raw_text ):#string is immutable
    list_of_word=[]
    ps = PorterStemmer()
    for word in raw_text:
        #word=word.replace(" ","")
        #list_of_word.append(word)
        words_token=word.lower().split()
        str=''
        for val in words_token:
            val=re.sub("[^a-zA-Z]", "",val)
            str=str+ps.stem(val)
        list_of_word.append(str)
    #print(list_of_word)    
    raw_text=" ".join( list_of_word )  
    
    #letters_only = re.sub("[^a-zA-Z]", " ", raw_text) 
    
    # 3. Convert to lower case, split into individual words
    #words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    #stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    
    #meaningful_words = [ps.stem(word) for word in words if not word in set(stopwords.words('english'))]  
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return raw_text
    #return( " ".join( meaningful_words )) 
dataset=pd.read_json('../input/train.json')
ingredient=dataset['ingredients']
sizeofingredients=ingredient.size
list_of_ingredient=[]
for i in range(0,sizeofingredients):
    list_of_ingredient.append(clean_text(dataset['ingredients'][i]))

test_dataset=pd.read_json('../input/test.json')
sizeofingredients=test_dataset['ingredients'].size
test_ingredients=[]
for i in range(0,sizeofingredients):
    test_ingredients.append(clean_text(test_dataset['ingredients'][i]))
total_ingredient= list_of_ingredient+test_ingredients   
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 2000)#max_features = 2000
X = cv.fit_transform(list_of_ingredient).toarray()     


Y=dataset['cuisine']
le = LabelEncoder()
Y = le.fit_transform(Y)
from sklearn.ensemble import RandomForestClassifier 
#from sklearn.multiclass import OneVsRestClassifier
clf=RandomForestClassifier()#(n_estimators=20,verbose=1,n_jobs=-1,min_samples_split=2,random_state=0) 
#mult_clf=OneVsRestClassifier(clf).fit(X,Y)
clf.fit(X,Y)


#cv.get_feature_names()    

#cv = CountVectorizer()

#X_test = cv.fit_transform(test_ingredients).toarray()
X_test = cv.fit_transform(test_ingredients).toarray() 

Y_pred=clf.predict(X_test) 
Y_pred = le.inverse_transform(Y_pred)
#Y_pred=gnb.predict(X_test)
#Y_pred=knn.predict(X_test)

cuisine_id=list(test_dataset['id'])
Y_ans=list(Y_pred)
np.savetxt('result2.csv', np.r_['1,2,0',np.array(cuisine_id),np.array(Y_ans)], delimiter=',', header = 'id,cuisine', comments = '', fmt='%s')