
#imports
import numpy as np, pandas as pd, spacy, string, json
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from spacy.vocab import Vocab
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.calibration import CalibratedClassifierCV

nlp = spacy.load("en_core_web_lg") #spacy nlp vocab

stop_words = set(stopwords.words('english')) #stopwords
lemmatizer = WordNetLemmatizer() #lemmatizer

train_data = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv") #train data
# print(train_data.head())


def listToString(s):  #helper
    str1 = " "
    return (str1.join(s))


new_data = pd.DataFrame(columns=['Text','Target','Keyword', 'Location'])
columns = list(new_data)
print(columns)
input_rows = []
for idx in train_data.index:
    target = train_data['target'][idx]
    utter = train_data['text'][idx]
    keyword = train_data['keyword'][idx]
    location = train_data['location'][idx]
    utter = utter.lower()
    single_split = utter.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in single_split]
    words = [word for word in stripped if word.isalpha()]
    words = [w for w in words if not w in stop_words]
    inner_lemma = [lemmatizer.lemmatize(word) for word in words]  
    inner_lemma_str = listToString(inner_lemma)
    values = [inner_lemma_str, target, keyword, location]
    zipped = zip(columns, values)
    a_dictionary = dict(zipped)
    input_rows.append(a_dictionary)
new_data = new_data.append(input_rows, True)

# print(new_data.head(10))

#replacing NA with 0
new_data["Keyword"] = new_data["Keyword"].fillna('None')
new_data["Location"] = new_data["Location"].fillna('None')

# print(new_data.head(50))


#adding tfidf vectors to df
tf = TfidfVectorizer()
utter_list = list(new_data['Text'])
tf_vector = tf.fit_transform(utter_list).toarray()
tf_vector_list = list(tf_vector)
new_data["tfidf_vec"] = tf_vector_list
new_data.head(10)

X1 = new_data['tfidf_vec']
X2 = new_data['Keyword']
X3 = new_data['Location']
Y = new_data['Target']

X1=np.array(X1)
X2=np.array(X2)
X3=np.array(X3)
Y=np.array(Y)


#encoding
le_x2 = LabelEncoder()
X2_label = le_x2.fit_transform(X2)

le_x3 = LabelEncoder()
X3_label = le_x3.fit_transform(X3)

new_data["key_label"] = X2_label
new_data["loc_label"] = X3_label


X = new_data[['key_label', 'loc_label','tfidf_vec']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(len(X_train))

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

#preprocessing function
def data_preprossing(X, train=True, ohe=None):
    
    X['temp'] = 1
    temp = X['tfidf_vec']
    X.drop(['tfidf_vec'], axis=1, inplace=True)
    
    if train:
        ohe = OneHotEncoder(handle_unknown='ignore')
        X = ohe.fit_transform(X).toarray()
    
    else:
        X = ohe.transform(X).toarray()
    X = np.delete(X, -1, 1)
    X = np.hstack((np.vstack(temp), X))

    return X, ohe


#training set
X_train_target, ohe_mc = data_preprossing(X_train, train=True)

#validation set
X_test_target, ohe_mc = data_preprossing(X_test, train=False, ohe=ohe_mc)


#SVM Classifier
clf = LinearSVC(multi_class="ovr", random_state=42)
clf.fit(X_train_target, Y_train)

Y_pred = clf.predict(X_test_target)

#accuracy
print(accuracy_score(Y_test,Y_pred)) #0.767564018384767
