import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train["comment_text"]
test_text = test["comment_text"]
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)

train_target = train["target"] > 0.5

#solvers = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
#C = [0.1, 0.5, 1, 2]
#cv_scores = []
#for solver in solvers:
#    for c in C:
#        classifier = LogisticRegression(C=c, solver=solver)        
#        
#        cv_score = np.mean(cross_val_score(classifier, train_word_features, train_target, cv=3, scoring='roc_auc'))
#        print('CV score for {} solver and C = {} is {}'.format(solver, c, cv_score))  
#        cv_scores.append(cv_score)
submission = pd.DataFrame.from_dict({'id': test['id']})

classifier = LogisticRegression(C=0.5, solver="saga")        

classifier.fit(train_word_features, train_target)
submission['prediction'] = classifier.predict_proba(test_word_features)[:, 1]
submission.to_csv('submission.csv', index=False)

features = np.array(word_vectorizer.get_feature_names())
np.array(word_vectorizer.get_feature_names())[np.where(classifier.coef_ > 1)[1]]
# Then, seeing which of them are identity-based
identity_words_1 = ['africans', 'aids', 'alien', 'aliens', 'atheist', 'black', 'blacks', 'cancer', 'chinese', 'christian',
                  'christians', 'evangelicals', 'fascist', 'fascists', 'felon', 'felons', 'feminism', 'feminist', 'feminists',
                 'gay', 'gays', 'heterosexual', 'hispanics', 'homosexual', 'homosexuality', 'homosexuals', 'indian', 'indians',
                 'islam', 'islamic', 'jew', 'jews', 'leftist', 'leftists', 'lesbian', 'liberalism', 'males', 'man', 'marxist',
                 'mexican', 'mexicans', 'mexico', 'muslim', 'muslims', 'palestinian', 'russian', 'slave', 'slaves', 'transgender',
                 'transgendered', 'white', 'whites', 'wife', 'woman', 'women']
                 
len(identity_words_1)
len(np.array(word_vectorizer.get_feature_names())[np.where(classifier.coef_ > 1)[1]])

np.array(word_vectorizer.get_feature_names())[np.where(classifier.coef_ > 2)[1]]
identity_words_2 = ['black', 'blacks', 'christians']

len(identity_words_2)
len(np.array(word_vectorizer.get_feature_names())[np.where(classifier.coef_ > 2)[1]])