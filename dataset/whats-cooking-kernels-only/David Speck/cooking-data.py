import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

print("Reading data...")
train = json.load(open("../input/train.json"))
test = json.load(open("../input/test.json"))

print("Vectorize data...")
labels = []
train_data = []
for recipe in train:
    train_data.append(" ".join(recipe["ingredients"]))
    if recipe["cuisine"] not in labels:
        labels.append(recipe["cuisine"])
        
test_data = []
for recipe in test:
    test_data.append(" ".join(recipe["ingredients"]))

join = " ".join(test[0]['ingredients'])

# create vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit( train_data )
# create map for labels
label_to_int = dict(((c, i) for i, c in enumerate(labels)))
int_to_label = dict(((i, c) for i, c in enumerate(labels)))

Y_train = [ label_to_int[r["cuisine"]] for r in train ]
X_train = vectorizer.transform( train_data )
X_test = vectorizer.transform( test_data )

print("Fit classifier...")

classifier = GradientBoostingClassifier(loss='deviance', n_estimators=100, learning_rate=0.1)
classifier.fit(X_train, Y_train)

print("Predicting on test data...")
y_pred = classifier.predict(X_test)

print ("Generate Submission File ... ")
test_id = [doc['id'] for doc in test]
test_cuisine = [int_to_label[i] for i in y_pred]
sub = pd.DataFrame({'id': test_id, 'cuisine': test_cuisine}, columns=['id', 'cuisine'])
sub.to_csv('svm_output.csv', index=False)
