import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC

# Import Data
TrainData = pd.read_csv('../input/train.csv')
TrainData = TrainData.drop('id', axis=1)

# Extract target
# Encode it to make it manageable by ML algo
Labels = TrainData.target.values
Labels = LabelEncoder().fit_transform(Labels)

# Remove target from train, else it's too easy ...
TrainData = TrainData.drop('target', axis=1)

# Split Train / Test
Xtrain, Xtest, ytrain, ytest = train_test_split(TrainData, Labels, test_size=0.20, random_state=1)

# Classifier1 (Random forest)
Classifier1 = RandomForestClassifier(n_estimators=50, n_jobs=-1)
clfbag = BaggingClassifier(Classifier1, n_estimators=5)
clfbag.fit(Xtrain, ytrain)
ypredsC1 = clfbag.predict_proba(Xtest)
print("Loss for RandomForestClassifier: ", log_loss(ytest, ypredsC1, eps=1e-15, normalize=True))

# Classifier2 (SVM)
Classifier2 = svm.SVC(decision_function_shape='ovo')
Classifier2.fit(Xtrain, ytrain)
ypredsC2 = clfbag.predict_proba(Xtest)
print("Loss for SVM: ", log_loss(ytest, ypredsC2, eps=1e-15, normalize=True))




print(" ")
print("Conclusion : in our case, calibration improved performance a lot ! (reduced loss)")
