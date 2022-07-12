import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load Yandex/CERN's evaluation Python script from the input data
exec(open("../input/evaluation.py").read())

# Agreement and correlation conditions
ks_cutoff  = 0.09
cvm_cutoff = 0.002

# Load the training/test data along with the check file samples
train = pd.read_csv("../input/training.csv")
test  = pd.read_csv("../input/test.csv")
check_agreement   = pd.read_csv("../input/check_agreement.csv")
check_correlation = pd.read_csv("../input/check_correlation.csv")

# We'll track the results of the feature-by-feature tests in a DataFrame
features = pd.DataFrame(columns=["name", "agreement", "correlation"])
features["name"] = train.columns[1:-4]

for (i, fea) in enumerate(features["name"]):
    # To test each feature, we'll predict signal using a RF trained only on that feature
    rf = RandomForestClassifier(n_estimators=20, random_state=1)
    rf.fit(train[[fea]], train["signal"])
    # Run the correlation test
    correlation_probs = rf.predict_proba(check_correlation[[fea]])[:,1]
    cvm = compute_cvm(correlation_probs, check_correlation['mass'])
    features.loc[i, "correlation"] = cvm
    # Run the agreement test
    agreement_probs = rf.predict_proba(check_agreement[[fea]])[:,1]
    ks = compute_ks(
        agreement_probs[check_agreement['signal'].values == 0],
        agreement_probs[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    features.loc[i, "agreement"] = ks

print("Estimated agreement and correlation test score for each individual feature\n")
print(features)
good_features = list(features[features["agreement"]<ks_cutoff]["name"])

rf = RandomForestClassifier(n_estimators=50, random_state=1)
rf.fit(train[good_features], train["signal"])

print("\nEvaluating predictions from a RF on the good features on these tests\n")

# Agreement Test
agreement_probs = rf.predict_proba(check_agreement[good_features])[:,1]
ks = compute_ks(
    agreement_probs[check_agreement['signal'].values == 0],
    agreement_probs[check_agreement['signal'].values == 1],
    check_agreement[check_agreement['signal'] == 0]['weight'].values,
    check_agreement[check_agreement['signal'] == 1]['weight'].values)

if ks<ks_cutoff:
    print("This passed the agreement test with ks=%0.6f<%0.3f" % (ks, ks_cutoff))
else:
    print("This failed the agreement test with ks=%0.6f>=%0.3f" % (ks, ks_cutoff))

# Correlation Test
correlation_probs = rf.predict_proba(check_correlation[good_features])[:,1]
cvm = compute_cvm(correlation_probs, check_correlation['mass'])
if cvm<cvm_cutoff:
    print("This passed the correlation test with CvM=%0.6f<%0.4f" % (cvm, cvm_cutoff))
else:
    print("This failed the correlation test with CvM=%0.6f>=%0.4f" % (cvm, cvm_cutoff))

# Predictions on test set
test_probs = rf.predict_proba(test[good_features])[:,1] 
submission = pd.DataFrame({"id": test["id"], "prediction": test_probs})
# Not saving the submission file here so all output goes to the log
# submission.to_csv("python_submission.csv", index=False)
