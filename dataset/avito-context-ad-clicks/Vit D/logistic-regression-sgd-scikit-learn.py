from csv import DictReader
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from datetime import datetime, date, time
from pprint import pprint
from math import sqrt, exp, log
import pandas as pd

train_file = '../input/trainSearchStream.tsv'
test_file = '../input/testSearchStream.tsv'
submission = 'submission_lr-sgd_own-hash_proba_bits1048576_FULL.csv'
submission_final = 'submission_lr-sgd_own-hash_proba_bits1048576_SBM.csv'

model = SGDClassifier(loss='log', penalty='l2', n_iter=1)

all_classes = np.array([0, 1])
bits = 20
train_count = 1
test_count = 1
logloss = 0.

# Train model
print(" Model training started at " + str(datetime.now()) + " . \n")
for t, line in enumerate(DictReader(open(train_file), delimiter='\t')):
    y = 0.

    try:
        y = int(line['IsClick'])
        del line['IsClick']
    except:
        pass
    try:
        del line['ID']
    except:
        pass
    try:
        del line['IsClick']
    except:
        pass
    try:
        del line['HistCTR']
    except:
        pass

    l = [0.] * len(line)

    for i, key in enumerate(line):
        val = line[key]
        l[i] = (abs(hash(key + '_' + val)) % bits)

    model.partial_fit(l, [y], classes=all_classes)
    pred = model.predict(l)
    logloss += log_loss([y], pred)

    if train_count % 40000 == 0:
        print("  Model trained on " + str(train_count) + " examples.  LogLoss: " + str(logloss * 1. / train_count) + " \n")

    if train_count == 600000:
        break
    train_count += 1

# Test/predict model
print("\n\n Prediction started at " + str(datetime.now()) + " ... \n")
with open(submission, 'a') as outfile:
    outfile.write('ID,IsClick\n')
    for t, line in enumerate(DictReader(open(test_file), delimiter='\t')):
        id = 0

        try:
            id = int(line['ID'])
            del line['ID']
        except:
            pass
        try:
            del line['IsClick']
        except:
            pass
        try:
            del line['HistCTR']
        except:
            pass

        l = [0.] * len(line)

        for i, key in enumerate(line):
            val = line[key]
            l[i] = (abs(hash(key + '_' + val)) % bits)

        outfile.write("%s,%.15f\n" % (id, model.predict_proba(l)[:,1][0]))

        if test_count % 40000 == 0:
            print("  Predicted " + str(test_count) + " examples \n")
        test_count += 1

print("\n\n Finished at " + str(datetime.now()) + " . \n")

# Save submit
sample = pd.read_csv('../input/sampleSubmission.csv')
index = sample.ID.values - 1
predictions = np.array(pd.read_csv(submission, header=None, dtype=np.string_), dtype=np.string_)
predictions_modified = predictions[1:, 1:]
sample['IsClick'] = predictions_modified[index]
sample.to_csv(submission_final, index=False, dtype=np.string_)