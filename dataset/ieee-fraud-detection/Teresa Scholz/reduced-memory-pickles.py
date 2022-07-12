# Placeholder to get datasets.
import pickle
X_train = pickle.load(open('/kaggle/input/reduced-memory/X_train.pkl', 'rb'))
X_test = pickle.load(open('/kaggle/input/reduced-memory/X_test.pkl', 'rb'))
y_train = pickle.load(open('/kaggle/input/reduced-memory/y_train.pickle', 'rb'))



# Data was joined and then the memory reduced:
#train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
#test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
