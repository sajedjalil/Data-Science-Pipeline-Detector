# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

pca = np.matrix("""8.21629007e-07  -2.38016031e-08  -3.87654684e-06  -8.91557022e-07
   -3.29758219e-04  -5.46764157e-06   9.99999941e-01  -5.60263298e-09
   -1.46703443e-08  -2.16380087e-08  -1.17986540e-08  -4.62434248e-10
    1.59481111e-09   9.51595228e-05   2.79412459e-08   1.74764611e-08
    1.52132292e-06  -5.97959298e-06  -2.14273804e-08  -3.29803669e-09;
   -1.13304245e-05   1.72890551e-06   4.35512243e-04   1.61252587e-03
    9.99992160e-01   2.50244434e-03   3.29532903e-04   2.80268606e-08
    3.48915374e-07   5.86749143e-06   4.43587327e-07   3.93050030e-07
   -1.21451799e-09   2.53635464e-03   1.62655501e-07   5.30020654e-07
   -3.08393141e-05   2.91459039e-04   7.95698481e-10  -1.00168648e-06;
    3.14990743e-05  -8.58670882e-07   4.62366328e-05   4.06337268e-04
   -2.52759365e-03  -4.19580668e-03  -9.59907283e-05  -2.48223838e-07
   -5.76601361e-06   5.39319276e-07   3.40814605e-07  -5.54906594e-07
    5.13912314e-07   9.99980305e-01   8.51478541e-05   4.03625848e-06
    2.72951755e-04   3.89058425e-03  -3.83230434e-07   1.63357645e-06;
   -9.96444885e-04   7.72244139e-05   1.40641917e-03   7.87020651e-03
   -2.52364361e-03   9.99913974e-01   4.19063578e-06  -6.07106726e-06
    3.51334028e-06   4.35050214e-05  -2.90210968e-06  -1.08697636e-05
   -3.93941637e-06   4.21914177e-03  -5.14410881e-06   1.37078676e-04
    2.61670760e-03  -8.72288544e-03   1.49953186e-05  -3.21916302e-05;
    -1.73726963e-03   8.54184366e-05   2.15401189e-03   2.19731232e-02
   -3.39305280e-04   8.55317758e-03   6.30650071e-06   4.97755864e-06
   -2.24924783e-06   3.97378853e-05   1.04163081e-05   2.31501609e-06
   -1.69067930e-06  -3.86405781e-03  -1.91391921e-05  -2.83674164e-04
    2.17114640e-03   9.99708216e-01  -2.13711270e-07  -2.78865119e-05""")

print(pca)
print(pca.shape)
from sklearn import linear_model

clf = linear_model.SGDClassifier(loss="log")
trains = pd.read_csv("../input/train.csv", chunksize=3000000)

for train in trains:
    print("loaded train")
    clusters = train.hotel_cluster
    train["book_year"] = np.where(train['srch_ci'].isnull(), pd.to_numeric(train['date_time'].str[:4]), pd.to_numeric(train['srch_ci'].str[:4]))
    train["book_month"] = np.where(train['srch_ci'].isnull(), pd.to_numeric(train['date_time'].str[5:7]), pd.to_numeric(train['srch_ci'].str[5:7]))

    training = train.drop(["hotel_cluster", "date_time", "srch_ci", "srch_co", "is_booking", "cnt"], 1)

    training.fillna(0, inplace=True)
    
    # print(training.columns.to_series().groupby(training.dtypes).groups)
    
    training = np.dot(training, pca.T)

    clf.partial_fit(training, clusters, classes=list(range(100)))
test = pd.read_csv("../input/test.csv")
print("loaded test")
test["book_year"] = (np.where(test['srch_ci'].isnull(), pd.to_numeric(test['date_time'].str[:4]), pd.to_numeric(test['srch_ci'].str[:4])))
test["book_month"] = (np.where(test['srch_ci'].isnull(), pd.to_numeric(test['date_time'].str[5:7]), pd.to_numeric(test['srch_ci'].str[5:7])))

test = test.drop(["date_time", "srch_ci", "srch_co", "id"], 1)
test = np.dot(test, pca.T)


# training.fillna(0, inplace=True)
test.fillna(0, inplace=True)

print("filled NA")

y_pred = clf.predict_log_proba(test)

top5 = clf.classes_[np.fliplr(np.argsort(y_pred[:,-5:]))]
print(y_pred.shape)
print("picked top 5")
print(top5.shape)
print(top5)
print(clf.classes_)
import datetime
print('Generate submission...')
now = datetime.datetime.now()
path = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
out = open(path, "w")
index = 0
out.write("id,hotel_cluster\n")
for row in y_pred:
    if (index % 1000000 == 0):
        print(index)
    out.write(str(index))
    out.write(",")
    out.write(" ".join(str(x) for x in row))
    # out.write(str(row))
    out.write("\n")
    index += 1
print("Generated submission")


y_pred = clf.predict_log_proba(test)

