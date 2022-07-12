# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 



#The idea behind this script is to use the geometry files given in the data. It is based around the idea that for
#individual element we will have variation on different axis i.e. X, Y and Z. And these variations are given multiple
# number of times in a single geometry file, probably telling us about variations on different front. I have used
# PCA to reduce that into teo features per axis(X, Y, Z).

# PS: I have no background in chemistry so I can't back it up by solid theoretical proofs.


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import lightgbm as lgb
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import pandas as pd
import numpy as np
import re
from sklearn.decomposition import PCA

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

def get_xyz_data(filename):
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)
    


ga_cols = []
al_cols = []
o_cols = []
in_cols = []

import warnings
warnings.filterwarnings("ignore")

for i in range(6):
    ga_cols.append("Ga_"+str(i))

for i in range(6):
    al_cols.append("Al_"+str(i))

for i in range(6):
    o_cols.append("O_"+str(i))

for i in range(6):
    in_cols.append("In_"+str(i))



ga_df= pd.DataFrame(columns=ga_cols)
al_df = pd.DataFrame(columns=al_cols)
o_df = pd.DataFrame(columns= o_cols)
in_df = pd.DataFrame(columns=in_cols)

for i in train.id.values:
    fn = "../input/train/{}/geometry.xyz".format(i)
    train_xyz, train_lat = get_xyz_data(fn)
    
    ga_list = []
    al_list = []
    o_list = []
    in_list = []
    
    for li in train_xyz:
        try:
            if li[1] == "Ga":
                ga_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "Al":
                al_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "In":
                in_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "O":
                o_list.append(li[0])
        except:
            pass
    
#     ga_list = [item for sublist in ga_list for item in sublist]
#     al_list = [item for sublist in al_list for item in sublist]
#     o_list = [item for sublist in o_list for item in sublist]
   
    
    try:
        model = PCA(n_components=2)
        ga_list = np.array(ga_list)
        temp_ga = model.fit_transform(ga_list.transpose())
        temp_ga = [item for sublist in temp_ga for item in sublist]
       
    except:
        temp_ga = [0,0,0,0,0,0]
#         print i
    try:
        model = PCA(n_components=2)
        al_list = np.array(al_list)
        temp_al = model.fit_transform(al_list.transpose())
        temp_al = [item for sublist in temp_al for item in sublist]
#         print i
    except:
        temp_al = [0,0,0,0,0,0]
#         print i
    try:
        model = PCA(n_components=2)
        o_list = np.array(o_list)
        temp_o = model.fit_transform(o_list.transpose())
        temp_o = [item for sublist in temp_o for item in sublist]
#         print i
    except:
        temp_o = [0,0,0,0,0,0]
#         print i
    
    try:
        model = PCA(n_components=2)
        in_list = np.array(in_list)
        temp_in = model.fit_transform(in_list.transpose())
        temp_in = [item for sublist in temp_in for item in sublist]
#         print i
    except:
        temp_in = [0,0,0,0,0,0]
#         print i

    temp_ga = pd.DataFrame(temp_ga).transpose()
    temp_ga.columns = ga_cols
    temp_ga.index = np.array([i])

    temp_al = pd.DataFrame(temp_al).transpose()
    temp_al.columns = al_cols
    temp_al.index = np.array([i])

    temp_o = pd.DataFrame(temp_o).transpose()
    temp_o.columns = o_cols
    temp_o.index = np.array([i])
    
    temp_in = pd.DataFrame(temp_in).transpose()
    temp_in.columns = in_cols
    temp_in.index = np.array([i])
    
    

    ga_df = pd.concat([ga_df,temp_ga])
    al_df = pd.concat([al_df,temp_al])
    o_df = pd.concat([o_df,temp_o])    
    in_df = pd.concat([in_df,temp_in])
    
ga_df["id"] = ga_df.index
al_df["id"] = al_df.index
o_df["id"] = o_df.index
in_df["id"] = in_df.index

train = pd.merge(train,ga_df,on = ["id"],how = "left")
train = pd.merge(train,al_df,on = ["id"],how = "left")
train = pd.merge(train,o_df,on = ["id"],how = "left")
train = pd.merge(train,in_df,on = ["id"],how = "left")
    
ga_df= pd.DataFrame(columns=ga_cols)
al_df = pd.DataFrame(columns=al_cols)
o_df = pd.DataFrame(columns= o_cols)
in_df = pd.DataFrame(columns=in_cols)    

for i in test.id.values:
    fn = "../input/test/{}/geometry.xyz".format(i)
    train_xyz, train_lat = get_xyz_data(fn)
    
    ga_list = []
    al_list = []
    o_list = []
    in_list = []
    
    for li in train_xyz:
        try:
            if li[1] == "Ga":
                ga_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "Al":
                al_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "In":
                in_list.append(li[0])
        except:
            pass
        try:
            if li[1] == "O":
                o_list.append(li[0])
        except:
            pass
    
#     ga_list = [item for sublist in ga_list for item in sublist]
#     al_list = [item for sublist in al_list for item in sublist]
#     o_list = [item for sublist in o_list for item in sublist]
   
    
    try:
        model = PCA(n_components=2)
        ga_list = np.array(ga_list)
        temp_ga = model.fit_transform(ga_list.transpose())
        temp_ga = [item for sublist in temp_ga for item in sublist]
       
    except:
        temp_ga = [0,0,0,0,0,0]
#         print i
    try:
        model = PCA(n_components=2)
        al_list = np.array(al_list)
        temp_al = model.fit_transform(al_list.transpose())
        temp_al = [item for sublist in temp_al for item in sublist]
#         print i
    except:
        temp_al = [0,0,0,0,0,0]
#         print i
    try:
        model = PCA(n_components=2)
        o_list = np.array(o_list)
        temp_o = model.fit_transform(o_list.transpose())
        temp_o = [item for sublist in temp_o for item in sublist]
#         print i
    except:
        temp_o = [0,0,0,0,0,0]
#         print i
    
    try:
        model = PCA(n_components=2)
        in_list = np.array(in_list)
        temp_in = model.fit_transform(in_list.transpose())
        temp_in = [item for sublist in temp_in for item in sublist]
#         print i
    except:
        temp_in = [0,0,0,0,0,0]
#         print i

    temp_ga = pd.DataFrame(temp_ga).transpose()
    temp_ga.columns = ga_cols
    temp_ga.index = np.array([i])

    temp_al = pd.DataFrame(temp_al).transpose()
    temp_al.columns = al_cols
    temp_al.index = np.array([i])

    temp_o = pd.DataFrame(temp_o).transpose()
    temp_o.columns = o_cols
    temp_o.index = np.array([i])
    
    temp_in = pd.DataFrame(temp_in).transpose()
    temp_in.columns = in_cols
    temp_in.index = np.array([i])
    
    

    ga_df = pd.concat([ga_df,temp_ga])
    al_df = pd.concat([al_df,temp_al])
    o_df = pd.concat([o_df,temp_o])    
    in_df = pd.concat([in_df,temp_in])
    

ga_df["id"] = ga_df.index
al_df["id"] = al_df.index
o_df["id"] = o_df.index
in_df["id"] = in_df.index

test = pd.merge(test,ga_df,on = ["id"],how = "left")
test = pd.merge(test,al_df,on = ["id"],how = "left")
test = pd.merge(test,o_df,on = ["id"],how = "left")
test = pd.merge(test,in_df,on = ["id"],how = "left")



X_train = train.head(2200)

X_val = train.tail(200)

y_train_1 = X_train['formation_energy_ev_natom']
y_train_2 = X_train["bandgap_energy_ev"]

y_val_1 = X_val['formation_energy_ev_natom']
y_val_2 = X_val["bandgap_energy_ev"]


X_train = X_train.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis =1)
X_val = X_val.drop(["formation_energy_ev_natom","bandgap_energy_ev"],axis =1)


X_train.In_0 = X_train.In_0.astype("float")
X_train.In_1 = X_train.In_1.astype("float")
X_train.In_2 = X_train.In_2.astype("float")
X_train.In_3 = X_train.In_3.astype("float")
X_train.In_4 = X_train.In_4.astype("float")
X_train.In_5 = X_train.In_5.astype("float")

X_val.In_0 = X_val.In_0.astype("float")
X_val.In_1 = X_val.In_1.astype("float")
X_val.In_2 = X_val.In_2.astype("float")
X_val.In_3 = X_val.In_3.astype("float")
X_val.In_4 = X_val.In_4.astype("float")
X_val.In_5 = X_val.In_5.astype("float")


test.In_0 = test.In_0.astype("float")
test.In_1 = test.In_1.astype("float")
test.In_2 = test.In_2.astype("float")
test.In_3 = test.In_3.astype("float")
test.In_4 = test.In_4.astype("float")
test.In_5 = test.In_5.astype("float")


params = {
    'num_leaves': 7,
    'objective': 'regression',
    'min_data_in_leaf': 18,
    'learning_rate': 0.05,
    'feature_fraction': 0.93,
    'bagging_fraction': 0.93,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 1
}

MAX_ROUNDS = 1500
val_pred = []
test_pred_1 = []
cate_vars = []

dtrain = lgb.Dataset(X_train.drop("id",axis = 1), label=y_train_1,categorical_feature=cate_vars)

dval = lgb.Dataset(X_val.drop("id",axis = 1), label=y_val_1, reference=dtrain,categorical_feature=cate_vars)

bst = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100)

print("\n".join(("%s: %.2f" % x) for x in sorted(zip(X_train.columns, bst.feature_importance("gain")),key=lambda x: x[1], reverse=True)))

val_pred.append(bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))

test_pred_1.append(bst.predict(test.drop("id",axis =1), num_iteration=bst.best_iteration or MAX_ROUNDS))



params = {
    'num_leaves': 8,
    'objective': 'regression',
    'min_data_in_leaf': 18,
    'learning_rate': 0.05,
    'feature_fraction': 0.93,
    'bagging_fraction': 0.93,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 1
}

MAX_ROUNDS = 1500
val_pred = []
test_pred_2 = []
cate_vars = []

dtrain = lgb.Dataset(X_train.drop("id",axis = 1), label=y_train_2,categorical_feature=cate_vars)

dval = lgb.Dataset(X_val.drop("id",axis = 1), label=y_val_2, reference=dtrain,categorical_feature=cate_vars)

bst = lgb.train(params, dtrain, num_boost_round=MAX_ROUNDS,valid_sets=[dtrain, dval], early_stopping_rounds=50, verbose_eval=100)

print("\n".join(("%s: %.2f" % x) for x in sorted(zip(X_train.columns, bst.feature_importance("gain")),key=lambda x: x[1], reverse=True)))

val_pred.append(bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))

test_pred_2.append(bst.predict(test.drop("id",axis =1), num_iteration=bst.best_iteration or MAX_ROUNDS))


sample = pd.read_csv("../input/sample_submission.csv")

sample["formation_energy_ev_natom"] = test_pred_1[0]
sample["bandgap_energy_ev"] = test_pred_2[0]

sample.to_csv("sub.csv",index = False)