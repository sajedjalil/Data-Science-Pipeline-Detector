import numpy as np
import csv
import sys
import math
import xgboost as xgb

Ntrain = 10506
Ntest = 116293
Nfea = 13
MISSING = 999.0
# Feature: Month, Week, Latitude, Longitude, NumMosq in Nearest Area, Near Dis, TMax, Tmin, Tavg, WaterBub, Dry, StnPressure

Xtrain = np.zeros((Ntrain, Nfea), dtype=np.float32)
Ytrain = []
Xtest = np.zeros((Ntest, Nfea), dtype=np.float32)

train_head = ""
spray_head = ""
weather_head = ""
weather_dic = {}
train_dic = {}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# build weather dic
fi = csv.reader(open("../input/weather.csv"))
weather_head = fi.__next__()
for line in fi:
    # simply discard station 1
    if line[0] == '1':
        continue
    weather_dic[line[1]] = line

# build train dic
fi = csv.reader(open("../input/train.csv"))
train_head = fi.__next__()
for line in fi:
    idx = train_head.index("Date")
    date = line[idx].split('-')
    key = "%s-%d" % (date[1], int(date[2]) / 7)
    if key not in train_dic:
        train_dic[key] = []
    train_dic[key].append(line)

def gen_month(line, head=train_head):
    idx = head.index("Date")
    date = line[idx].split('-')
    return float(date[1])

def gen_week(line, head=train_head):
    idx = head.index("Date")
    date = line[idx].split('-')
    return int(date[1]) * 4 + int(date[2]) / 7

def gen_latitude(line, head=train_head):
    idx = head.index("Latitude")
    return float(line[idx])

def gen_longitude(line, head=train_head):
    idx = head.index("Longitude")
    return float(line[idx])

def gen_tmax(line, head=train_head):
    idx1 = weather_head.index("Tmax")
    idx2 = head.index("Date")
    return float(weather_dic[line[idx2]][idx1])

def gen_tmin(line, head=train_head):
    idx1 = weather_head.index("Tmin")
    idx2 = head.index("Date")
    return float(weather_dic[line[idx2]][idx1])

def gen_tavg(line, head=train_head):
    idx1 = weather_head.index("Tavg")
    idx2 = head.index("Date")
    return float(weather_dic[line[idx2]][idx1])

def gen_water(line, head=train_head):
    idx1 = weather_head.index("DewPoint")
    idx2 = head.index("Date")
    return float(weather_dic[line[idx2]][idx1])

def gen_snow(line, head=train_head):
    idx1 = weather_head.index("WetBulb")
    idx2 = head.index("Date")
    return float(weather_dic[line[idx2]][idx1])

def gen_pressure(line, head=train_head):
    idx1 = weather_head.index("StnPressure")
    idx2 = head.index("Date")
    return float(weather_dic[line[idx2]][idx1])

def gen_moisq(line, head=train_head):
    idx = train_head.index("NumMosquitos")
    idx1 = head.index("Date")
    #idx2 = train_head.index("NumMosquitos")
    idx3 = head.index("Latitude")
    idx4 = head.index("Longitude")
    train_idx3 = train_head.index("Latitude")
    train_idx4 = train_head.index("Longitude")
    lati = float(line[idx3])
    logi = float(line[idx4])
    date = line[idx1].split('-')
    key = "%s-%d" % (date[1], int(date[2]) / 7)
    min_dis = MISSING
    sol = MISSING
    second_dis = MISSING
    sol2 = MISSING
    temp = []
    tmp = {}
    for line in train_dic[key]:
        dis = (float(line[train_idx3]) - lati) ** 2 + (float(line[train_idx4]) - logi) ** 2
        res = int(line[idx])
        temp.append((dis, res))
        if int(dis) not in tmp:
            tmp[int(dis)] = []
        tmp[int(dis)].append(res)
    temp = sorted(temp, key=lambda s:s[0])
    try:
        min_dis = temp[0][0]
        sol = sum(tmp[int(min_dis)])
        sol /= len(tmp[int(min_dis)]) * 1.0
        second_dis = min_dis
        for item in temp:
            if item[0] != second_dis:
                second_dis = item[0]
                break
        sol2 = sum(tmp[int(second_dis)])
        sol2 /= len(tmp[int(second_dis)]) * 1.0
    except:
        pass
    return (min_dis, sol, second_dis, sol2)

# build train
fi = csv.reader(open("../input/train.csv"))
fi.__next__()
i = 0

sum_wneg = 0.0
sum_wpos = 0.0
#print "make training data"
for line in fi:
    Xtrain[i][0] = gen_snow(line)
    Xtrain[i][1] = gen_tavg(line)
    Xtrain[i][2] = gen_tmax(line)
    Xtrain[i][3] = gen_tmin(line)
    Xtrain[i][4] = gen_week(line)
    #Xtrain[i][5] = gen_moisq(line)
    Xtrain[i][6] = gen_month(line)
    Xtrain[i][7] = gen_water(line)
    Xtrain[i][8] = gen_latitude(line)
    Xtrain[i][9] = gen_longitude(line)
    mos = gen_moisq(line)
    Xtrain[i][5] = mos[0]
    Xtrain[i][10] = mos[1]
    Xtrain[i][11] = mos[2]
    Xtrain[i][12] = mos[3]
    label = int(line[train_head.index("WnvPresent")])
    Ytrain.append(label)
    if label == 0:
        sum_wneg += 1.0
    else:
        sum_wpos += 1.0
    i += 1

#print "make test data"
ids = []
fi = csv.reader(open("../input/test.csv"))
test_head = fi.__next__()

i = 0

for line in fi:
    ids.append(line[0])
    Xtest[i][0] = gen_snow(line, test_head)
    Xtest[i][1] = gen_tavg(line, test_head)
    Xtest[i][2] = gen_tmax(line, test_head)
    Xtest[i][3] = gen_tmin(line, test_head)
    Xtest[i][4] = gen_week(line, test_head)
    #Xtrain[i][5] = gen_moisq(line)
    Xtest[i][6] = gen_month(line, test_head)
    Xtest[i][7] = gen_water(line, test_head)
    Xtest[i][8] = gen_latitude(line, test_head)
    Xtest[i][9] = gen_longitude(line, test_head)
    mos = gen_moisq(line, test_head)
    Xtest[i][5] = mos[0]
    Xtest[i][10] = mos[1]
    Xtest[i][11] = mos[2]
    Xtest[i][12] = mos[3]
    i += 1

#print "training"
dtrain = xgb.DMatrix(Xtrain, label=Ytrain, missing = MISSING)
dtest = xgb.DMatrix(Xtest, missing = MISSING)
param = {}
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param['objective'] = 'binary:logitraw'
# scale weight of positive examples
param['scale_pos_weight'] = sum_wneg/sum_wpos
param['eta'] = 0.2
param['max_depth'] = 10
param['eval_metric'] = 'auc'
param['silent'] = 1
param['min_child_weight'] = 100
param['subsample'] = 0.72
param['colsample_bytree'] = 1
param['nthread'] = 4

num_round = 62

#xgb.cv(param, dtrain, num_round, nfold=5)
bst = xgb.train(param, dtrain, num_round)

#print "testing"
ypred = bst.predict(dtest)

fo = csv.writer(open("submission.csv", "w"), lineterminator="\n")
fo.writerow(["Id","WnvPresent"])
i = 0
for item in ids:
    fo.writerow([ids[i], sigmoid(ypred[i])])
    i += 1



