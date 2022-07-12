import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.cross_validation import StratifiedKFold


def Outputs(data):
    return 1.-(1./(1.+np.exp(-data)))


def MungeData(train, test):
    features = train.columns[1:-1]

    ss = StandardScaler()
    targets = train.TARGET.values
    newtrain = train[features].copy()
    newtrain.insert(0, 'ID', train.ID.values)
    newtrain[features] = \
        np.round(ss.fit_transform(train[features].values), 6)
    newtrain['TARGET'] = targets
    del train
    newtest = test[features].copy()
    newtest.insert(0, 'ID', test.ID.values)
    newtest[features] = \
        np.round(ss.transform(test[features].values), 6)
    del test
    return newtrain, newtest


def GPIndividual1(data):
    predictions = (((((5.428570 + data["num_var42"]) + ((data["num_meses_var5_ult3"] + np.abs(np.minimum( (data["num_var13"]),  ((np.maximum( (data["var15"]),  (5.428570)) * data["var15"])))))/2.0))/2.0) + ((data["var15"] < np.sinh((data["saldo_var30"] * 2.0))).astype(float))) +
                    np.tanh((np.minimum( (np.minimum( (np.cos(data["num_var22_ult1"])),  (data["var38"]))),  ((((((data["num_var42"] < data["var38"]).astype(float)) + (-(data["num_var22_ult1"])))/2.0) / 2.0))) + ((data["num_var42_0"] <= data["saldo_var30"]).astype(float)))) +
                    np.sin(np.maximum( (data["ind_var8_0"]),  (np.minimum( (np.maximum( (data["ind_var12_0"]),  ((np.minimum( (data["var38"]),  ((1.0/(1.0 + np.exp(- (data["num_var30"] + np.abs(data["ind_var5_0"]))))))) - np.maximum( (data["num_var42_0"]),  (data["num_var5"])))))),  ((-(np.round(data["num_var5"])))))))) +
                    np.tanh(np.maximum( ((((data["num_aport_var13_hace3"] + (((data["var15"] / 2.0) < (-(0.367879))).astype(float))) + data["saldo_medio_var5_hace3"])/2.0)),  (np.tanh((data["num_aport_var13_hace3"] + (((np.sinh(data["ind_var13_0"]) / 2.0) < data["saldo_var42"]).astype(float))))))) +
                    ((((((data["var15"] >= 2.675680).astype(float)) - ((data["imp_ent_var16_ult1"] >= (np.sin(data["saldo_var5"]) / 2.0)).astype(float))) - ((data["ind_var30"] >= ((data["var15"] <= 0.367879).astype(float))).astype(float))) / 2.0) / 2.0) +
                    ((np.round(data["saldo_var13_corto"]) < ((data["num_var13"] >= ((((data["num_var30"] > data["num_aport_var13_hace3"]).astype(float)) + ((data["var36"] + (data["num_aport_var13_hace3"] / 2.0))/2.0))/2.0)).astype(float))).astype(float)) +
                    np.ceil((np.round(data["num_var42_0"]) * np.minimum( (data["num_aport_var13_ult1"]),  (np.sin((data["imp_var43_emit_ult1"] * np.maximum( (np.minimum( (np.sinh(data["num_var42_0"])),  (data["imp_ent_var16_ult1"]))),  ((np.maximum( (data["imp_var43_emit_ult1"]),  (data["imp_var43_emit_ult1"])) - data["num_var42_0"]))))))))) +
                    np.sin((data["num_var42_0"] * np.maximum( (np.maximum( (data["imp_op_var40_ult1"]),  (((data["saldo_medio_var13_corto_hace2"] + np.tanh(np.floor(data["var38"])))/2.0)))),  (np.sin((data["num_var42_0"] * np.maximum( ((data["var38"] / 2.0)),  (data["imp_var43_emit_ult1"])))))))) +
                    np.sin(((np.maximum( (data["var15"]),  (np.abs(data["var15"]))) < (data["var36"] * data["num_var22_ult1"])).astype(float))) +
                    np.sin((((data["saldo_medio_var13_corto_hace2"] + data["ind_var8_0"])/2.0) * ((data["var38"] >= (data["var36"] * (((data["num_var5_0"] * (data["var38"] - data["ind_var8_0"])) <= np.sin(data["var38"])).astype(float)))).astype(float)))) +
                    (np.minimum( (data["var15"]),  (data["num_aport_var13_hace3"])) * ((data["ind_var8_0"] >= ((np.minimum( (data["var15"]),  (data["saldo_var30"])) * data["num_var42"]) * ((data["var15"] >= np.minimum( (data["imp_ent_var16_ult1"]),  ((data["imp_ent_var16_ult1"] - np.sinh(data["num_var42"]))))).astype(float)))).astype(float))) +
                    (3.141593 * (data["saldo_medio_var5_hace2"] * ((data["imp_ent_var16_ult1"] > np.sin((0.058823 * (data["saldo_medio_var5_hace2"] * np.minimum( (np.floor(np.sin(data["saldo_medio_var5_hace2"]))),  (data["num_var5"])))))).astype(float)))))

    return Outputs(predictions)


def GPIndividual2(data):
    predictions = ((np.sin(data["num_var30"]) + np.maximum( ((2.0 - data["num_var42_0"])),  (((2.302585 - ((data["num_var8_0"] > (((0.434294 >= (-(data["var15"]))).astype(float)) * data["saldo_var30"])).astype(float))) * 2.0)))) +
                    np.minimum( (((data["var38"] + ((data["var38"] > 0.094340).astype(float))) + ((data["num_meses_var5_ult3"] > data["num_var30"]).astype(float)))),  (np.sin(((np.cos(data["var38"]) + (-(((data["num_var22_ult1"] + (data["ind_var30"] * 2.0))/2.0))))/2.0)))) +
                    ((((data["saldo_var30"] > data["num_var42_0"]).astype(float)) + (((data["ind_var39_0"] > (-(data["var15"]))).astype(float)) * ((data["num_var8_0"] * np.tanh(data["saldo_var42"])) - np.sin(data["var15"]))))/2.0) +
                    (((((data["ind_var30"] * np.minimum( (((data["var36"] - data["num_var22_ult1"]) / 2.0)),  ((data["num_var22_ult1"] * data["var15"])))) + data["num_aport_var13_hace3"])/2.0) + ((data["var15"] < np.tanh(np.floor(np.minimum( (data["var36"]),  (data["num_var22_ult1"]))))).astype(float)))/2.0) +
                    (-(((np.abs(data["saldo_var5"]) >= np.abs(((np.minimum( (((data["saldo_var42"] + (((data["num_var13"] != data["ind_var13_corto"]).astype(float)) - data["imp_op_var40_ult1"]))/2.0)),  (((data["num_var42"] >= data["ind_var13_0"]).astype(float)))) + data["ind_var13_0"])/2.0))).astype(float)))) +
                    np.maximum( ((np.abs(data["num_var30"]) - 2.212120)),  (((((data["imp_ent_var16_ult1"] < ((data["num_var30"] >= ((data["saldo_var30"] < data["ind_var13_corto_0"]).astype(float))).astype(float))).astype(float)) - ((np.tanh(data["saldo_var30"]) < data["ind_var13_corto_0"]).astype(float))) / 2.0))) +
                    (((data["saldo_medio_var5_hace3"] < data["ind_var30"]).astype(float)) * np.minimum( (data["saldo_var5"]),  (np.minimum( (data["num_op_var40_ult1"]),  (np.floor((np.sin(np.maximum( (((data["saldo_medio_var13_corto_hace2"] + data["num_op_var40_ult1"])/2.0)),  (((data["saldo_medio_var13_corto_hace2"] + data["num_var42"])/2.0)))) * 2.0))))))) +
                    (np.floor(data["ind_var13_corto"]) * ((data["imp_op_var40_ult1"] >= (data["saldo_var13"] * ((data["ind_var13_corto"] > ((data["var15"] + ((((data["saldo_var42"] * 2.0) >= ((data["var15"] + ((2.212120 <= data["num_var13_corto"]).astype(float)))/2.0)).astype(float)) * 2.0))/2.0)).astype(float)))).astype(float))) +
                    np.maximum( (np.maximum( (np.minimum( (0.138462),  (((np.abs(data["var15"]) < 0.434294).astype(float))))),  ((-((np.cos((data["imp_op_var40_ult1"] * 2.0)) * 2.0)))))),  ((-((data["ind_var5"] * ((data["saldo_medio_var5_hace2"] >= 0.602941).astype(float))))))) +
                    ((((data["var15"] <= np.sin((data["num_op_var40_ult1"] - (((-(np.sin(data["num_op_var39_hace3"]))) >= (data["imp_op_var40_ult1"] - data["saldo_var5"])).astype(float))))).astype(float)) * 2.0) * 20.750000) +
                    np.minimum( (np.minimum( ((data["saldo_var30"] * ((((data["var38"] - data["num_var13"]) / 2.0) <= data["num_op_var39_hace3"]).astype(float)))),  (np.sin(((0.602941 - data["saldo_medio_var13_corto_hace2"]) * 2.0))))),  ((np.floor(np.cos(data["delta_imp_aport_var13_1y3"])) * data["var38"]))) +
                    (((data["saldo_medio_var5_hace3"] * data["imp_var43_emit_ult1"]) + ((data["num_meses_var5_ult3"] * np.floor((0.094340 - np.sin(data["saldo_var5"])))) + (np.sin(((data["imp_aport_var13_hace3"] + data["num_var30"])/2.0)) * data["saldo_medio_var5_hace3"])))/2.0))

    return Outputs(predictions)


if __name__ == "__main__":
    print('Started!')
    print('Trained on 80 percent of the Data')
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    print('Data Munging')
    train, test = MungeData(train, test)
    print('Data Munged')
    visibletrain = blindtrain = train
    skf = StratifiedKFold(train.TARGET.values,
                          n_folds=5,
                          shuffle=False,
                          random_state=42)

    for train_index, test_index in skf:
        visibletrain = train.iloc[train_index]
        blindtrain = train.iloc[test_index]
        break
    mother = pd.DataFrame({'GP1': GPIndividual1(visibletrain),
                           'GP2': GPIndividual2(visibletrain)})
    y_pred = mother.mean(axis=1).values
    print('Visible Log Loss:', log_loss(visibletrain.TARGET.values, y_pred))
    print('Visible ROC:', roc_auc_score(visibletrain.TARGET.values, y_pred))
    mother = pd.DataFrame({'GP1': GPIndividual1(blindtrain),
                           'GP2': GPIndividual2(blindtrain)})
    y_pred = mother.mean(axis=1).values
    print('Blind Log Loss:', log_loss(blindtrain.TARGET.values, y_pred))
    print('Blind ROC:', roc_auc_score(blindtrain.TARGET.values, y_pred))
    submission = pd.DataFrame({"ID": blindtrain.ID,
                               "TARGET": blindtrain.TARGET,
                               "PREDICTION": y_pred})
    submission.to_csv("gptrain.csv", index=False)
    mother = pd.DataFrame({'GP1': GPIndividual1(test),
                           'GP2': GPIndividual2(test)})
    y_pred = mother.mean(axis=1).values
    submission = pd.DataFrame({"ID": test.ID, "TARGET": y_pred})
    submission.to_csv("gptest.csv", index=False)
    print('Completed!')