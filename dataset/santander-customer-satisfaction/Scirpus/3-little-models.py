import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize


def Outputs(data):
    return 1.-(1./(1.+np.exp(-data)))


def drop_sparse(train, test):
    remove = []
    c = train.columns[1:-1]
    for i in range(len(c)-1):
        v = train[c[i]].values
        for j in range(i+1, len(c)):
            if np.array_equal(v, train[c[j]].values):
                remove.append(c[j])

    train.drop(remove, axis=1, inplace=True)
    test.drop(remove, axis=1, inplace=True)

    remove = []
    for col in train.columns:
        if train[col].std() == 0:
            remove.append(col)
    return train, test


def drop_duplicates(train):
    features = train.columns[1:-1]
    train.drop_duplicates(features, keep=False, inplace=True)
    return train


def MungeData(train, test):
    features = train.columns[1:-1]
    train.insert(1, 'SumZeros', (train[features] == 0).astype(int).sum(axis=1))
    test.insert(1, 'SumZeros', (test[features] == 0).astype(int).sum(axis=1))
    train.var38 = np.log(train.var38)
    test.var38 = np.log(test.var38)
    print('drop_sparse')
    train, test = drop_sparse(train, test)
    print('drop_duplicates')
    train = drop_duplicates(train)
    features = train.columns[1:-1]
    # Add PCA Components
    pca = PCA(n_components=2)
    x_train_projected = pca.fit_transform(normalize(train[features], axis=0))
    x_test_projected = pca.transform(normalize(test[features], axis=0))
    train.insert(1, 'PCA_0', x_train_projected[:, 0])
    train.insert(1, 'PCA_1', x_train_projected[:, 1])
    test.insert(1, 'PCA_0', x_test_projected[:, 0])
    test.insert(1, 'PCA_1', x_test_projected[:, 1])
    features = train.columns[1:-1]
    ss = StandardScaler()
    train[features] = np.round(ss.fit_transform(train[features]), 6)
    test[features] = np.round(ss.transform(test[features]), 6)
    return train, test


def GPIndividual1(data):
    predictions = ((((2.675680 + ((data["num_var13"] > data["var15"]).astype(float))) + ((data["num_var30_0"] >= (data["var15"] - data["num_var30_0"])).astype(float))) + np.tanh(data["num_var30"])) +
                  np.tanh((np.tanh((((data["saldo_var30"] > data["num_var42_0"]).astype(float)) + (((data["var38"] - data["imp_op_var39_efect_ult3"]) + data["saldo_var24"])/2.0))) + (data["saldo_medio_var5_ult3"] - data["ind_var30_0"]))) +
                  (((data["num_var22_ult3"] + (((data["ind_var30_0"] + data["num_var42_0"])/2.0) + data["ind_var8_0"]))/2.0) * np.minimum( (np.minimum( (data["saldo_var30"]),  (np.cos(data["num_var13_corto_0"])))),  (np.minimum( (np.tanh(data["ind_var30_0"])),  (data["var15"]))))) +
                  (((data["num_meses_var5_ult3"] >= np.round(np.ceil(data["num_var30"]))).astype(float)) + np.tanh(np.tanh(((data["saldo_medio_var8_ult1"] + ((-(((data["num_var22_ult1"] + ((data["num_var30"] >= np.round(np.round(np.ceil(data["num_meses_var5_ult3"])))).astype(float)))/2.0))) / 2.0))/2.0)))) +
                  ((((data["saldo_medio_var8_hace3"] + ((((data["saldo_medio_var5_ult1"] > data["imp_trans_var37_ult1"]).astype(float)) + ((data["saldo_medio_var5_hace2"] > np.sinh(data["saldo_medio_var12_ult1"])).astype(float)))/2.0))/2.0) + data["num_meses_var13_largo_ult3"]) + (-(((data["num_op_var40_efect_ult3"] >= (((data["imp_amort_var34_ult1"] != data["num_meses_var13_largo_ult3"]).astype(float)) / 2.0)).astype(float))))) +
                  ((((data["var3"] <= np.sin(((-((1.0/(1.0 + np.exp(- ((data["var15"] < data["num_ent_var16_ult1"]).astype(float))))))) - data["var15"]))).astype(float)) + (-(((data["num_ent_var16_ult1"] >= (((-(data["var3"])) + data["imp_sal_var16_ult1"])/2.0)).astype(float)))))/2.0) +
                  ((np.sin(np.floor((((data["saldo_medio_var12_hace3"] + data["ind_var13_corto_0"])/2.0) * ((data["var15"] <= np.minimum( (data["ind_var13_corto_0"]),  (np.round(np.minimum( (data["var15"]),  (data["ind_var30_0"])))))).astype(float))))) + np.floor(np.cos((data["num_var30"] + data["ind_var30_0"]))))/2.0) +
                  np.maximum( (((((data["saldo_medio_var8_ult1"] >= np.maximum( ((-(np.sin(data["saldo_medio_var12_hace3"])))),  (data["num_var1_0"]))).astype(float)) + ((data["ind_var31_0"] <= np.minimum( (data["saldo_medio_var12_hace2"]),  (data["ind_var12"]))).astype(float))) / 2.0)),  (data["ind_var20_0"])) +
                  np.minimum( (np.sinh(np.minimum( (np.tanh((data["ind_var30_0"] + np.tanh((data["imp_op_var41_ult1"] * data["num_var7_recib_ult1"]))))),  (np.tanh((data["num_sal_var16_ult1"] * data["saldo_var30"])))))),  ((data["imp_op_var41_ult1"] * ((data["ind_var33_0"] <= data["num_var30_0"]).astype(float))))) +
                  np.maximum( (np.abs(data["num_venta_var44_ult1"])),  (np.minimum( (data["ind_var43_emit_ult1"]),  (np.abs((((data["num_var31_0"] + data["num_op_var39_hace3"]) + ((data["num_var30"] < (data["num_op_var39_hace3"] + ((data["saldo_medio_var5_hace3"] >= ((data["num_reemb_var13_ult1"] == data["ind_var43_emit_ult1"]).astype(float))).astype(float)))).astype(float)))/2.0)))))) +
                  np.minimum( ((data["imp_reemb_var17_ult1"] * data["num_meses_var33_ult3"])),  (np.minimum( ((data["num_var13_largo"] * (data["saldo_medio_var8_ult1"] * (1.0/(1.0 + np.exp(- data["num_var13_largo"])))))),  ((data["ind_var10cte_ult1"] + (((data["imp_trans_var37_ult1"] * 2.0) >= np.minimum( (data["num_op_var41_efect_ult3"]),  (data["imp_reemb_var17_ult1"]))).astype(float))))))) +
                  ((data["num_trasp_var17_in_ult1"] >= (data["imp_op_var41_comer_ult3"] * (((data["imp_op_var39_comer_ult1"] + ((data["saldo_medio_var17_hace2"] >= (data["num_var35"] * ((data["saldo_medio_var5_hace3"] > data["num_trasp_var17_in_hace3"]).astype(float)))).astype(float))) > np.maximum( (np.tanh(data["num_var13_largo_0"])),  (data["saldo_medio_var44_hace3"]))).astype(float)))).astype(float)) +
                  (-(np.minimum( (((data["var15"] > np.cos(data["num_meses_var39_vig_ult3"])).astype(float))),  (np.minimum( ((1.0/(1.0 + np.exp(- (data["saldo_var14"] - ((data["num_meses_var39_vig_ult3"] >= ((data["saldo_medio_var44_ult1"] == data["saldo_var5"]).astype(float))).astype(float))))))),  (((data["num_var45_ult3"] + ((data["num_meses_var39_vig_ult3"] >= data["num_var7_recib_ult1"]).astype(float)))/2.0))))))) +
                  np.minimum( (np.maximum( (data["ind_var41_0"]),  (data["var15"]))),  (np.sin((np.minimum( (np.maximum( (data["ind_var41_0"]),  (data["num_var40_0"]))),  (((data["saldo_medio_var5_ult1"] > data["num_var40_0"]).astype(float)))) * ((data["saldo_medio_var5_ult1"] < data["num_meses_var5_ult3"]).astype(float)))))) +
                  np.maximum( (((data["num_var5_0"] + (data["num_var22_hace2"] + data["ind_var9_ult1"])) * data["num_meses_var17_ult3"])),  (np.minimum( (data["SumZeros"]),  (np.tanh(data["num_var22_ult1"]))))) +
                  np.maximum( (np.maximum( ((np.sinh(((data["num_var22_hace3"] > data["imp_aport_var17_hace3"]).astype(float))) * data["saldo_var14"])),  (np.minimum( (np.maximum( (data["ind_var13_largo_0"]),  (np.round(((data["saldo_medio_var8_hace2"] + data["num_meses_var13_corto_ult3"])/2.0))))),  (data["num_var1_0"]))))),  (np.minimum( (data["ind_var24_0"]),  (np.floor(data["num_var40"]))))))

    return Outputs(predictions)

def GPIndividual2(data):
    predictions = ((((2.675680 + ((data["var15"] <= data["PCA_1"]).astype(float))) + ((data["num_var30"] + np.maximum( (data["var38"]),  (data["num_var30"])))/2.0)) + ((data["var15"] <= (data["num_meses_var12_ult3"] * 2.0)).astype(float))) +
                  ((data["ind_var30_0"] * (np.minimum( (data["saldo_var5"]),  (np.cos(data["num_var30"]))) * 2.0)) + np.tanh((np.tanh((np.abs(data["ind_var13"]) - ((data["imp_op_var39_efect_ult3"] + ((data["ind_var24_0"] > (data["saldo_medio_var5_ult1"] * 2.0)).astype(float)))/2.0))) * 2.0))) +
                  (np.round(np.minimum( (np.maximum( (np.maximum( (data["saldo_medio_var5_hace2"]),  (data["saldo_var8"]))),  (((data["saldo_medio_var5_ult1"] >= data["imp_aport_var13_ult1"]).astype(float))))),  (np.cos(np.maximum( (data["num_var30"]),  (((-(np.tanh(data["saldo_var8"]))) * 2.0))))))) - ((data["num_var22_ult3"] / 2.0) / 2.0)) +
                  ((((data["imp_trasp_var33_in_hace3"] > np.tanh((data["var15"] + 0.730769))).astype(float)) + ((np.floor(data["ind_var20_0"]) + ((data["num_ent_var16_ult1"] < ((data["var15"] + data["ind_var20_0"])/2.0)).astype(float)))/2.0))/2.0) +
                  (((((2.718282 < (data["var15"] + data["saldo_medio_var8_hace3"])).astype(float)) + np.minimum( (np.cos(data["var38"])),  (((data["var15"] < (-(0.636620))).astype(float)))))/2.0) - ((data["num_op_var40_efect_ult3"] > 2.718282).astype(float))) +
                  ((((data["num_meses_var13_largo_ult3"] + ((data["saldo_medio_var5_hace3"] >= data["num_meses_var13_largo_ult3"]).astype(float)))/2.0) + ((((data["var36"] + np.floor(data["num_meses_var5_ult3"]))/2.0) * (data["ind_var30"] * ((((data["var36"] > data["saldo_medio_var8_hace3"]).astype(float)) >= data["var36"]).astype(float)))) + data["saldo_medio_var8_hace3"]))/2.0) +
                  ((((data["saldo_medio_var12_hace2"] > np.cos(data["num_op_var39_hace3"])).astype(float)) + ((data["var3"] <= np.minimum( (data["saldo_medio_var8_ult1"]),  (np.ceil(data["saldo_var8"])))).astype(float)))/2.0) +
                  (-(((data["num_sal_var16_ult1"] > ((data["saldo_var5"] - data["ind_var14"]) - ((data["imp_var7_recib_ult1"] >= (data["var38"] * np.minimum( (data["saldo_var17"]),  (((data["imp_compra_var44_hace3"] >= np.floor(((data["ind_var2_0"] + data["ind_var14"])/2.0))).astype(float)))))).astype(float)))).astype(float)))) +
                  np.maximum( (np.tanh(np.tanh((-(((data["num_med_var22_ult3"] >= (-((((data["imp_reemb_var17_ult1"] > data["var15"]).astype(float)) + np.minimum( (data["var15"]),  (np.abs(data["saldo_medio_var17_hace3"]))))))).astype(float))))))),  (np.round(np.minimum( (data["var15"]),  (data["saldo_var14"]))))) +
                  np.sin(((np.minimum( (0.058823),  (data["num_var42"])) * ((data["delta_num_venta_var44_1y3"] >= ((data["delta_num_venta_var44_1y3"] + data["num_meses_var39_vig_ult3"])/2.0)).astype(float))) * np.floor(np.floor(((np.minimum( (data["num_med_var45_ult3"]),  (data["ind_var5_0"])) + data["num_meses_var39_vig_ult3"])/2.0))))) +
                  (0.138462 * np.maximum( (np.maximum( (np.abs(data["saldo_medio_var8_hace2"])),  ((data["num_var45_hace3"] * np.minimum( (1.732051),  (np.minimum( (data["num_meses_var5_ult3"]),  (data["num_var13_largo"])))))))),  (((np.maximum( (data["var15"]),  (np.sinh(data["num_op_var40_hace3"]))) <= data["num_meses_var5_ult3"]).astype(float))))) +
                  np.sinh((((data["num_var33_0"] * np.maximum( (data["imp_reemb_var17_ult1"]),  (np.round(data["num_var45_ult1"])))) + np.minimum( (data["saldo_medio_var13_largo_ult1"]),  (np.sin(((data["num_aport_var13_ult1"] + np.maximum( (np.round(np.maximum( (data["imp_reemb_var17_ult1"]),  (np.round(data["num_var45_ult1"]))))),  (data["imp_op_var41_comer_ult1"])))/2.0)))))/2.0)) +
                  (np.ceil(np.sin(np.maximum( (data["saldo_medio_var12_hace3"]),  (np.maximum( (np.ceil(np.sin(np.maximum( (data["saldo_medio_var12_hace3"]),  (data["imp_ent_var16_ult1"]))))),  (np.sinh(data["num_var14_0"]))))))) * data["saldo_var12"]) +
                  np.minimum( ((np.maximum( (1.630430),  (data["num_var22_hace3"])) - data["num_var30"])),  (((data["PCA_0"] > np.ceil((data["ind_var17_0"] * ((data["saldo_medio_var5_hace3"] + (data["num_var7_recib_ult1"] * ((1.630430 < (data["saldo_var13_largo"] / 2.0)).astype(float))))/2.0)))).astype(float)))) +
                  np.maximum( (data["delta_num_venta_var44_1y3"]),  (np.maximum( (np.maximum( (((np.minimum( (data["imp_op_var41_efect_ult3"]),  (data["num_trasp_var11_ult1"])) > ((((((data["var3"] <= data["imp_var7_recib_ult1"]).astype(float)) <= data["num_med_var22_ult3"]).astype(float)) >= data["saldo_var26"]).astype(float))).astype(float))),  (data["ind_var18_0"]))),  ((-((data["num_var22_hace3"] * data["imp_var7_recib_ult1"]))))))) +
                  np.minimum( (np.floor(np.cos(((data["num_var33"] + ((data["imp_op_var39_efect_ult3"] < (data["imp_op_var39_efect_ult1"] - np.sin(((data["imp_op_var39_efect_ult3"] > data["saldo_var33"]).astype(float))))).astype(float)))/2.0)))),  (np.cos(((np.tanh(data["imp_op_var39_efect_ult3"]) + (data["saldo_var40"] - data["imp_op_var39_efect_ult3"]))/2.0)))))
    return Outputs(predictions)


def GPIndividual3(data):
    predictions = ((np.maximum( (data["ind_var12"]),  (np.maximum( (data["num_var13_corto_0"]),  ((2.718282 - (np.minimum( (data["var15"]),  (((data["saldo_var6"] + ((data["num_var43_emit_ult1"] == data["ind_var30"]).astype(float)))/2.0))) * 2.0)))))) + np.sin(data["num_var30"])) +
                  np.tanh((((data["saldo_var5"] > data["ind_var13_largo"]).astype(float)) + (((data["saldo_var5"] > np.round((data["var15"] - data["saldo_var5"]))).astype(float)) + (((data["var38"] - data["imp_op_var39_efect_ult3"]) + (data["ind_var13_largo_0"] - data["ind_var30_0"]))/2.0)))) +
                  np.tanh((data["saldo_medio_var13_largo_ult3"] + (((data["saldo_var8"] - ((np.minimum( ((data["num_var22_ult1"] * data["num_var30"])),  (data["ind_var30_0"])) + np.maximum( (data["num_var22_ult3"]),  (data["num_var22_ult1"])))/2.0)) + (data["saldo_medio_var5_hace2"] + data["saldo_medio_var17_ult1"]))/2.0))) +
                  (((data["var36"] <= (-(((data["num_var30"] >= ((((data["saldo_medio_var5_hace3"] >= data["imp_reemb_var17_hace3"]).astype(float)) + data["num_var13_largo_0"])/2.0)).astype(float))))).astype(float)) - ((data["imp_op_var40_efect_ult3"] >= (np.abs(data["ind_var13_largo"]) + data["saldo_var5"])).astype(float))) +
                  (((data["saldo_medio_var8_hace3"] + np.tanh(((data["saldo_var8"] * np.maximum( (data["num_var22_ult3"]),  (((data["num_ent_var16_ult1"] > data["num_ent_var16_ult1"]).astype(float))))) - ((data["num_ent_var16_ult1"] > np.cos(data["num_var30"])).astype(float)))))/2.0) - ((data["num_sal_var16_ult1"] >= data["saldo_medio_var29_ult1"]).astype(float))) +
                  np.tanh((data["ind_var8_0"] * np.minimum( (((data["var15"] + data["var3"])/2.0)),  ((-(((((-(data["saldo_medio_var13_largo_ult3"])) > data["saldo_medio_var8_ult1"]).astype(float)) * ((data["ind_var8"] >= np.sin(np.sin(data["delta_imp_trasp_var17_out_1y3"]))).astype(float))))))))) +
                  np.maximum( (np.sin(np.minimum( ((((data["saldo_medio_var12_ult1"] < data["saldo_medio_var5_hace2"]).astype(float)) / 2.0)),  ((((data["saldo_medio_var5_ult3"] > data["num_var14_0"]).astype(float)) - data["var15"]))))),  (np.maximum( (data["num_meses_var13_largo_ult3"]),  (data["ind_var6_0"])))) +
                  (0.138462 * (((-(np.ceil(data["var38"]))) * ((data["saldo_var5"] < np.minimum( (0.138462),  (data["num_var5"]))).astype(float))) - ((data["num_var5"] > ((data["num_var40_0"] <= data["saldo_medio_var5_hace2"]).astype(float))).astype(float)))) +
                  (np.sin((data["saldo_medio_var5_hace2"] * np.minimum( (data["ind_var14_0"]),  (np.maximum( (data["num_var45_ult3"]),  (np.sinh(np.minimum( (np.minimum( (data["num_venta_var44_ult1"]),  (np.maximum( (data["ind_var14_0"]),  (data["num_var45_hace3"]))))),  (data["num_meses_var39_vig_ult3"]))))))))) - (-((data["imp_reemb_var17_ult1"] * data["saldo_var32"])))) +
                  np.maximum( (data["delta_num_venta_var44_1y3"]),  (((data["ind_var30"] >= (1.0/(1.0 + np.exp(- np.tanh((1.0/(1.0 + np.exp(- np.sinh(((data["saldo_medio_var29_ult1"] + data["var15"]) + np.sin(((data["num_var45_hace2"] >= np.ceil(((data["saldo_medio_var13_corto_hace3"] > 1.197370).astype(float)))).astype(float))))))))))))).astype(float)))) +
                  np.maximum( (data["ind_var20_0"]),  ((np.minimum( (data["var15"]),  (((data["saldo_medio_var13_corto_hace3"] >= np.maximum( (data["saldo_medio_var13_corto_hace3"]),  (((data["ind_var30_0"] + np.cos(((data["var15"] + np.ceil(np.cos(np.abs(data["saldo_medio_var12_hace3"]))))/2.0)))/2.0)))).astype(float)))) / 2.0))) +
                  (data["num_aport_var17_ult1"] * np.maximum( (data["num_var22_hace2"]),  ((((np.maximum( (data["num_var22_hace2"]),  ((np.maximum( (data["num_var22_ult1"]),  (data["saldo_medio_var13_largo_hace3"])) * data["num_var24"]))) != np.tanh(data["ind_var9_cte_ult1"])).astype(float)) * np.maximum( (data["num_op_var39_ult1"]),  ((data["num_var30"] * data["saldo_medio_var8_ult3"]))))))) +
                  np.maximum( ((np.minimum( (data["saldo_medio_var8_hace2"]),  (((data["num_var37_med_ult2"] > 2.212120).astype(float)))) + data["imp_compra_var44_hace3"])),  (np.abs(np.round(np.minimum( (np.cos(data["num_op_var39_hace3"])),  (((2.212120 < np.round(data["num_var30"])).astype(float)))))))) +
                  ((data["saldo_medio_var12_hace3"] * 2.0) * (data["saldo_medio_var12_ult3"] * np.round((data["saldo_medio_var12_ult3"] * (data["saldo_medio_var12_ult3"] * ((np.sin(np.sin(data["saldo_medio_var12_hace3"])) > data["ind_var1"]).astype(float))))))) +
                  np.minimum( (np.maximum( (data["saldo_var5"]),  (np.ceil(((data["saldo_medio_var12_hace3"] == data["ind_var10cte_ult1"]).astype(float)))))),  (np.tanh(((data["ind_var10cte_ult1"] + np.floor(((data["num_op_var41_efect_ult3"] < (-(data["ind_var12"]))).astype(float))))/2.0)))) +
                  (data["ind_var24_0"] * ((data["num_op_var40_ult3"] >= (data["num_var45_hace3"] * np.minimum( (data["imp_aport_var33_ult1"]),  (np.minimum( (np.maximum( (data["num_var35"]),  (data["num_var24"]))),  (np.sinh(np.ceil(np.tanh(data["num_var35"]))))))))).astype(float))))
    return Outputs(predictions)


if __name__ == "__main__":
    print('Started!')
    INPUT_PATH = '../input/'
    train = pd.read_csv(INPUT_PATH + 'train.csv')
    test = pd.read_csv(INPUT_PATH + 'test.csv')
    train, test = MungeData(train, test)
    print('ROC1:', roc_auc_score(train.TARGET.values, GPIndividual1(train)))
    print('ROC2:', roc_auc_score(train.TARGET.values, GPIndividual2(train)))
    print('ROC3:', roc_auc_score(train.TARGET.values, GPIndividual3(train)))
    print('ROC:', roc_auc_score(train.TARGET.values,
                                (GPIndividual1(train) +
                                 GPIndividual2(train) +
                                 GPIndividual3(train))/3))

    x = (GPIndividual1(train) +
         GPIndividual2(train) +
         GPIndividual3(train))/3
    x = np.maximum(0, np.minimum(x, 1))
    print('Ari Train Log Loss:', log_loss(train.TARGET.values, x.values))
    print('Ari Train ROC:', roc_auc_score(train.TARGET.values, x.values))
    submission = pd.DataFrame({"ID": train.ID,
                               "PREDICTION": x,
                               "TARGET": train.TARGET})
    submission.to_csv("arigptrainsubmission.csv", index=False)
    x = np.power(GPIndividual1(train) *
                 GPIndividual2(train) *
                 GPIndividual3(train), 1./3)
    x = np.maximum(0, np.minimum(x, 1))
    print('Geo Train Log Loss:', log_loss(train.TARGET.values, x.values))
    print('Geo Train ROC:', roc_auc_score(train.TARGET.values, x.values))
    submission = pd.DataFrame({"ID": train.ID,
                               "PREDICTION": x,
                               "TARGET": train.TARGET})
    submission.to_csv("geogptrainsubmission.csv", index=False)

    x = (GPIndividual1(test) +
         GPIndividual2(test) +
         GPIndividual3(test))/3
    x = np.maximum(0, np.minimum(x, 1))
    submission = pd.DataFrame({"ID": test.ID, "TARGET": x})
    submission.to_csv("arigptestsubmission.csv", index=False)
    x = np.power(GPIndividual1(test) *
                 GPIndividual2(test) *
                 GPIndividual3(test), 1./3)
    x = np.maximum(0, np.minimum(x, 1))
    submission = pd.DataFrame({"ID": test.ID,
                               "TARGET": x})
    submission.to_csv("geogptestsubmission.csv", index=False)
    print('Completed!')
