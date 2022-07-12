import math
import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss


def Outputs(data):
    return 1.-(1./(1.+np.exp(-data)))


def Binarize(columnName, df, features=None):
    df[columnName] = df[columnName].astype(str)
    if(features is None):
        features = np.unique(df[columnName].values)
    print(features)
    for x in features:
        df[columnName+'_' + x] = df[columnName].map(lambda y:
                                                    1 if y == x else 0)
    df.drop(columnName, inplace=True, axis=1)
    return df, features


def MungeData(train, test):
    features = train.columns[2:]
    todrop = ['v22','v91']
    print(todrop)
    train.drop(todrop,
               axis=1, inplace=True)
    test.drop(todrop,
              axis=1, inplace=True)

    features = train.columns[2:]
    for col in features:
        if((train[col].dtype == 'object')):
            print(col)
            train, binfeatures = Binarize(col, train)
            test, _ = Binarize(col, test, binfeatures)
            nb = BernoulliNB()
            nb.fit(train[col+'_'+binfeatures].values, train.target.values)
            train[col] = \
                nb.predict_proba(train[col+'_'+binfeatures].values)[:, 1]
            test[col] = \
                nb.predict_proba(test[col+'_'+binfeatures].values)[:, 1]
            train.drop(col+'_'+binfeatures, inplace=True, axis=1)
            test.drop(col+'_'+binfeatures, inplace=True, axis=1)

    features = train.columns[2:]
    train[features] = train[features].astype(float)
    test[features] = test[features].astype(float)
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)
    return train, test


def GPIndividual1(data):
    predictions = (((-(np.minimum( (data["v50"]),  (((1.0/(1.0 + np.exp(- data["v66"]))) * 2.0))))) - ((1.0/(1.0 + np.exp(- np.ceil(np.maximum( (np.ceil((data["v56"] + np.ceil((data["v79"] * 2.0))))),  (data["v66"])))))) * 2.0)) +
                    ((-(((data["v38"] > (data["v74"] * data["v21"])).astype(float)))) + np.minimum( (((data["v50"] >= data["v66"]).astype(float))),  (((data["v3"] >= np.floor(((data["v50"] + ((data["v74"] != np.sin(data["v74"])).astype(float)))/2.0))).astype(float))))) +
                    (0.058823 * (((data["v79"] - data["v125"]) - (((data["v47"] * 2.0) - np.sin(np.sinh(np.sinh(data["v12"])))) * data["v24"])) - np.floor(np.sinh(np.round(data["v24"]))))) +
                    ((data["v50"] < (-(np.sinh(np.cos(((data["v50"] < (-((0.585714 + ((1.0/(1.0 + np.exp(- data["v66"]))) * 2.0))))).astype(float))))))).astype(float)) +
                    ((np.maximum( ((data["v66"] + np.minimum( (5.428570),  (data["v40"])))),  (data["v14"])) - np.cos((data["v66"] * 2.0))) * (0.434294 - np.sinh(np.cos(((data["v14"] != data["v57"]).astype(float)))))) +
                    (np.abs((0.301030 * data["v40"])) * (((data["v50"] >= np.sin(np.ceil(data["v12"]))).astype(float)) - ((data["v129"] > ((data["v86"] >= data["v40"]).astype(float))).astype(float)))) +
                    (-((np.minimum( (np.maximum( (data["v56"]),  (np.sin(np.maximum( (np.minimum( (((data["v30"] > 0.602941).astype(float))),  (np.cos(data["v113"])))),  (data["v110"])))))),  (((data["v113"] <= ((data["v40"] + np.sin(0.094340))/2.0)).astype(float)))) / 2.0))) +
                    (np.minimum( (0.094340),  ((0.094340 * (data["v85"] - (data["v4"] - np.abs((np.ceil(data["v50"]) * data["v113"]))))))) - ((data["v113"] > ((data["v113"] != data["v24"]).astype(float))).astype(float))) +
                    ((((0.094340 > (1.0/(1.0 + np.exp(- (data["v10"] - ((np.maximum( (data["v72"]),  (data["v88"])) < np.floor(((data["v50"] < 0.094340).astype(float)))).astype(float))))))).astype(float)) >= ((math.floor(5.428570) != np.ceil(data["v88"])).astype(float))).astype(float)) +
                    (data["v88"] * np.ceil(np.sin(((np.minimum( (((data["v38"] > data["v88"]).astype(float))),  (((data["v97"] > (data["v70"] + (2.212120 / 2.0))).astype(float)))) != np.abs(((2.718282 < data["v84"]).astype(float)))).astype(float))))) +
                    (np.minimum( ((data["v77"] * data["v130"])),  ((-(((((data["v56"] + np.sinh(data["v3"]))/2.0) >= np.maximum( (31.006277),  (data["v3"]))).astype(float)))))) + ((data["v130"] >= 5.428570).astype(float))) +
                    np.minimum( (np.cos(((data["v40"] + data["v30"])/2.0))),  (np.minimum( (np.cos(np.minimum( (data["v12"]),  (((data["v124"] + data["v40"])/2.0))))),  (((5.200000 < ((data["v124"] + np.minimum( (data["v45"]),  (data["v124"])))/2.0)).astype(float)))))) +
                    (((((data["v30"] * ((0.602941 < np.minimum( (data["v50"]),  (data["v50"]))).astype(float))) <= (-3.0 - np.minimum( (data["v34"]),  (data["v21"])))).astype(float)) / 2.0) - ((data["v62"] < (data["v35"] - 3.141593)).astype(float))) +
                    np.tanh(((5.428570 <= np.sinh((data["v21"] + (((((data["v73"] < data["v102"]).astype(float)) > np.floor((((((1.0/(1.0 + np.exp(- data["v21"]))) == 5.428570).astype(float)) == ((0.720430 <= data["v113"]).astype(float))).astype(float)))).astype(float)) / 2.0)))).astype(float))) +
                    (2.0 * ((np.maximum( (((data["v103"] > ((data["v114"] >= ((data["v79"] >= data["v83"]).astype(float))).astype(float))).astype(float))),  (data["v79"])) == np.cos((data["v85"] * (data["v83"] * np.maximum( (data["v37"]),  (data["v18"])))))).astype(float))) +
                    (np.floor(np.maximum( (np.sin((1.732051 * data["v13"]))),  (((2.0 <= (((np.tanh(data["v30"]) + data["v129"])/2.0) * np.tanh(np.round(np.ceil(data["v23"]))))).astype(float))))) * 2.0) +
                    (((np.round(np.minimum( (data["v3"]),  ((-(np.minimum( (data["v66"]),  ((((2.0 < ((data["v98"] + ((1.197370 + (data["v38"] * data["v93"]))/2.0))/2.0)).astype(float)) * 2.0)))))))) * 2.0) * 2.0) * 2.0) +
                    ((np.floor(np.floor(np.minimum( (np.cos(((data["v40"] + (np.sin((data["v129"] * 2.0)) + (data["v129"] + data["v66"])))/2.0))),  (np.cos(((data["v40"] + data["v50"])/2.0)))))) / 2.0) / 2.0) +
                    np.minimum( (0.138462),  ((((np.minimum( ((0.138462 * ((data["v93"] >= data["v114"]).astype(float)))),  (data["v24"])) * data["v83"]) > ((0.058823 >= ((data["v24"] > np.round(np.minimum( (data["v72"]),  (data["v24"])))).astype(float))).astype(float))).astype(float)))) +
                    ((np.floor(np.cos(np.maximum( (data["v56"]),  ((-(data["v52"])))))) + ((0.058823 < np.minimum( ((-(data["v66"]))),  (np.maximum( (data["v56"]),  (np.minimum( (data["v74"]),  (np.maximum( (0.602941),  (data["v66"]))))))))).astype(float)))/2.0) +
                    ((1.0/(1.0 + np.exp(- -3.0))) * np.minimum( ((data["v66"] + (data["v66"] * data["v114"]))),  (((2.212120 * data["v79"]) * (data["v34"] - data["v21"]))))) +
                    ((2.212120 <= (np.maximum( (data["v129"]),  (data["v96"])) * (data["v34"] * np.floor(np.maximum( (data["v90"]),  (data["v71"])))))).astype(float)) +
                    (np.floor(np.cos(((np.cos(np.maximum( (np.maximum( (data["v93"]),  (data["v120"]))),  (data["v121"]))) + ((((data["v13"] > 2.675680).astype(float)) * np.ceil((data["v85"] + data["v14"]))) * 2.0))/2.0))) * 2.0) +
                    np.sin(((np.sin(data["v88"]) > (2.409090 + np.minimum( (data["v79"]),  ((data["v79"] * np.sinh((data["v86"] * ((data["v64"] < ((data["v81"] < np.floor(np.sin(data["v88"]))).astype(float))).astype(float))))))))).astype(float))) +
                    ((((np.minimum( ((((data["v99"] > ((data["v52"] > data["v21"]).astype(float))).astype(float)) * data["v110"])),  (((data["v57"] > data["v21"]).astype(float)))) > ((data["v57"] >= ((np.sin(data["v3"]) >= data["v57"]).astype(float))).astype(float))).astype(float)) * 2.0) * 2.0) +
                    (((np.minimum( (data["v55"]),  (data["v58"])) >= (data["v130"] + np.cos(((data["v55"] < ((((((data["v74"] <= np.minimum( (data["v55"]),  (data["v30"]))).astype(float)) + data["v62"])/2.0) < data["v130"]).astype(float))).astype(float))))).astype(float)) * 2.0) +
                    (((np.minimum( (data["v23"]),  ((data["v102"] - data["v8"]))) >= ((data["v102"] != data["v23"]).astype(float))).astype(float)) + ((np.minimum( (data["v59"]),  (((np.minimum( (data["v23"]),  (data["v2"])) >= 0.693147).astype(float)))) >= (1.0/(1.0 + np.exp(- data["v102"])))).astype(float))) +
                    (data["v29"] * ((np.round(np.tanh(((data["v53"] * ((data["v115"] > (((data["v85"] * data["v54"]) + ((np.ceil(data["v13"]) > data["v130"]).astype(float)))/2.0)).astype(float))) / 2.0))) == np.ceil(data["v13"])).astype(float))) +
                    np.minimum( ((-(((np.maximum( (data["v35"]),  (data["v126"])) > ((1.732051 - ((data["v66"] > np.abs(np.floor(data["v126"]))).astype(float))) * 2.0)).astype(float))))),  (np.floor(np.sin(np.abs(data["v27"]))))) +
                    np.sinh(np.sinh(((data["v49"] > (2.675680 - np.sinh(((np.maximum( (data["v30"]),  (((data["v72"] > np.sin(2.675680)).astype(float)))) < ((data["v6"] < (data["v126"] + data["v75"])).astype(float))).astype(float))))).astype(float)))) +
                    np.round(np.round(np.round(np.minimum( (((data["v25"] > data["v14"]).astype(float))),  (((((data["v68"] < data["v105"]).astype(float)) < ((2.675680 < (np.minimum( (data["v7"]),  (data["v99"])) * 2.0)).astype(float))).astype(float))))))) +
                    (-(np.minimum( (np.minimum( (0.058823),  (((data["v10"] + np.abs(np.floor(np.minimum( (data["v38"]),  (0.058823)))))/2.0)))),  (((data["v50"] + np.sin(np.floor(((data["v38"] > (data["v50"] / 2.0)).astype(float)))))/2.0))))) +
                    (data["v33"] * (9.869604 * ((((9.869604 < (data["v124"] - np.maximum( (data["v127"]),  (data["v33"])))).astype(float)) >= ((((data["v64"] != 9.869604).astype(float)) != np.sin(np.maximum( (data["v93"]),  (data["v33"])))).astype(float))).astype(float)))) +
                    (np.minimum( (2.302585),  (((data["v52"] > (2.302585 - np.minimum( (data["v74"]),  ((np.tanh(np.ceil((2.302585 * data["v50"]))) - data["v121"]))))).astype(float)))) * 2.0) +
                    (np.cos((1.584910 - ((np.minimum( (data["v50"]),  (data["v24"])) > np.cos(np.minimum( (np.minimum( (np.round(np.minimum( (data["v110"]),  (data["v24"])))),  (data["v79"]))),  (np.round(((data["v56"] + data["v67"])/2.0)))))).astype(float)))) / 2.0) +
                    (-(((2.718282 <= np.maximum( (np.maximum( (data["v2"]),  (np.maximum( (((data["v72"] + data["v92"])/2.0)),  ((data["v110"] + (((data["v87"] - data["v77"]) + data["v131"])/2.0))))))),  ((data["v87"] - data["v92"])))).astype(float)))) +
                    ((31.006277 < np.sinh(((data["v129"] + ((np.minimum( (data["v107"]),  (((np.sinh(((data["v129"] + data["v40"])/2.0)) + (data["v119"] - data["v95"]))/2.0))) + np.sinh(((data["v72"] + data["v119"])/2.0)))/2.0))/2.0))).astype(float)) +
                    np.minimum( (np.cos((data["v35"] * (data["v35"] * ((data["v90"] <= (1.0/(1.0 + np.exp(- data["v64"])))).astype(float)))))),  ((np.floor(np.tanh(np.cos((data["v50"] * ((np.cos(data["v64"]) <= data["v47"]).astype(float)))))) / 2.0))) +
                    ((((data["v11"] >= (((data["v99"] + np.maximum( (np.abs(data["v114"])),  (np.ceil(np.maximum( (data["v111"]),  (np.maximum( (np.maximum( (data["v85"]),  (data["v86"]))),  (np.abs(data["v92"])))))))))/2.0) * 2.0)).astype(float)) * 2.0) * 2.0) +
                    np.floor(np.cos((data["v88"] * np.sin((np.minimum( (np.minimum( ((data["v68"] + np.sinh(data["v12"]))),  ((np.minimum( (data["v75"]),  (data["v125"])) * data["v35"])))),  (np.minimum( (data["v75"]),  (data["v125"])))) / 2.0))))) +
                    np.minimum( ((np.minimum( (0.094340),  (((data["v79"] >= np.abs(data["v66"])).astype(float)))) * 2.0)),  (((data["v34"] >= np.maximum( (((0.094340 > data["v34"]).astype(float))),  (data["v129"]))).astype(float)))) +
                    np.cos(np.maximum( (1.584910),  ((np.maximum( (np.maximum( (np.maximum( ((data["v103"] - data["v58"])),  (data["v122"]))),  ((data["v103"] - data["v81"])))),  ((data["v81"] - data["v58"]))) - ((data["v127"] >= data["v57"]).astype(float)))))) +
                    ((1.584910 < np.maximum( (data["v29"]),  (((data["v35"] - np.cos(((data["v89"] / 2.0) * ((data["v71"] > ((data["v2"] <= np.ceil(data["v92"])).astype(float))).astype(float))))) * 2.0)))).astype(float)) +
                    ((data["v81"] >= (((1.0/(1.0 + np.exp(- (((((data["v58"] > data["v14"]).astype(float)) * data["v39"]) <= np.maximum( (data["v14"]),  (np.sin(np.sinh((data["v58"] - data["v54"])))))).astype(float))))) * 2.0) * 2.0)).astype(float)) +
                    np.floor(np.floor((np.maximum( (data["v76"]),  (data["v107"])) * np.floor(np.cos(np.minimum( ((data["v40"] * data["v3"])),  (np.minimum( (data["v54"]),  (((data["v39"] + ((data["v107"] >= (1.0/(1.0 + np.exp(- data["v39"])))).astype(float)))/2.0)))))))))) +
                    ((np.floor((((((data["v87"] > data["v129"]).astype(float)) == np.cos(((np.floor(np.abs(data["v87"])) + np.minimum( (np.abs(((data["v87"] / 2.0) / 2.0))),  (data["v34"])))/2.0))).astype(float)) * 2.0)) * 2.0) * 2.0) +
                    (data["v79"] * ((2.212120 <= (((data["v24"] <= (((data["v109"] <= ((data["v85"] <= (1.0/(1.0 + np.exp(- data["v129"])))).astype(float))).astype(float)) * data["v34"])).astype(float)) * np.maximum( (data["v115"]),  (data["v34"])))).astype(float))) +
                    (data["v115"] * (((data["v41"] * data["v90"]) <= np.sin(((data["v117"] > (3.141593 + np.ceil((((np.sin(data["v2"]) >= data["v41"]).astype(float)) - np.maximum( (data["v122"]),  (data["v30"])))))).astype(float)))).astype(float))) +
                    np.floor(np.cos(((data["v5"] + (data["v66"] * np.minimum( (np.maximum( (0.602941),  (np.tanh(np.sin(data["v66"]))))),  (np.sinh(((np.maximum( (np.floor(data["v5"])),  (data["v95"])) + data["v50"])/2.0))))))/2.0))) +
                    (((data["v40"] >= ((5.200000 + np.tanh((((np.maximum( (data["v8"]),  (data["v66"])) > (((((data["v107"] < data["v12"]).astype(float)) > data["v7"]).astype(float)) * data["v52"])).astype(float)) * data["v66"])))/2.0)).astype(float)) * 2.0) +
                    np.sin(((((0.367879 <= np.maximum( (data["v40"]),  (((data["v43"] > 0.693147).astype(float))))).astype(float)) * data["v74"]) * ((data["v52"] + (data["v9"] * data["v109"]))/2.0))) +
                    np.minimum( (0.138462),  (((data["v75"] < np.sinh(np.minimum( ((-(data["v110"]))),  (np.minimum( ((data["v50"] * data["v26"])),  (np.minimum( (np.abs(data["v50"])),  (np.maximum( (data["v66"]),  (data["v75"])))))))))).astype(float)))) +
                    (0.094340 * ((-(((data["v50"] <= (data["v21"] * np.minimum( ((31.006277 + data["v81"])),  ((((0.094340 >= 0.094340)) * data["v110"]))))).astype(float)))) * ((data["v21"] > data["v12"]).astype(float)))) +
                    np.minimum( (0.058823),  ((((np.cos(data["v80"]) * np.maximum( (data["v119"]),  (np.round(((data["v34"] >= ((1.0/(1.0 + np.exp(- ((data["v14"] <= data["v34"]).astype(float))))) / 2.0)).astype(float)))))) > ((data["v102"] >= np.abs(data["v4"])).astype(float))).astype(float)))) +
                    ((data["v34"] <= (-2.0 - ((((data["v106"] <= ((((data["v34"] <= data["v1"]).astype(float)) != 31.006277).astype(float))).astype(float)) != np.cos(((data["v8"] * data["v124"]) * np.ceil(np.cos(data["v8"]))))).astype(float)))).astype(float)) +
                    ((1.630430 < np.minimum( (data["v5"]),  ((data["v108"] * np.maximum( ((data["v43"] - ((data["v111"] >= np.sinh(data["v71"])).astype(float)))),  ((data["v94"] - ((data["v122"] == data["v5"]).astype(float))))))))).astype(float)) +
                    ((data["v73"] <= np.minimum( (data["v122"]),  ((data["v71"] - np.cos(((data["v36"] < (((data["v52"] * ((data["v80"] > np.maximum( (np.cos(data["v16"])),  (data["v71"]))).astype(float))) + data["v2"])/2.0)).astype(float))))))).astype(float)) +
                    ((np.maximum( (data["v31"]),  (np.maximum( (data["v96"]),  (np.tanh(np.ceil(data["v130"])))))) < ((np.ceil(data["v130"]) < np.minimum( (((data["v17"] >= data["v36"]).astype(float))),  (((data["v36"] >= np.cos(data["v72"])).astype(float))))).astype(float))).astype(float)) +
                    np.sinh(((((2.0 != (3.0 - np.sin(data["v40"]))).astype(float)) < (-(((data["v113"] + ((data["v40"] <= ((data["v97"] + (3.0 - np.sin(data["v63"])))/2.0)).astype(float)))/2.0)))).astype(float))) +
                    np.cos(np.maximum( (np.maximum( (1.584910),  (np.minimum( ((data["v129"] - ((data["v25"] <= data["v81"]).astype(float)))),  (data["v14"]))))),  (np.minimum( ((data["v72"] - (data["v75"] * ((2.0 + 1.414214)/2.0)))),  (data["v125"]))))) +
                    (np.sinh(((((np.floor(data["v118"]) > (data["v107"] + 2.675680)).astype(float)) == ((np.cos(data["v106"]) < ((((data["v106"] < 2.675680).astype(float)) != np.cos(data["v58"])).astype(float))).astype(float))).astype(float))) * 2.0) +
                    (((data["v57"] + ((data["v117"] != data["v112"]).astype(float))) < np.minimum( (data["v16"]),  (((np.abs(data["v72"]) + (np.sin(data["v107"]) / 2.0)) + (data["v26"] - data["v29"]))))).astype(float)) +
                    (((data["v42"] >= ((data["v62"] <= (data["v62"] + data["v70"])).astype(float))).astype(float)) * (5.428570 * (data["v58"] * (data["v62"] * ((5.428570 <= (data["v62"] + data["v44"])).astype(float)))))) +
                    (np.minimum( (np.maximum( (np.cos(data["v60"])),  (data["v81"]))),  (np.floor(np.floor(np.cos((((data["v34"] + data["v104"])/2.0) * (1.0/(1.0 + np.exp(- np.floor(((data["v4"] > ((data["v1"] > data["v81"]).astype(float))).astype(float)))))))))))) * 2.0))

    return Outputs(predictions)


def GPIndividual2(data):
    predictions = ((-(((((np.maximum( (data["v66"]),  (np.maximum( (data["v79"]),  (data["v110"])))) + ((data["v50"] + np.cos(data["v50"]))/2.0))/2.0) + (1.0/(1.0 + np.exp(- data["v31"])))) + ((data["v50"] + np.cos(data["v50"]))/2.0)))) +
                    ((-((((-(((data["v56"] > np.maximum( (np.cos(np.abs(data["v47"]))),  ((data["v47"] - (data["v40"] + (data["v24"] * 2.0)))))).astype(float)))) < data["v74"]).astype(float)))) / 2.0) +
                    np.minimum( (np.minimum( ((np.abs(data["v3"]) / 2.0)),  ((0.730769 - np.sin(np.round(data["v66"])))))),  (np.sinh((0.730769 + ((data["v12"] + 2.212120)/2.0))))) +
                    ((((((data["v50"] <= (-(np.cos(data["v74"])))).astype(float)) + np.tanh((data["v38"] * (data["v125"] - data["v38"]))))/2.0) + np.tanh((data["v74"] * np.ceil((data["v21"] - data["v119"])))))/2.0) +
                    np.abs(np.minimum( (((data["v38"] == data["v20"]).astype(float))),  ((data["v66"] * np.abs((data["v66"] * (data["v66"] * np.abs(np.minimum( (((data["v50"] + np.cos(data["v50"]))/2.0)),  (np.floor(0.367879))))))))))) +
                    np.minimum( (((np.abs(data["v50"]) == data["v50"]).astype(float))),  (((((1.0/(1.0 + np.exp(- data["v56"]))) + data["v66"])/2.0) * ((data["v114"] + (-(((data["v50"] + ((data["v50"] + data["v66"])/2.0))/2.0))))/2.0)))) +
                    (0.058823 * np.round((data["v31"] - (data["v24"] * (5.200000 * ((((data["v31"] > np.maximum( (data["v30"]),  (data["v24"]))).astype(float)) > np.maximum( (data["v64"]),  (np.round(data["v24"])))).astype(float))))))) +
                    ((np.cos(data["v74"]) <= ((data["v50"] >= np.maximum( (2.409090),  (np.maximum( (np.maximum( ((data["v119"] + 3.141593)),  ((data["v125"] + 2.409090)))),  ((2.409090 + ((data["v52"] > data["v80"]).astype(float)))))))).astype(float))).astype(float)) +
                    np.cos((1.584910 - ((data["v50"] + 20.750000) * ((0.367879 >= np.cos(np.minimum( (data["v10"]),  (np.minimum( (data["v10"]),  (np.minimum( (data["v50"]),  (data["v76"])))))))).astype(float))))) +
                    (-(((data["v113"] >= ((((data["v56"] != data["v113"]).astype(float)) != ((5.200000 <= np.maximum( ((data["v38"] * data["v126"])),  ((((data["v56"] >= data["v50"]).astype(float)) + data["v3"])))).astype(float))).astype(float))).astype(float)))) +
                    np.minimum( ((data["v106"] * data["v60"])),  (np.minimum( (((data["v114"] < ((0.058823 > data["v40"]).astype(float))).astype(float))),  (np.minimum( ((data["v106"] * data["v59"])),  ((np.sin(np.tanh((data["v3"] * data["v40"]))) / 2.0))))))) +
                    (((data["v40"] + (((-2.0 - data["v47"]) - data["v113"]) - np.sin(np.abs((np.floor(data["v24"]) * 2.0)))))/2.0) * (1.0/(1.0 + np.exp(- ((-2.0 - data["v113"]) * 2.0))))) +
                    np.sinh(((data["v50"] >= (8.0 - (np.maximum( (np.maximum( (data["v88"]),  (np.round(((data["v130"] + ((-(data["v114"])) * 2.0))/2.0))))),  (np.maximum( (data["v31"]),  ((data["v81"] * data["v16"]))))) * 2.0))).astype(float))) +
                    ((((data["v47"] + (data["v24"] * data["v56"])) * ((data["v66"] >= np.maximum( ((((data["v129"] * 2.0) / 2.0) * ((data["v56"] >= (1.0/(1.0 + np.exp(- data["v24"])))).astype(float)))),  (data["v24"]))).astype(float))) / 2.0) / 2.0) +
                    np.tanh((0.094340 * ((((np.abs(data["v66"]) <= ((data["v66"] + np.floor(np.sinh(np.cos(data["v50"]))))/2.0)).astype(float)) * 2.0) - data["v21"]))) +
                    (0.058823 * (((np.minimum( (data["v21"]),  (np.floor((data["v2"] * data["v110"])))) + ((0.367879 < data["v34"]).astype(float))) + ((data["v34"] > 0.367879).astype(float))) - ((0.367879 <= data["v30"]).astype(float)))) +
                    ((np.minimum( (data["v23"]),  (data["v2"])) >= np.sinh(np.cos(((data["v127"] <= (1.732051 * ((data["v50"] > (np.sin(data["v30"]) * ((data["v127"] > data["v2"]).astype(float)))).astype(float)))).astype(float))))).astype(float)) +
                    (-(((np.maximum( (data["v94"]),  (data["v35"])) >= ((5.200000 + np.floor(np.minimum( ((-(((data["v50"] + np.cos(5.200000))/2.0)))),  ((data["v61"] + np.floor(data["v99"]))))))/2.0)).astype(float)))) +
                    (-(np.minimum( (np.maximum( (((data["v124"] >= ((data["v102"] <= data["v124"]).astype(float))).astype(float))),  ((data["v102"] / 2.0)))),  (((data["v42"] <= ((data["v130"] <= np.sinh(data["v38"])).astype(float))).astype(float)))))) +
                    ((data["v35"] >= (((((data["v57"] >= ((5.428570 < np.maximum( (data["v130"]),  (data["v72"]))).astype(float))).astype(float)) != ((np.minimum( (data["v102"]),  (data["v23"])) >= (((data["v23"] != data["v35"]).astype(float)) * 2.0)).astype(float))).astype(float)) * 2.0)).astype(float)) +
                    (((data["v105"] >= (8.0 - (data["v105"] - ((data["v59"] >= ((data["v71"] * 2.0) * 2.0)).astype(float))))).astype(float)) * (((data["v71"] * 2.0) * data["v2"]) * data["v2"])) +
                    np.minimum( (np.minimum( ((-(((5.428570 <= (data["v72"] + data["v17"])).astype(float))))),  (np.cos((data["v106"] * data["v49"]))))),  ((0.301030 - np.minimum( (data["v71"]),  ((np.cos(data["v30"]) * data["v107"])))))) +
                    np.abs(((0.058823 * (data["v10"] * ((data["v102"] <= ((data["v111"] < data["v59"]).astype(float))).astype(float)))) * (data["v10"] * ((((data["v75"] >= data["v39"]).astype(float)) <= data["v71"]).astype(float))))) +
                    (data["v118"] * ((8.0 <= ((data["v38"] + (data["v40"] * (np.abs(data["v66"]) * np.maximum( (data["v124"]),  ((np.abs(data["v107"]) * np.maximum( (data["v124"]),  (data["v38"]))))))))/2.0)).astype(float))) +
                    (0.058823 * ((data["v52"] + data["v125"]) * (-(((((data["v21"] >= (-(data["v30"]))).astype(float)) > ((data["v52"] == data["v125"]).astype(float))).astype(float)))))) +
                    (np.floor(np.sin(((1.570796 + np.maximum( (np.maximum( (np.maximum( (np.maximum( (data["v93"]),  (((data["v86"] + np.round(data["v58"]))/2.0)))),  ((data["v30"] - (data["v58"] / 2.0))))),  (data["v120"]))),  (data["v90"])))/2.0))) * 2.0) +
                    np.minimum( (((0.367879 < np.round(np.cos(data["v89"]))).astype(float))),  (((data["v50"] <= ((-3.0 + np.tanh((np.tanh(((data["v14"] <= (0.840000 - data["v66"])).astype(float))) - data["v66"])))/2.0)).astype(float)))) +
                    np.tanh(np.round(np.sin((1.0/(1.0 + np.exp(- (data["v53"] - np.maximum( (data["v32"]),  (np.maximum( (data["v32"]),  (np.maximum( (data["v4"]),  (np.sin(np.maximum( (1.630430),  ((data["v83"] * data["v87"]))))))))))))))))) +
                    (np.round(np.floor(np.maximum( (np.cos(data["v32"])),  (np.sin(np.maximum( (data["v12"]),  (((((((data["v39"] + data["v21"])/2.0) + data["v83"])/2.0) + data["v46"])/2.0)))))))) * 2.0) +
                    (np.minimum( (np.floor(np.sin((((2.409090 + data["v34"]) + ((data["v28"] >= data["v73"]).astype(float)))/2.0)))),  ((data["v28"] * np.sin(data["v37"])))) * ((data["v113"] > (-(data["v131"]))).astype(float))) +
                    ((5.200000 < np.maximum( (data["v120"]),  (((data["v72"] + (data["v23"] + ((data["v103"] + ((((data["v77"] < data["v23"]).astype(float)) == ((data["v79"] >= data["v130"]).astype(float))).astype(float))) * 2.0)))/2.0)))).astype(float)) +
                    ((((0.301030 * data["v50"]) * ((data["v74"] + np.round(np.maximum( ((data["v74"] * (data["v66"] + data["v107"]))),  (np.abs(np.abs(data["v30"]))))))/2.0)) / 2.0) / 2.0) +
                    ((data["v44"] * np.ceil(((np.maximum( ((data["v97"] * data["v98"])),  (data["v51"])) >= np.maximum( (2.302585),  (data["v111"]))).astype(float)))) * np.maximum( (data["v33"]),  ((np.tanh(data["v7"]) * data["v98"])))) +
                    np.round(np.tanh(np.minimum( ((data["v122"] * np.round(((data["v19"] + data["v108"])/2.0)))),  ((data["v53"] * np.minimum( (0.301030),  ((data["v122"] * np.round(((data["v100"] + np.cos(data["v129"]))/2.0)))))))))) +
                    np.ceil(np.minimum( (np.minimum( (data["v85"]),  (((np.sinh(data["v85"]) < ((data["v15"] <= ((data["v15"] == data["v10"]).astype(float))).astype(float))).astype(float))))),  (((np.cos((data["v15"] * np.round(data["v109"]))) + np.sinh(data["v10"]))/2.0)))) +
                    (data["v56"] * (np.maximum( ((data["v26"] + data["v7"])),  (((0.0 == ((data["v52"] < data["v107"]).astype(float))).astype(float)))) * (-((data["v111"] * ((data["v74"] > np.abs(data["v75"])).astype(float))))))) +
                    (((np.cos((data["v66"] * data["v3"])) <= ((((0.636620 > data["v31"]).astype(float)) + ((data["v49"] + (np.floor(((np.sin(data["v103"]) + ((data["v3"] != data["v103"]).astype(float)))/2.0)) * 2.0))/2.0))/2.0)).astype(float)) * 2.0) +
                    ((((data["v55"] > (np.maximum( (data["v121"]),  (np.tanh(np.cos(np.sin(data["v88"]))))) * 2.0)).astype(float)) >= (((1.0/(1.0 + np.exp(- data["v88"]))) <= (data["v79"] + 2.302585)).astype(float))).astype(float)) +
                    (-((((10.4371) < (((data["v3"] + (data["v38"] * np.maximum( (data["v112"]),  ((data["v50"] + data["v50"]))))) + data["v39"]) + 2.302585)).astype(float)))) +
                    (((1.0/(1.0 + np.exp(- ((data["v97"] > (data["v84"] - (np.minimum( (data["v129"]),  (np.cos(data["v56"]))) + data["v49"]))).astype(float))))) <= (data["v49"] - (data["v106"] + np.sinh(np.sinh(np.abs(data["v45"])))))).astype(float)) +
                    np.sinh((np.minimum( (((data["v50"] + np.cos(0.138462))/2.0)),  (((((1.0/(1.0 + np.exp(- data["v50"]))) + (data["v40"] + 5.200000)) <= np.maximum( (data["v84"]),  (data["v50"]))).astype(float)))) * 2.0)) +
                    (((((np.sin((31.006277 + data["v101"])) == ((data["v101"] != 31.006277).astype(float))).astype(float)) == ((np.sin(data["v33"]) < ((data["v87"] < (2.28274)).astype(float))).astype(float))).astype(float)) * 31.006277) +
                    (((data["v99"] <= ((data["v96"] + np.minimum( (data["v62"]),  (np.minimum( (data["v125"]),  (np.minimum( (np.minimum( ((-(data["v20"]))),  (np.minimum( (data["v114"]),  (np.minimum( (data["v4"]),  (data["v21"]))))))),  (np.cos(2.718282))))))))/2.0)).astype(float)) * 2.0) +
                    ((np.ceil(np.sinh(data["v14"])) * (-(((2.675680 <= np.maximum( (np.maximum( (np.maximum( (((data["v80"] + data["v97"])/2.0)),  (np.maximum( (data["v98"]),  (data["v97"]))))),  (np.round(data["v7"])))),  (data["v92"]))).astype(float))))) * 2.0) +
                    ((-(((-3.0 >= (data["v34"] * (data["v131"] + np.maximum( ((data["v71"] * np.sin(data["v131"]))),  (((data["v129"] >= (np.sin(data["v71"]) * data["v34"])).astype(float))))))).astype(float)))) / 2.0) +
                    (((data["v131"] < 0.840000).astype(float)) * np.tanh(np.minimum( (((data["v56"] < data["v131"]).astype(float))),  (((data["v6"] < np.floor(np.floor((data["v48"] - ((data["v29"] <= ((1.197370 + 0.840000)/2.0)).astype(float)))))).astype(float)))))) +
                    np.minimum( ((data["v122"] * data["v35"])),  (((((data["v47"] < (data["v79"] - (np.tanh((1.0/(1.0 + np.exp(- ((data["v18"] >= ((data["v40"] >= ((data["v130"] == data["v47"]).astype(float))).astype(float))).astype(float)))))) * 2.0))).astype(float)) / 2.0) / 2.0))) +
                    ((2.212120 < np.maximum( ((data["v29"] + data["v75"])),  ((((data["v1"] - (np.round(data["v106"]) * 2.0)) - data["v106"]) / 2.0)))).astype(float)) +
                    np.round(((np.cos(data["v58"]) + np.minimum( (data["v84"]),  ((data["v7"] * (data["v62"] * np.ceil((-(np.cos(((data["v106"] + np.round(((data["v106"] + np.round(data["v7"]))/2.0)))/2.0))))))))))/2.0)) +
                    np.floor(np.sin(np.maximum( ((data["v21"] + data["v36"])),  ((((data["v123"] + (-(data["v30"])))/2.0) * ((data["v30"] <= ((data["v23"] + (-(9.869604)))/2.0)).astype(float))))))) +
                    (-(((np.ceil(data["v44"]) == np.minimum( (data["v50"]),  (((((np.round(data["v48"]) <= ((data["v129"] >= np.tanh((data["v127"] * data["v50"]))).astype(float))).astype(float)) == np.ceil((-(data["v23"])))).astype(float))))).astype(float)))) +
                    (np.floor(np.sin(((3.141593 + ((np.maximum( (data["v131"]),  (data["v112"])) + (((data["v112"] >= data["v80"]).astype(float)) + (-(data["v80"]))))/2.0))/2.0))) * 2.0) +
                    ((1.732051 <= np.minimum( (data["v39"]),  (np.floor(np.maximum( (np.maximum( (data["v81"]),  (((data["v14"] + (-(np.round((data["v38"] * data["v62"])))))/2.0)))),  (np.maximum( (data["v83"]),  (np.maximum( (data["v81"]),  (data["v5"])))))))))).astype(float)) +
                    (np.minimum( (((data["v131"] <= ((data["v73"] >= (1.0/(1.0 + np.exp(- data["v103"])))).astype(float))).astype(float))),  (np.minimum( (((2.675680 <= data["v21"]).astype(float))),  (((((data["v131"] >= data["v109"]).astype(float)) >= data["v47"]).astype(float)))))) * 2.0) +
                    np.floor(((0.720430 < (data["v117"] - np.maximum( (data["v109"]),  (np.maximum( ((0.094340 - data["v71"])),  (np.sinh((data["v41"] - np.tanh(((np.ceil(data["v49"]) < data["v40"]).astype(float))))))))))).astype(float))) +
                    (data["v104"] * (data["v131"] * (data["v109"] * (5.200000 * ((np.sin(((1.0/(1.0 + np.exp(- np.maximum( (np.maximum( (data["v27"]),  (data["v85"]))),  (data["v4"]))))) * 2.0)) >= ((data["v4"] != data["v130"]).astype(float))).astype(float)))))) +
                    np.sinh(np.sinh(np.floor(np.floor(np.cos(((((data["v68"] * (np.cos(data["v125"]) * data["v87"])) + (((data["v122"] + data["v28"]) >= ((data["v68"] != data["v46"]).astype(float))).astype(float)))/2.0) / 2.0)))))) +
                    np.floor(np.cos((data["v117"] * (0.434294 - (np.maximum( (data["v70"]),  (np.maximum( ((data["v56"] + np.maximum( (data["v21"]),  (data["v56"])))),  (np.maximum( (((data["v23"] + data["v56"])/2.0)),  (data["v14"])))))) / 2.0))))) +
                    (-(((1.570796 <= np.minimum( (data["v35"]),  (np.maximum( (data["v21"]),  (np.maximum( (data["v69"]),  (((data["v113"] + (data["v25"] + np.sinh(np.abs(data["v40"])))) / 2.0)))))))).astype(float)))) +
                    ((data["v44"] >= ((3.141593 + ((((data["v102"] >= ((np.sin(np.minimum( (data["v125"]),  (((data["v36"] > data["v88"]).astype(float))))) > data["v75"]).astype(float))).astype(float)) >= ((data["v88"] <= data["v99"]).astype(float))).astype(float)))/2.0)).astype(float)) +
                    (-((((2.718282 <= np.abs((data["v97"] * np.maximum( (data["v114"]),  ((((data["v70"] > data["v96"]).astype(float)) - (data["v39"] * 2.0))))))).astype(float)) * 2.0))) +
                    np.floor(np.cos(np.minimum( (data["v103"]),  ((data["v100"] + ((data["v14"] > np.floor(np.cos(np.minimum( (data["v103"]),  ((data["v100"] + ((data["v25"] > np.round((data["v14"] + data["v128"]))).astype(float)))))))).astype(float))))))) +
                    ((((2.675680 < np.maximum( (data["v21"]),  (np.maximum( (data["v38"]),  ((data["v35"] + np.cos(data["v72"]))))))).astype(float)) == np.cos(((data["v72"] >= np.tanh(data["v40"])).astype(float)))).astype(float)) +
                    (((np.cos((data["v95"] - np.cos(((data["v11"] + data["v55"])/2.0)))) >= ((np.cos((data["v95"] - np.cos(((data["v95"] > data["v18"]).astype(float))))) != (-(-1.0))).astype(float))).astype(float)) * 2.0))

    return Outputs(predictions)


def GPIndividual3(data):
    predictions = ((-((((data["v50"] + (np.maximum( (data["v79"]),  (data["v66"])) + np.cos(((data["v50"] + np.floor(data["v50"]))/2.0))))/2.0) + np.maximum( (data["v56"]),  ((1.0/(1.0 + np.exp(- np.maximum( (data["v47"]),  (data["v66"])))))))))) +
                    ((np.sinh(np.sinh(np.minimum( (np.sin(data["v50"])),  ((data["v50"] + np.sin(data["v66"])))))) * (np.cos((data["v50"] * 2.0)) / 2.0)) / 2.0) +
                    np.minimum( ((2.212120 + data["v12"])),  ((0.094340 * ((((-(data["v24"])) + data["v34"]) * data["v47"]) + ((-(data["v14"])) + (data["v34"] + (-(data["v24"])))))))) +
                    (((((((0.636620 == data["v10"]).astype(float)) > ((data["v50"] + 1.197370)/2.0)).astype(float)) > ((data["v10"] + 1.197370)/2.0)).astype(float)) + np.tanh((np.minimum( (data["v3"]),  (((data["v50"] > data["v78"]).astype(float)))) / 2.0))) +
                    (0.058823 * (-((data["v125"] - (np.sinh((((0.138462 > data["v56"]).astype(float)) - ((np.minimum( (np.round(data["v30"])),  (1.0)) > np.abs(data["v123"])).astype(float)))) * 2.0))))) +
                    np.minimum( (np.sin((data["v74"] * data["v38"]))),  (np.sin(((data["v47"] + np.cos(np.sinh((((1.0/(1.0 + np.exp(- (data["v74"] * data["v38"])))) < (np.sin((data["v56"] - data["v66"])) * 2.0)).astype(float)))))/2.0)))) +
                    ((((np.sin((data["v56"] - ((data["v114"] == np.sin(data["v114"])).astype(float)))) - data["v114"]) * np.tanh(np.sin(((data["v74"] >= data["v66"]).astype(float))))) + ((1.197370 < data["v56"]).astype(float)))/2.0) +
                    (0.138462 * ((np.cos((data["v129"] - np.minimum( (data["v114"]),  (data["v10"])))) + np.minimum( (data["v114"]),  ((((data["v40"] * data["v3"]) - data["v3"]) - data["v21"]))))/2.0)) +
                    (0.058823 * (np.floor(data["v119"]) * np.maximum( (data["v87"]),  ((np.maximum( (data["v56"]),  ((data["v110"] * ((data["v72"] + 2.212120)/2.0)))) * ((data["v41"] <= np.maximum( (data["v87"]),  (data["v87"]))).astype(float))))))) +
                    np.sinh(np.sinh(((data["v88"] >= ((((data["v47"] <= ((((np.sin(data["v93"]) != ((data["v93"] != data["v89"]).astype(float))).astype(float)) > np.tanh(data["v54"])).astype(float))).astype(float)) * 2.0) * 2.0)).astype(float)))) +
                    (-(((data["v113"] > np.maximum( (np.abs(np.ceil(np.ceil(((np.ceil(data["v117"]) < np.abs(np.ceil(data["v31"]))).astype(float)))))),  (((data["v38"] <= np.sin(np.maximum( (data["v88"]),  (np.ceil(data["v38"]))))).astype(float))))).astype(float)))) +
                    (2.302585 * ((3.0 <= np.minimum( ((data["v38"] - np.minimum( (data["v131"]),  (3.0)))),  (np.maximum( ((data["v88"] - data["v52"])),  (np.maximum( (data["v46"]),  (data["v88"]))))))).astype(float))) +
                    ((data["v79"] * (0.058823 * np.cos(np.minimum( ((data["v79"] * (data["v15"] * np.maximum( (data["v50"]),  (data["v66"]))))),  (np.minimum( (np.sinh(data["v113"])),  (np.minimum( (data["v129"]),  (data["v79"]))))))))) * 2.0) +
                    np.minimum( (np.ceil((data["v130"] - data["v19"]))),  (np.minimum( (((data["v49"] < np.cos(data["v21"])).astype(float))),  (np.minimum( (((data["v8"] < (-(np.maximum( (data["v50"]),  (data["v75"]))))).astype(float))),  (((data["v21"] <= -3.0).astype(float)))))))) +
                    ((-(np.tanh(np.ceil(np.minimum( (((data["v31"] > ((np.sin(data["v31"]) + data["v113"])/2.0)).astype(float))),  (np.minimum( (((np.cos(data["v113"]) + data["v66"])/2.0)),  ((data["v66"] * (1.0/(1.0 + np.exp(- data["v79"])))))))))))) / 2.0) +
                    (((data["v108"] - (np.round(data["v24"]) * np.minimum( (data["v110"]),  (((data["v50"] >= 0.318310).astype(float)))))) * 0.094340) * np.minimum( (data["v24"]),  (((data["v66"] >= (data["v108"] - data["v60"])).astype(float))))) +
                    (np.ceil((data["v35"] * (data["v42"] * ((((5.200000 <= (np.maximum( (data["v50"]),  (data["v130"])) - np.cos(((data["v50"] >= (data["v19"] / 2.0)).astype(float))))).astype(float)) >= np.cos(data["v42"])).astype(float))))) * 2.0) +
                    np.floor(np.cos((data["v42"] * ((np.maximum( (data["v34"]),  (data["v31"])) + ((data["v97"] >= (((data["v114"] < np.cos(data["v58"])).astype(float)) + data["v114"])).astype(float)))/2.0)))) +
                    np.sinh(np.round((data["v66"] * np.floor(np.tanh((data["v129"] + ((data["v3"] <= np.cos((data["v49"] * np.ceil(np.ceil((data["v75"] * 0.138462)))))).astype(float)))))))) +
                    np.abs((data["v114"] * ((2.718282 <= ((np.maximum( (data["v114"]),  (np.sinh((np.abs((data["v114"] + ((data["v125"] > ((data["v79"] > (1.0/(1.0 + np.exp(- data["v131"])))).astype(float))).astype(float)))) / 2.0)))) + data["v28"])/2.0)).astype(float)))) +
                    np.round((data["v52"] * ((-1.0 > (data["v16"] + ((data["v74"] < (np.cos(data["v108"]) - ((data["v127"] >= (1.0/(1.0 + np.exp(- data["v52"])))).astype(float)))).astype(float)))).astype(float)))) +
                    ((data["v70"] + data["v27"]) * ((data["v53"] > ((2.212120 + np.maximum( ((((data["v69"] + -1.0)/2.0) * 2.0)),  (((((-1.0 != 5.200000)) >= data["v70"]).astype(float)))))/2.0)).astype(float))) +
                    np.minimum( (np.minimum( (((data["v106"] > np.maximum( (data["v61"]),  ((1.584910 + (((((data["v109"] > data["v61"]).astype(float)) > data["v84"]).astype(float)) / 2.0))))).astype(float))),  (np.cos(data["v48"])))),  (np.cos(((data["v111"] + data["v88"])/2.0)))) +
                    ((data["v99"] <= np.floor((data["v65"] - (np.sin(data["v25"]) + np.cos((1.0/(1.0 + np.exp(- (np.floor((data["v101"] - data["v56"])) / 2.0))))))))).astype(float)) +
                    np.tanh(np.floor(np.cos((((data["v122"] > (data["v2"] + np.cos(np.floor(np.cos(((data["v2"] + data["v122"])/2.0)))))).astype(float)) + ((data["v2"] + (1.0/(1.0 + np.exp(- np.minimum( (data["v2"]),  (data["v114"]))))))/2.0))))) +
                    (np.sinh(data["v36"]) * (0.094340 * ((data["v40"] > (-(np.sinh(np.sinh(np.minimum( (data["v50"]),  (np.sinh(np.sinh(np.minimum( (data["v62"]),  (data["v50"]))))))))))).astype(float)))) +
                    ((np.sin(np.round(((np.cos(np.abs((data["v64"] + np.sinh(data["v125"])))) + np.floor(np.sin(np.round(((data["v38"] + (1.0/(1.0 + np.exp(- (3.95176)))))/2.0)))))/2.0))) * 2.0) * 2.0) +
                    ((data["v57"] < (np.minimum( (data["v16"]),  (np.maximum( ((data["v72"] - np.floor(data["v15"]))),  (np.maximum( (data["v66"]),  (np.floor((data["v60"] - data["v72"])))))))) - ((data["v72"] != data["v66"]).astype(float)))).astype(float)) +
                    (0.318310 * (-(((data["v50"] < np.minimum( (np.minimum( (np.minimum( (data["v47"]),  (-1.0))),  (data["v47"]))),  (np.minimum( (data["v110"]),  (np.minimum( (np.floor((0.094340 + data["v50"]))),  (data["v79"]))))))).astype(float))))) +
                    np.round(np.sin(np.maximum( (data["v94"]),  (np.ceil(np.maximum( (np.maximum( (np.ceil(np.maximum( (((data["v131"] + (data["v72"] + ((data["v111"] <= data["v65"]).astype(float))))/2.0)),  (np.round(data["v27"]))))),  (3.0))),  (3.0))))))) +
                    ((((((6.10517) < (data["v129"] + (1.0/(1.0 + np.exp(- data["v24"]))))).astype(float)) >= ((np.maximum( (data["v2"]),  (((data["v127"] < ((data["v127"] < data["v45"]).astype(float))).astype(float)))) >= np.minimum( (data["v88"]),  ((-(data["v4"]))))).astype(float))).astype(float)) * 2.0) +
                    (-(((3.141593 < (np.round((data["v129"] * np.floor(np.floor(((((data["v128"] < np.minimum( (2.718282),  (data["v55"]))).astype(float)) <= ((data["v23"] <= np.floor(data["v117"])).astype(float))).astype(float)))))) * data["v122"])).astype(float)))) +
                    np.abs(np.minimum( (0.138462),  (((((data["v79"] + (data["v56"] * 2.0))/2.0) > ((data["v120"] != np.maximum( (np.round(np.ceil((np.maximum( (data["v56"]),  (9.869604)) + data["v52"])))),  (9.869604))).astype(float))).astype(float))))) +
                    np.sinh(np.sinh(np.sinh(((data["v2"] > np.ceil(np.ceil((np.maximum( (data["v27"]),  (2.718282)) + (data["v86"] - ((data["v90"] < ((data["v79"] < data["v122"]).astype(float))).astype(float))))))).astype(float))))) +
                    (0.058823 * (-(((data["v56"] + (np.maximum( (np.maximum( ((data["v56"] - np.sinh(data["v34"]))),  ((data["v38"] + np.maximum( (data["v80"]),  (data["v129"])))))),  (np.sinh(data["v73"]))) + data["v52"]))/2.0)))) +
                    np.floor(np.sin((((np.sinh(((data["v109"] <= np.abs(data["v40"])).astype(float))) <= data["v122"]).astype(float)) + ((2.675680 + (((data["v35"] + ((data["v12"] <= data["v39"]).astype(float)))/2.0) / 2.0))/2.0)))) +
                    ((((np.floor(data["v81"]) >= (data["v36"] + ((data["v103"] >= ((5.200000 <= data["v39"]).astype(float))).astype(float)))).astype(float)) > np.cos(((np.abs((data["v5"] * data["v36"])) + data["v9"])/2.0))).astype(float)) +
                    (np.round(((((data["v19"] / 2.0) / 2.0) + (((((((1.197370 == ((data["v44"] / 2.0) + ((data["v107"] >= np.round(data["v114"])).astype(float)))).astype(float)) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0))/2.0)) * 2.0) +
                    np.floor((np.minimum( (data["v55"]),  (np.minimum( ((data["v69"] - data["v31"])),  (np.floor(((data["v69"] + np.sinh((data["v108"] - data["v90"])))/2.0)))))) * ((3.141593 < data["v131"]).astype(float)))) +
                    (np.floor(((np.floor(np.abs(np.sin((data["v53"] + np.maximum( (np.maximum( (data["v97"]),  ((np.sinh(np.ceil(np.abs(data["v12"]))) / 2.0)))),  (((data["v10"] >= data["v63"]).astype(float)))))))) * 2.0) * 2.0)) * 2.0) +
                    (np.maximum( ((data["v107"] * ((data["v34"] > 2.302585).astype(float)))),  ((0.058823 * ((data["v79"] >= 0.058823).astype(float))))) + (-(np.floor(np.cos(np.minimum( (data["v17"]),  (((data["v79"] + data["v82"])/2.0)))))))) +
                    (((np.cos(np.minimum( (np.cos((data["v21"] - data["v120"]))),  (np.minimum( ((data["v77"] * data["v107"])),  (data["v54"]))))) >= ((data["v1"] < ((9.869604 + data["v126"])/2.0)).astype(float))).astype(float)) * 2.0) +
                    np.sinh(np.sinh(np.sinh(np.floor(np.sin((0.720430 + ((data["v94"] + np.maximum( (np.minimum( (np.maximum( (data["v65"]),  (((data["v41"] == data["v41"]).astype(float))))),  (data["v61"]))),  (np.round(np.cos(data["v128"])))))/2.0))))))) +
                    (np.floor(np.cos((((data["v99"] > (-(np.sinh((data["v107"] + ((2.212120 - np.floor(data["v118"])) * 2.0)))))).astype(float)) * (np.cos(np.maximum( (data["v93"]),  (data["v120"]))) / 2.0)))) * 2.0) +
                    np.round(np.minimum( ((data["v48"] * data["v29"])),  (np.round((data["v97"] * (np.sinh(np.sin(((3.0 < np.maximum( (np.round(data["v10"])),  (np.round((data["v1"] * data["v10"]))))).astype(float)))) / 2.0)))))) +
                    np.floor(np.minimum( ((-(((np.ceil(np.cos(data["v29"])) == np.cos(data["v50"])).astype(float))))),  (np.cos((data["v29"] * ((data["v44"] + ((((np.sin(data["v52"]) >= data["v62"]).astype(float)) >= data["v71"]).astype(float)))/2.0)))))) +
                    np.minimum( (np.cos(((data["v85"] + np.sinh(((((data["v4"] > ((data["v130"] <= data["v29"]).astype(float))).astype(float)) > np.cos(data["v30"])).astype(float))))/2.0))),  (np.tanh(((np.cos(data["v29"]) < ((data["v29"] == data["v98"]).astype(float))).astype(float))))) +
                    (np.floor((np.minimum( (((data["v90"] + ((data["v72"] >= data["v14"]).astype(float)))/2.0)),  (data["v90"])) * ((2.675680 <= ((data["v105"] + (((data["v105"] >= np.cos(data["v72"])).astype(float)) * data["v129"]))/2.0)).astype(float)))) * 2.0) +
                    np.floor(np.cos(((((((data["v81"] < 0.138462).astype(float)) + np.maximum( (0.138462),  (data["v30"])))/2.0) + np.maximum( ((-(data["v29"]))),  ((-((data["v29"] * data["v114"]))))))/2.0))) +
                    ((((-(0.720430)) >= np.minimum( (np.cos(np.floor((data["v72"] * ((data["v89"] > (data["v51"] * data["v36"])).astype(float)))))),  (np.cos(np.floor((data["v60"] * np.floor(data["v13"]))))))).astype(float)) * 2.0) +
                    ((1.570796 <= np.minimum( (data["v115"]),  (np.maximum( (np.minimum( (data["v12"]),  (np.maximum( (data["v72"]),  (data["v3"]))))),  (np.maximum( (data["v64"]),  (np.maximum( ((data["v78"] * data["v72"])),  (np.maximum( (data["v78"]),  (np.floor(data["v98"])))))))))))).astype(float)) +
                    (-((((data["v39"] > np.maximum( (data["v19"]),  ((data["v42"] * data["v39"])))).astype(float)) * ((data["v105"] >= ((np.maximum( (((data["v105"] >= data["v67"]).astype(float))),  (data["v42"])) + data["v58"])/2.0)).astype(float))))) +
                    ((np.abs(((np.cos(data["v58"]) >= ((np.sin((1.732051 * data["v13"])) < ((np.sin(((data["v58"] * 2.0) * 2.0)) < (((data["v13"] * 2.0) != 1.732051).astype(float))).astype(float))).astype(float))).astype(float))) * 2.0) * 2.0) +
                    ((-(((np.minimum( (((data["v117"] < np.sin(data["v97"])).astype(float))),  (((8.0 <= data["v39"]).astype(float)))) >= (((((data["v97"] == 1.630430).astype(float)) + 2.675680) > np.maximum( (data["v45"]),  (data["v97"]))).astype(float))).astype(float)))) * 2.0) +
                    ((((np.cos((data["v105"] * data["v130"])) == ((data["v114"] < data["v68"]).astype(float))).astype(float)) > (data["v6"] * ((data["v68"] >= np.abs(np.cos((np.minimum( (data["v107"]),  (data["v125"])) * data["v114"])))).astype(float)))).astype(float)) +
                    (20.750000 * np.floor(np.maximum( (np.sin((data["v73"] * np.maximum( ((data["v7"] * 2.0)),  (data["v109"]))))),  (np.cos(np.cos(((data["v55"] * 2.0) + np.cos(np.floor(data["v7"]))))))))) +
                    (-(((np.sin(data["v51"]) >= ((data["v113"] <= ((np.maximum( (np.maximum( (data["v118"]),  (((np.sinh(data["v85"]) < data["v54"]).astype(float))))),  (((data["v85"] < data["v118"]).astype(float)))) != ((data["v5"] < data["v74"]).astype(float))).astype(float))).astype(float))).astype(float)))) +
                    np.floor(np.maximum( (np.cos((data["v69"] - np.cos(((data["v69"] - np.cos(((data["v100"] > data["v69"]).astype(float)))) / 2.0))))),  (np.cos(np.tanh(((1.0/(1.0 + np.exp(- data["v4"]))) - data["v44"])))))) +
                    (np.round((np.sinh(np.floor(np.cos(((((data["v34"] + data["v104"])/2.0) + ((data["v88"] >= (1.0/(1.0 + np.exp(- data["v89"])))).astype(float)))/2.0)))) * 2.0)) * 2.0) +
                    ((np.tanh(np.minimum( (data["v62"]),  ((((data["v29"] + np.sin(data["v29"])) > data["v8"]).astype(float))))) == ((((np.ceil(((data["v29"] + (data["v30"] + data["v29"]))/2.0)) >= data["v122"]).astype(float)) >= data["v2"]).astype(float))).astype(float)) +
                    ((1.0/(1.0 + np.exp(- ((1.0/(1.0 + np.exp(- data["v103"]))) / 2.0)))) * ((round(-3.0) > (data["v107"] + (data["v114"] * (data["v61"] * np.minimum( (data["v52"]),  (((data["v2"] * data["v80"]) * 2.0))))))).astype(float))) +
                    (np.maximum( (data["v82"]),  (((data["v82"] - data["v117"]) * data["v34"]))) * (data["v117"] * ((3.0 < (data["v34"] + np.sin(np.tanh(1.584910)))).astype(float)))) +
                    np.cos(np.maximum( (np.minimum( ((np.minimum( (data["v12"]),  (data["v50"])) + data["v129"])),  (data["v50"]))),  (np.maximum( ((data["v2"] - ((data["v106"] >= np.maximum( (data["v122"]),  (data["v50"]))).astype(float)))),  (1.584910))))) +
                    ((np.floor(((np.cos(((0.602941 + (data["v48"] - (((1.0/(1.0 + np.exp(- data["v52"]))) >= data["v111"]).astype(float))))/2.0)) + np.ceil(((data["v121"] >= 0.058823).astype(float))))/2.0)) * 2.0) * 2.0))

    return Outputs(predictions)


def GPIndividual4(data):
    predictions = ((-1.0 - ((np.maximum( (data["v79"]),  (data["v66"])) + (((data["v56"] + np.sin((data["v50"] + (((((data["v79"] > -2.0).astype(float)) + data["v50"]) > 0.138462).astype(float)))))/2.0) + data["v50"]))/2.0)) +
                    np.tanh(((data["v66"] * data["v74"]) * ((data["v40"] - (-(data["v10"]))) - (data["v66"] * ((data["v66"] * data["v74"]) - (((data["v66"] / 2.0) <= data["v50"]).astype(float))))))) +
                    np.minimum( ((np.sin(np.tanh(((data["v3"] >= np.maximum( (data["v31"]),  (np.ceil(np.ceil(data["v50"]))))).astype(float)))) / 2.0)),  (((data["v12"] + (1.584910 * ((data["v129"] < ((data["v12"] >= data["v3"]).astype(float))).astype(float))))/2.0))) +
                    (0.058823 * ((((((data["v114"] - data["v125"]) - data["v24"]) - ((data["v21"] >= 0.301030).astype(float))) - data["v38"]) - data["v66"]) - ((data["v21"] >= 0.058823).astype(float)))) +
                    ((data["v10"] < (((np.sinh(((data["v10"] < (-(1.197370))).astype(float))) * (-(data["v12"]))) + (-(np.sinh((np.ceil(np.cos(data["v42"])) * 2.0))))) / 2.0)).astype(float)) +
                    ((np.round(data["v47"]) * np.tanh(np.sinh((((data["v34"] + np.abs((data["v56"] * np.round(((data["v24"] < ((data["v24"] < data["v47"]).astype(float))).astype(float))))))/2.0) / 2.0)))) / 2.0) +
                    np.minimum( (np.minimum( ((((data["v110"] < data["v56"]).astype(float)) / 2.0)),  (((np.abs(data["v66"]) < (((data["v56"] < np.tanh(np.maximum( (data["v66"]),  (0.138462)))).astype(float)) / 2.0)).astype(float))))),  (((data["v45"] < data["v51"]).astype(float)))) +
                    np.sin(np.ceil(np.ceil(((data["v3"] + (data["v66"] * ((data["v3"] + (((data["v66"] >= np.round(data["v82"])).astype(float)) * (-3.0 * (((data["v129"] * 2.0) >= data["v129"]).astype(float)))))/2.0)))/2.0)))) +
                    np.sinh(((data["v50"] <= np.sinh(np.floor((-(((np.maximum( (data["v50"]),  (data["v66"])) + (1.0/(1.0 + np.exp(- 2.0))))/2.0)))))).astype(float))) +
                    (0.094340 * (np.sin((data["v34"] + np.minimum( ((np.round(data["v47"]) * data["v66"])),  (data["v66"])))) + np.sin((data["v66"] + (data["v66"] - data["v56"]))))) +
                    (0.693147 - np.cos((1.0/(1.0 + np.exp(- np.maximum( (np.round(data["v79"])),  (np.maximum( (data["v50"]),  (np.maximum( (0.094340),  (np.abs(((((data["v66"] + data["v50"])/2.0) + data["v75"]) - data["v66"]))))))))))))) +
                    np.minimum( ((0.585714 - (1.0/(1.0 + np.exp(- np.minimum( (data["v24"]),  (((data["v47"] >= 0.585714).astype(float))))))))),  (((data["v119"] >= np.maximum( (data["v80"]),  (np.maximum( (data["v113"]),  (data["v24"]))))).astype(float)))) +
                    (-(np.tanh(((np.cos(((data["v72"] + np.minimum( (data["v38"]),  (np.floor(data["v10"]))))/2.0)) < (((1.584910 * ((data["v127"] / 2.0) / 2.0)) < np.minimum( (data["v38"]),  (data["v1"]))).astype(float))).astype(float))))) +
                    (-(((data["v113"] >= np.ceil(np.cos(np.maximum( (np.maximum( (data["v49"]),  (((data["v113"] + np.maximum( (data["v69"]),  (((data["v52"] + data["v72"])/2.0))))/2.0)))),  ((data["v49"] * np.maximum( (data["v90"]),  (data["v36"])))))))).astype(float)))) +
                    (data["v113"] * (np.tanh(np.minimum( (0.094340),  (((np.round(data["v66"]) >= ((data["v56"] < ((data["v14"] + np.maximum( (np.sinh(((0.058823 > data["v16"]).astype(float)))),  (data["v88"])))/2.0)).astype(float))).astype(float))))) * 2.0)) +
                    (data["v88"] * np.round((((1.0/(1.0 + np.exp(- data["v38"]))) + (data["v108"] * ((((data["v108"] - np.round(data["v21"])) * 2.0) - data["v129"]) * ((data["v21"] >= (1.732051 * 2.0)).astype(float)))))/2.0))) +
                    (((5.428570 <= np.maximum( (np.abs(np.maximum( ((np.sinh(data["v51"]) * ((data["v129"] + (-(np.sinh(data["v88"]))))/2.0))),  (data["v16"])))),  (data["v51"]))).astype(float)) * 2.0) +
                    (np.floor((np.floor(np.cos((data["v97"] * (1.0/(1.0 + np.exp(- ((data["v36"] > ((data["v3"] <= data["v27"]).astype(float))).astype(float)))))))) - ((data["v3"] >= np.maximum( (9.869604),  (9.869604))).astype(float)))) * 2.0) +
                    ((655.0) * ((data["v11"] >= (data["v93"] + np.maximum( (data["v41"]),  (np.maximum( (data["v83"]),  (np.maximum( (data["v53"]),  (((data["v40"] < data["v75"]).astype(float)))))))))).astype(float))) +
                    ((((np.sin(np.sin(np.maximum( (np.maximum( (data["v2"]),  (np.abs(data["v21"])))),  (data["v21"])))) <= ((2.718282 <= np.maximum( (data["v117"]),  (data["v67"]))).astype(float))).astype(float)) > np.round((data["v66"] * data["v3"]))).astype(float)) +
                    np.minimum( (np.cos((data["v27"] * np.tanh(data["v1"])))),  (np.floor(np.cos((data["v80"] * np.maximum( (0.623656),  ((data["v7"] - np.maximum( (data["v98"]),  (np.round(np.floor(np.cos(data["v80"]))))))))))))) +
                    (((3.0 < (((3.0 + data["v84"])/2.0) - ((data["v114"] <= (((np.sinh(data["v124"]) / 2.0) <= ((np.round(data["v1"]) != 3.0).astype(float))).astype(float))).astype(float)))).astype(float)) * 2.0) +
                    np.minimum( ((data["v56"] * np.cos(np.maximum( (np.maximum( (1.584910),  (((data["v76"] + np.ceil(data["v10"]))/2.0)))),  (((data["v76"] + data["v97"])/2.0)))))),  (np.cos(np.minimum( (np.ceil(np.ceil(data["v72"]))),  (data["v107"]))))) +
                    ((((((np.round(np.cos(data["v19"])) > data["v93"]).astype(float)) == np.cos(((data["v93"] * np.sin(data["v55"])) * data["v109"]))).astype(float)) >= ((data["v117"] <= ((3.141593 + data["v109"])/2.0)).astype(float))).astype(float)) +
                    (((5.200000 - np.minimum( ((data["v24"] * data["v127"])),  (data["v66"]))) <= (data["v50"] * (np.minimum( (data["v75"]),  ((np.minimum( (0.094340),  (data["v66"])) * 2.0))) * 2.0))).astype(float)) +
                    (np.cos(np.maximum( (np.maximum( ((-(data["v50"]))),  (np.maximum( (np.maximum( (data["v90"]),  (((data["v66"] + np.ceil(data["v79"]))/2.0)))),  (np.minimum( (data["v97"]),  (data["v87"]))))))),  (np.maximum( (1.584910),  (data["v90"]))))) * 2.0) +
                    np.abs((np.floor((np.maximum( (data["v24"]),  (np.tanh(np.floor(np.maximum( (data["v79"]),  (data["v79"])))))) * data["v79"])) * ((data["v79"] - data["v10"]) * (data["v112"] * 0.094340)))) +
                    ((1.414214 < (data["v67"] - (((data["v57"] > ((1.414214 < (data["v67"] - ((data["v19"] <= ((data["v52"] <= (data["v67"] + data["v129"])).astype(float))).astype(float)))).astype(float))).astype(float)) / 2.0))).astype(float)) +
                    ((0.602941 - (1.0/(1.0 + np.exp(- (((((data["v40"] * data["v112"]) + data["v52"])/2.0) + np.round(np.maximum( (data["v40"]),  (((data["v107"] <= np.maximum( (data["v110"]),  (data["v30"]))).astype(float))))))/2.0))))) / 2.0) +
                    np.minimum( (np.cos(((data["v30"] + (-((data["v27"] / 2.0))))/2.0))),  (np.minimum( (((data["v69"] < np.minimum( (data["v57"]),  (np.minimum( (data["v38"]),  (data["v4"]))))).astype(float))),  (np.cos((-(((data["v27"] + data["v4"])/2.0)))))))) +
                    (-(((((np.minimum( (data["v34"]),  (((1.0/(1.0 + np.exp(- (data["v24"] + np.maximum( (data["v126"]),  ((-(data["v97"])))))))) * data["v38"]))) > np.cos(np.cos(data["v24"]))).astype(float)) > (1.0/(1.0 + np.exp(- data["v38"])))).astype(float)))) +
                    (((((data["v16"] > 1.732051).astype(float)) * ((data["v3"] > data["v108"]).astype(float))) != (((data["v27"] > 1.732051).astype(float)) * (((data["v27"] > data["v124"]).astype(float)) * ((data["v124"] > (1.0/(1.0 + np.exp(- data["v3"])))).astype(float))))).astype(float)) +
                    np.maximum( (np.round(((-2.0 > (np.maximum( (data["v66"]),  (data["v129"])) - (data["v50"] / 2.0))).astype(float)))),  (((data["v67"] >= (np.sin(data["v99"]) + ((data["v99"] != data["v50"]).astype(float)))).astype(float)))) +
                    np.minimum( ((((data["v123"] >= data["v24"]).astype(float)) * np.minimum( (((data["v69"] == data["v101"]).astype(float))),  (np.cos(np.minimum( (data["v21"]),  (data["v76"]))))))),  (((np.maximum( (data["v20"]),  (np.cos(data["v31"]))) + np.cos(data["v21"]))/2.0))) +
                    (-(((np.maximum( (np.minimum( (np.maximum( ((data["v58"] * 2.0)),  (data["v87"]))),  (data["v34"]))),  (((np.cos(data["v58"]) + data["v34"]) / 2.0))) > np.maximum( (1.630430),  ((data["v112"] * data["v58"])))).astype(float)))) +
                    ((np.sin(np.abs(data["v14"])) <= np.floor(np.cos(((((data["v111"] + np.minimum( (data["v24"]),  (np.tanh(np.minimum( (data["v24"]),  (np.tanh((data["v15"] - data["v24"]))))))))/2.0) / 2.0) / 2.0)))).astype(float)) +
                    ((0.138462 * ((data["v31"] >= (((np.maximum( (data["v87"]),  (np.maximum( (((data["v85"] != (np.cos(data["v7"]) / 2.0)).astype(float))),  (data["v32"])))) > ((data["v88"] < data["v11"]).astype(float))).astype(float)) * 2.0)).astype(float))) * 2.0) +
                    ((((0.434294 > np.maximum( (data["v48"]),  (((data["v119"] < data["v125"]).astype(float))))).astype(float)) > np.maximum( (data["v95"]),  (np.maximum( (data["v53"]),  (((data["v119"] < (1.0/(1.0 + np.exp(- ((data["v71"] < np.tanh(data["v56"])).astype(float)))))).astype(float))))))).astype(float)) +
                    ((np.cos(((np.sinh(data["v110"]) - data["v59"]) * np.minimum( (data["v54"]),  (np.minimum( (data["v56"]),  (np.ceil(data["v29"]))))))) >= ((5.200000 >= np.abs(3.141593)).astype(float))).astype(float)) +
                    np.floor((data["v68"] * np.minimum( (np.floor(np.cos(((data["v5"] + np.tanh(np.abs(np.sinh((data["v60"] * data["v114"])))))/2.0)))),  (np.sin(np.abs(data["v114"])))))) +
                    np.floor(np.cos(((data["v35"] + np.maximum( (np.maximum( (data["v6"]),  (np.maximum( (np.maximum( (data["v114"]),  (data["v7"]))),  ((np.cos((data["v61"] * 2.0)) * 2.0)))))),  (np.maximum( (data["v61"]),  ((data["v126"] - data["v63"]))))))/2.0))) +
                    (((np.abs(np.abs(np.abs(np.ceil(np.maximum( (data["v115"]),  (data["v43"])))))) == np.tanh((data["v38"] + np.abs(np.ceil(data["v86"]))))).astype(float)) * np.maximum( (np.sinh(data["v73"])),  (np.abs(data["v120"])))) +
                    ((np.tanh(data["v17"]) > np.maximum( (data["v113"]),  (((data["v86"] >= np.minimum( (data["v49"]),  ((np.minimum( (data["v3"]),  (np.cos(data["v54"]))) / 2.0)))).astype(float))))).astype(float)) +
                    np.sin(np.minimum( (data["v71"]),  ((-3.0 * ((data["v6"] > (((data["v102"] >= ((data["v6"] > data["v117"]).astype(float))).astype(float)) - np.sin(np.minimum( (data["v74"]),  (-2.0))))).astype(float)))))) +
                    ((1.414214 < np.minimum( (data["v126"]),  ((data["v76"] + np.minimum( ((data["v126"] + np.floor(np.floor(np.cos(data["v2"]))))),  (((data["v20"] >= np.cos((0.693147 - np.cos(data["v88"])))).astype(float)))))))).astype(float)) +
                    (((data["v131"] + np.abs(data["v21"]))/2.0) * (data["v120"] * np.floor(np.floor(np.cos((5.428570 + (1.0/(1.0 + np.exp(- (data["v68"] + ((0.693147 != np.sinh(data["v83"])).astype(float)))))))))))) +
                    ((5.200000 <= (data["v13"] * np.maximum( (np.maximum( (data["v99"]),  (data["v40"]))),  (np.maximum( (np.maximum( (((data["v55"] + data["v88"])/2.0)),  ((data["v113"] * data["v114"])))),  (data["v2"])))))).astype(float)) +
                    (((((np.floor(np.sin(np.maximum( (((data["v120"] + 1.0) / 2.0)),  (np.maximum( (data["v7"]),  (np.abs(((data["v123"] + (1.0/(1.0 + np.exp(- data["v121"]))))/2.0)))))))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) +
                    (-(((data["v27"] >= (2.675680 - (((data["v116"] - data["v117"]) > (((data["v116"] >= ((np.floor(data["v83"]) != (((data["v116"] - data["v93"]) != 0.367879).astype(float))).astype(float))).astype(float)) * 2.0)).astype(float)))).astype(float)))) +
                    np.floor((((((1.197370 < np.floor(data["v6"])).astype(float)) * data["v127"]) * ((data["v112"] > ((data["v51"] <= data["v2"]).astype(float))).astype(float))) * (1.197370 * ((data["v51"] > data["v30"]).astype(float))))) +
                    ((np.sinh(data["v1"]) * data["v71"]) * ((np.sinh(((data["v24"] > np.abs((data["v51"] * (data["v30"] + data["v51"])))).astype(float))) >= (3.141593 - data["v51"])).astype(float))) +
                    (((data["v39"] + data["v126"]) * ((data["v75"] + ((np.ceil(np.maximum( (data["v43"]),  (data["v126"]))) != ((data["v75"] < data["v43"]).astype(float))).astype(float))) * ((data["v5"] >= 2.212120).astype(float)))) / 2.0) +
                    np.sinh(np.sinh(np.sinh(((np.ceil(np.abs((data["v68"] + np.ceil((data["v68"] + ((((data["v68"] == data["v9"]).astype(float)) + data["v98"])/2.0)))))) < np.round(data["v9"])).astype(float))))) +
                    np.minimum( (np.cos(data["v6"])),  (((data["v86"] * data["v39"]) * np.ceil(((np.ceil(data["v70"]) <= np.minimum( (data["v51"]),  (((np.minimum( (data["v86"]),  (((data["v51"] <= data["v6"]).astype(float)))) == data["v86"]).astype(float))))).astype(float)))))) +
                    (1.630430 * ((((data["v92"] < (data["v15"] - 2.675680)).astype(float)) >= np.cos((data["v32"] * ((1.630430 < (data["v32"] * ((data["v104"] < ((data["v50"] <= data["v114"]).astype(float))).astype(float)))).astype(float))))).astype(float))) +
                    np.minimum( (np.cos(data["v9"])),  ((np.minimum( (np.minimum( (data["v66"]),  ((1.732051 * data["v51"])))),  (data["v70"])) * ((data["v56"] > (((1.732051 >= np.maximum( (data["v120"]),  (data["v122"]))).astype(float)) * 2.0)).astype(float))))) +
                    np.sinh(np.sinh(np.floor(np.sinh(np.floor(np.cos(((data["v71"] + np.maximum( (((((data["v14"] <= 2.302585).astype(float)) == ((2.302585 > data["v125"]).astype(float))).astype(float))),  (np.minimum( ((data["v119"] * 2.0)),  (data["v30"])))))/2.0))))))) +
                    ((((data["v66"] + np.sinh(((((data["v50"] / 2.0) / 2.0) > data["v129"]).astype(float)))) < data["v50"]).astype(float)) * ((np.minimum( (data["v50"]),  (np.cos((data["v66"] + 2.212120)))) / 2.0) / 2.0)) +
                    (np.sinh(np.floor(np.sin((data["v7"] * ((data["v90"] >= np.maximum( (data["v129"]),  ((np.tanh(data["v40"]) + ((data["v103"] <= data["v62"]).astype(float)))))).astype(float)))))) * 2.0) +
                    ((2.409090 < np.maximum( (np.maximum( (data["v92"]),  (data["v9"]))),  ((data["v95"] + ((np.minimum( (data["v69"]),  (np.minimum( (data["v55"]),  (data["v114"])))) > ((((1.0/(1.0 + np.exp(- data["v96"]))) * 2.0) > data["v69"]).astype(float))).astype(float)))))).astype(float)) +
                    (np.sinh(((((data["v120"] != data["v31"]).astype(float)) == np.cos((data["v85"] * (2.302585 * (((data["v85"] + (2.302585 * (data["v31"] / 2.0)))/2.0) / 2.0))))).astype(float))) * 2.0) +
                    ((data["v123"] <= np.floor((-(np.tanh((np.sin(data["v27"]) * (data["v100"] + np.minimum( (data["v121"]),  (((((np.cos(data["v21"]) <= data["v65"]).astype(float)) >= data["v65"]).astype(float))))))))))).astype(float)) +
                    (np.floor(np.cos(((((data["v40"] + ((data["v86"] + (-((((data["v127"] - data["v39"]) >= ((data["v24"] == data["v86"]).astype(float))).astype(float)))))/2.0))/2.0) / 2.0) / 2.0))) * 2.0) +
                    ((13.1357) * ((np.round(np.floor(np.abs(np.cos((data["v32"] * (2.718282 + ((data["v130"] > data["v62"]).astype(float)))))))) == ((data["v86"] != data["v119"]).astype(float))).astype(float))))

    return Outputs(predictions)


def GPIndividual5(data):
    predictions = ((-((((np.minimum( (data["v50"]),  ((1.0/(1.0 + np.exp(- data["v66"]))))) + np.maximum( (((((data["v79"] + data["v56"])/2.0) + 2.718282)/2.0)),  (np.sinh(np.maximum( (np.maximum( (data["v66"]),  (data["v47"]))),  (data["v79"]))))))/2.0) * 2.0))) +
                    (np.sin((data["v74"] * (np.ceil((data["v129"] + data["v24"])) + data["v38"]))) - (((np.round((data["v50"] / 2.0)) / 2.0) + ((data["v38"] > ((1.570796 + data["v50"])/2.0)).astype(float)))/2.0)) +
                    np.minimum( ((((data["v3"] > np.ceil(np.minimum( (np.sinh(data["v50"])),  (data["v50"])))).astype(float)) * 2.0)),  (((((np.sinh(data["v50"]) + (data["v50"] * data["v50"]))/2.0) >= np.sinh(data["v66"])).astype(float)))) +
                    (0.094340 * (data["v34"] - (((data["v125"] + ((data["v40"] * 2.0) * data["v79"]))/2.0) + np.maximum( (data["v14"]),  (((((data["v40"] < data["v79"]).astype(float)) < (1.0/(1.0 + np.exp(- 0.058823)))).astype(float))))))) +
                    np.minimum( ((np.minimum( (0.138462),  (np.maximum( (data["v50"]),  (np.cos(data["v66"]))))) * np.sin(np.ceil(data["v50"])))),  (np.maximum( ((data["v105"] - data["v66"])),  (np.cos(data["v50"]))))) +
                    ((1.197370 < (data["v10"] * np.floor(np.tanh(((-2.0 + np.maximum( ((1.197370 * np.sinh(data["v21"]))),  (np.maximum( ((data["v14"] * data["v21"])),  (np.sinh(data["v82"]))))))/2.0))))).astype(float)) +
                    ((-(np.minimum( (np.maximum( (0.058823),  (np.round(data["v79"])))),  (np.cos((np.round(((data["v113"] + np.sinh(data["v79"]))/2.0)) * (np.round(data["v79"]) * ((data["v66"] <= 0.058823).astype(float))))))))) / 2.0) +
                    (np.cos(np.sinh(np.maximum( (5.428570),  (data["v3"])))) - (1.0/(1.0 + np.exp(- np.maximum( ((1.0/(1.0 + np.exp(- data["v21"])))),  ((np.maximum( (data["v6"]),  ((data["v75"] * data["v66"]))) - np.tanh((data["v75"] * data["v50"]))))))))) +
                    (-(np.floor(np.sin(np.maximum( (np.maximum( (np.maximum( (np.maximum( (np.round(data["v21"])),  (data["v73"]))),  (np.maximum( (np.maximum( ((-(data["v21"]))),  (np.floor(data["v16"])))),  (data["v2"]))))),  ((-(data["v14"]))))),  (data["v29"])))))) +
                    np.minimum( (np.cos(data["v90"])),  (np.minimum( ((-(((data["v113"] >= (((9.91924) >= (data["v72"] * 2.0)).astype(float))).astype(float))))),  (np.ceil(((data["v12"] + np.round(data["v62"]))/2.0)))))) +
                    (0.094340 * ((((data["v40"] * np.sinh(np.minimum( (data["v66"]),  ((data["v3"] * data["v66"]))))) * np.round(data["v66"])) + (-(data["v40"])))/2.0)) +
                    ((0.301030 / 2.0) * ((data["v79"] + ((data["v119"] + (np.sinh(np.minimum( ((data["v34"] - data["v47"])),  ((-(data["v24"]))))) * data["v47"]))/2.0))/2.0)) +
                    ((((1.732051 <= (data["v66"] * np.minimum( (np.minimum( (data["v50"]),  ((-((np.maximum( (data["v14"]),  (data["v77"])) / 2.0)))))),  (np.cos((data["v50"] - data["v75"])))))).astype(float)) > np.cos(data["v75"])).astype(float)) +
                    (((((data["v88"] > np.sinh(np.maximum( (2.212120),  (((data["v59"] + (2.212120 * np.floor(data["v39"])))/2.0))))).astype(float)) == ((((5.200000 + data["v74"])/2.0) > data["v117"]).astype(float))).astype(float)) * 2.0) +
                    np.sinh((np.maximum( (((-1.0 + (-(data["v10"])))/2.0)),  (np.floor(np.floor(np.cos(((data["v97"] + ((data["v14"] >= ((data["v52"] >= np.ceil(data["v7"])).astype(float))).astype(float)))/2.0)))))) * 2.0)) +
                    np.sin(np.minimum( (data["v71"]),  (np.minimum( (((((1.0/(1.0 + np.exp(- data["v100"]))) * (np.cos(data["v97"]) * np.ceil(data["v56"]))) / 2.0) / 2.0)),  ((0.094340 * (data["v56"] * data["v47"]))))))) +
                    ((0.585714 + (-((1.0/(1.0 + np.exp(- ((data["v129"] >= (0.585714 - np.maximum( (np.maximum( (data["v56"]),  (((1.0/(1.0 + np.exp(- 0.636620))) - np.maximum( (data["v56"]),  ((-(data["v21"])))))))),  (0.636620)))).astype(float))))))))/2.0) +
                    ((((data["v16"] / 2.0) <= ((data["v6"] != np.tanh(data["v80"])).astype(float))).astype(float)) * np.floor(np.sin(np.abs(((data["v30"] + (-(np.sinh(np.maximum( (data["v80"]),  (((data["v18"] + data["v50"])/2.0)))))))/2.0))))) +
                    (data["v79"] * np.floor(((((data["v78"] + ((data["v38"] > data["v103"]).astype(float)))/2.0) + np.cos((((data["v123"] > np.floor(data["v19"])).astype(float)) / 2.0)))/2.0))) +
                    np.minimum( (np.maximum( (data["v24"]),  (np.minimum( (np.cos(data["v21"])),  (5.428570))))),  ((-(((data["v4"] > (data["v17"] + ((((2.675680 <= (data["v17"] + data["v17"])).astype(float)) <= data["v41"]).astype(float)))).astype(float)))))) +
                    ((0.094340 * (((data["v30"] * data["v52"]) <= ((data["v10"] > np.tanh(data["v125"])).astype(float))).astype(float))) * (((data["v30"] * 2.0) <= ((((data["v102"] > data["v10"]).astype(float)) >= data["v110"]).astype(float))).astype(float))) +
                    np.ceil((data["v21"] * (5.428570 * (data["v129"] * np.round((data["v23"] * (data["v111"] * ((data["v129"] > (5.428570 + (data["v129"] * data["v100"]))).astype(float))))))))) +
                    (-(((np.cos(((np.cos(data["v14"]) < ((data["v2"] < np.floor(data["v19"])).astype(float))).astype(float))) < (data["v35"] - np.floor(((data["v127"] + 2.409090)/2.0)))).astype(float)))) +
                    (np.maximum( (data["v15"]),  (((data["v84"] > data["v127"]).astype(float)))) * (((data["v84"] > ((np.cos(data["v55"]) + data["v110"])/2.0)).astype(float)) * ((2.0 < (data["v99"] - np.tanh(np.tanh(data["v127"])))).astype(float)))) +
                    np.ceil(np.minimum( (np.floor(np.cos(((data["v5"] + np.tanh(np.round((data["v107"] - data["v10"]))))/2.0)))),  ((data["v107"] - (-(((data["v38"] < np.cos(data["v69"])).astype(float)))))))) +
                    (((data["v27"] / 2.0) >= (((data["v88"] >= data["v95"]).astype(float)) + np.cos(np.sin(np.sin(np.sin(np.sin((data["v108"] - ((data["v88"] < np.round(data["v88"])).astype(float)))))))))).astype(float)) +
                    (np.sinh(np.maximum( (31.006277),  (np.sinh(data["v126"])))) * (np.floor(np.sin(np.maximum( (((data["v70"] + data["v50"])/2.0)),  (np.maximum( (data["v7"]),  (((data["v123"] + np.tanh(2.409090))/2.0))))))) * 2.0)) +
                    ((((data["v35"] != data["v53"]).astype(float)) <= ((np.minimum( (data["v57"]),  (data["v29"])) < np.minimum( (data["v53"]),  (np.minimum( (data["v116"]),  (np.floor(np.minimum( (data["v56"]),  (data["v53"])))))))).astype(float))).astype(float)) +
                    ((np.maximum( ((data["v72"] * (data["v98"] * 1.732051))),  (((1.732051 > data["v115"]).astype(float)))) < ((np.cos(data["v21"]) > (data["v120"] * np.maximum( ((data["v98"] * data["v4"])),  (data["v112"])))).astype(float))).astype(float)) +
                    np.minimum( (np.round((-(((np.minimum( (data["v10"]),  (data["v38"])) > np.abs(data["v13"])).astype(float)))))),  (((data["v33"] + (data["v124"] * data["v108"])) * (data["v118"] + (data["v116"] * data["v23"]))))) +
                    ((np.cos(((data["v4"] + np.sin(data["v100"]))/2.0)) < ((np.cos((data["v51"] / 2.0)) <= ((((1.732051 >= data["v21"]).astype(float)) == np.sin(np.sinh(np.sinh(data["v54"])))).astype(float))).astype(float))).astype(float)) +
                    ((np.abs(np.minimum( (np.sin(np.maximum( (data["v109"]),  ((1.0/(1.0 + np.exp(- data["v21"]))))))),  (np.cos((1.0/(1.0 + np.exp(- data["v107"]))))))) <= np.minimum( (np.maximum( (data["v21"]),  ((1.0/(1.0 + np.exp(- data["v129"])))))),  (np.minimum( (data["v23"]),  (data["v107"]))))).astype(float)) +
                    ((data["v87"] + data["v84"]) * np.floor(np.floor((np.floor(np.sin(((2.409090 + np.maximum( (data["v87"]),  (np.sin((data["v99"] - ((data["v56"] >= data["v87"]).astype(float)))))))/2.0))) * 2.0)))) +
                    np.floor(np.cos(np.minimum( (data["v27"]),  (np.maximum( (np.maximum( (((data["v131"] + ((data["v26"] + data["v53"])/2.0))/2.0)),  ((data["v114"] - np.round(data["v117"]))))),  (np.ceil(np.maximum( (data["v21"]),  (np.ceil(data["v53"])))))))))) +
                    ((data["v68"] <= ((((((0.301030 > np.abs(np.abs(np.cos(np.minimum( (data["v50"]),  ((data["v92"] / 2.0))))))).astype(float)) * 2.0) * 2.0) - ((data["v9"] < 1.584910).astype(float))) * 2.0)).astype(float)) +
                    (5.428570 * (5.428570 * np.floor(np.cos(np.cos(np.maximum( (np.maximum( (data["v120"]),  (np.maximum( (np.sinh(data["v85"])),  ((data["v11"] * data["v6"])))))),  (np.sinh(data["v77"])))))))) +
                    np.sinh(((((np.cos((data["v109"] * np.round(np.sin(data["v20"])))) != np.round(np.floor(data["v9"]))).astype(float)) <= np.cos((((data["v62"] * 2.0) * data["v100"]) * (-(data["v83"]))))).astype(float))) +
                    ((2.212120 < np.minimum( (data["v120"]),  (np.maximum( (data["v72"]),  ((np.maximum( ((data["v105"] - (np.maximum( (data["v96"]),  (np.maximum( ((data["v105"] - np.abs(data["v120"]))),  (data["v72"])))) * 2.0))),  (data["v96"])) * 2.0)))))).astype(float)) +
                    ((np.minimum( (np.maximum( (data["v50"]),  (np.minimum( (np.cos(data["v34"])),  (np.cos(data["v66"])))))),  (np.minimum( (np.cos((data["v129"] * 2.0))),  (((0.0 == (-(np.ceil((data["v56"] * 2.0))))).astype(float)))))) / 2.0) / 2.0) +
                    np.floor(((data["v43"] * np.minimum( (data["v116"]),  (np.minimum( (data["v126"]),  ((data["v43"] * np.minimum( (((data["v25"] >= (np.maximum( (data["v92"]),  (data["v40"])) * 2.0)).astype(float))),  (np.cos(data["v29"]))))))))) / 2.0)) +
                    np.floor(np.maximum( (np.cos(np.minimum( (np.minimum( (data["v21"]),  (np.sinh(np.sinh((data["v40"] + ((data["v124"] >= ((data["v124"] >= data["v21"]).astype(float))).astype(float)))))))),  ((data["v40"] + data["v4"]))))),  (np.cos(data["v94"])))) +
                    ((data["v117"] >= ((data["v109"] + (3.141593 - ((((data["v115"] >= (data["v6"] + np.floor(data["v40"]))).astype(float)) >= np.minimum( (data["v65"]),  (np.ceil((data["v102"] + data["v6"]))))).astype(float))))/2.0)).astype(float)) +
                    (-(((1.570796 < np.minimum( (data["v72"]),  (np.maximum( (np.maximum( (np.maximum( (np.round(np.maximum( (data["v107"]),  (data["v114"])))),  (data["v50"]))),  ((data["v107"] - data["v113"])))),  (data["v7"]))))).astype(float)))) +
                    ((0.094340 * ((data["v12"] >= (8.0 * ((np.abs(data["v112"]) < np.maximum( (np.maximum( (data["v109"]),  (np.maximum( (data["v125"]),  (np.abs(np.ceil(data["v114"]))))))),  (data["v50"]))).astype(float)))).astype(float))) * 2.0) +
                    ((((np.floor(np.cos(((0.138462 + (-(np.cos(data["v8"]))))/2.0))) != np.ceil(data["v94"])).astype(float)) <= (np.floor(np.cos((data["v108"] - np.round(np.cos(data["v73"]))))) * 2.0)).astype(float)) +
                    ((-(((data["v79"] > ((1.630430 - np.sin(np.floor(np.maximum( (data["v23"]),  (((data["v89"] > (-((data["v17"] - ((data["v17"] != data["v79"]).astype(float)))))).astype(float))))))) * 2.0)).astype(float)))) * 2.0) +
                    (((np.tanh(data["v38"]) >= (((((data["v3"] * ((data["v117"] <= (-(data["v72"]))).astype(float))) < 0.0).astype(float)) != np.maximum( (0.693147),  (((data["v50"] > data["v3"]).astype(float))))).astype(float))).astype(float)) / 2.0) +
                    (data["v112"] * (np.floor(data["v65"]) * ((2.409090 < (data["v129"] * np.tanh(((np.maximum( (data["v29"]),  ((data["v24"] - data["v112"]))) + data["v75"])/2.0)))).astype(float)))) +
                    (1.570796 - np.maximum( (1.584910),  (((np.maximum( (data["v118"]),  (data["v52"])) + np.maximum( (1.584910),  ((data["v74"] * np.floor(np.floor(np.abs(((data["v131"] + np.maximum( (data["v27"]),  (data["v52"])))/2.0))))))))/2.0)))) +
                    ((((np.ceil(data["v40"]) - data["v110"]) - data["v113"]) - data["v23"]) * np.minimum( (0.058823),  (np.ceil(((data["v24"] >= ((data["v113"] > data["v60"]).astype(float))).astype(float)))))) +
                    np.round((data["v40"] * ((1.630430 < ((data["v19"] + np.maximum( (data["v97"]),  (((data["v17"] + ((data["v19"] < 1.630430).astype(float)))/2.0))))/2.0)).astype(float)))) +
                    (((data["v6"] + (data["v31"] + ((((8.0 >= (data["v50"] * 2.0)).astype(float)) >= np.cos(data["v88"])).astype(float)))) <= np.minimum( (data["v64"]),  (np.tanh(np.minimum( (data["v95"]),  (np.minimum( (data["v92"]),  (data["v88"])))))))).astype(float)) +
                    np.sinh((5.200000 * ((np.ceil(np.cos(data["v114"])) == np.cos((data["v122"] - np.cos(data["v85"])))).astype(float)))) +
                    (((data["v62"] + (((data["v67"] + 0.720430) + data["v62"]) + data["v15"])) * ((np.sin((data["v67"] + data["v15"])) == np.round(np.maximum( (data["v50"]),  (data["v110"])))).astype(float))) * 2.0) +
                    (((((5.200000 == data["v75"]).astype(float)) < np.round(data["v84"])).astype(float)) * ((data["v75"] < (data["v10"] - (5.200000 + (data["v44"] - data["v81"])))).astype(float))) +
                    ((data["v97"] > (2.675680 - ((((np.minimum( (data["v120"]),  (data["v97"])) > (data["v111"] + data["v76"])).astype(float)) >= (data["v102"] * (((data["v116"] + data["v102"])/2.0) * data["v111"]))).astype(float)))).astype(float)) +
                    ((data["v12"] >= np.sinh((data["v10"] + (1.0/(1.0 + np.exp(- ((np.abs(np.abs(((data["v10"] + (1.0/(1.0 + np.exp(- ((data["v12"] + data["v34"])/2.0))))) * data["v42"]))) + data["v38"])/2.0))))))).astype(float)) +
                    np.minimum( ((-(((2.718282 < np.ceil(((np.maximum( (data["v80"]),  (data["v73"])) + np.maximum( ((-(data["v14"]))),  (data["v117"])))/2.0))).astype(float))))),  (np.floor(np.sin(np.maximum( (data["v116"]),  ((data["v121"] * data["v60"]))))))) +
                    (((2.302585 < (data["v85"] - (((((data["v11"] > data["v57"]).astype(float)) - data["v101"]) + data["v109"])/2.0))).astype(float)) / 2.0) +
                    np.floor(np.cos(np.minimum( (np.maximum( (data["v30"]),  (((data["v6"] + ((data["v120"] + np.ceil(np.maximum( (data["v30"]),  (((np.tanh(2.718282) - data["v72"]) / 2.0)))))/2.0))/2.0)))),  (data["v120"])))) +
                    (data["v54"] * np.floor((np.sin((data["v115"] + ((data["v54"] > data["v90"]).astype(float)))) * np.sin((((data["v83"] + (((data["v83"] + np.minimum( (data["v62"]),  (data["v90"])))/2.0) / 2.0))/2.0) / 2.0))))) +
                    (np.floor(np.sinh((np.floor((np.maximum( (np.minimum( (data["v33"]),  (data["v115"]))),  (np.maximum( ((data["v34"] / 2.0)),  (data["v18"])))) * ((data["v10"] > (5.200000 - data["v115"])).astype(float)))) * 2.0))) * 2.0) +
                    ((((np.ceil(data["v70"]) == np.floor(np.maximum( (np.minimum( (data["v121"]),  ((data["v54"] / 2.0)))),  (((9.869604 == data["v70"]).astype(float)))))).astype(float)) / 2.0) * np.floor((1.732051 - np.cos(data["v65"])))) +
                    (np.sin(np.floor(np.floor((np.cos(data["v9"]) * ((data["v44"] >= (((data["v27"] <= ((data["v118"] >= data["v61"]).astype(float))).astype(float)) + np.ceil(np.minimum( (data["v16"]),  (data["v92"]))))).astype(float)))))) * 2.0))

    return Outputs(predictions)


def GPIndividual6(data):
    predictions = ((-((((data["v50"] + 1.732051) + (np.cos(data["v50"]) + ((((data["v79"] + ((np.maximum( (data["v79"]),  (np.ceil(data["v66"]))) + data["v50"]) * 2.0))/2.0) + data["v56"])/2.0)))/2.0))) +
                    (np.minimum( (np.sin(np.maximum( ((np.maximum( (np.floor(np.cos(data["v74"]))),  (data["v74"])) - data["v66"])),  (np.floor(np.cos(data["v10"])))))),  (((data["v3"] >= np.ceil(data["v50"])).astype(float)))) * 2.0) +
                    (-((((0.094340 * (data["v110"] * data["v24"])) + ((((np.minimum( (data["v110"]),  (np.minimum( (data["v110"]),  ((1.0/(1.0 + np.exp(- data["v40"]))))))) > (1.0/(1.0 + np.exp(- (-(data["v129"])))))).astype(float)) + np.tanh(data["v38"]))/2.0))/2.0))) +
                    np.maximum( (np.minimum( (np.cos(data["v10"])),  (((np.sinh(data["v74"]) > ((data["v50"] + np.round(np.cos(data["v74"])))/2.0)).astype(float))))),  (((((data["v3"] >= np.ceil(np.minimum( (data["v125"]),  (data["v10"])))).astype(float)) / 2.0) / 2.0))) +
                    (np.minimum( (((1.0/(1.0 + np.exp(- data["v12"]))) * data["v79"])),  ((np.floor(np.minimum( (0.058823),  (data["v56"]))) / 2.0))) * ((data["v66"] >= (((data["v113"] / 2.0) - data["v56"]) * 0.058823)).astype(float))) +
                    (((0.094340 + ((np.cos(data["v66"]) + np.minimum( (data["v56"]),  (data["v66"])))/2.0)) / 2.0) * ((data["v34"] + ((0.138462 >= data["v56"]).astype(float)))/2.0)) +
                    (-((0.058823 * np.maximum( ((data["v24"] + data["v21"])),  (np.minimum( ((((data["v21"] + 8.0)/2.0) * data["v100"])),  (((data["v75"] + np.maximum( (data["v21"]),  (data["v72"]))) - data["v56"])))))))) +
                    np.minimum( ((((data["v12"] <= np.cos(9.869604)).astype(float)) * ((np.floor(np.sin(data["v50"])) < np.sinh(np.minimum( (data["v21"]),  (((data["v12"] == 9.869604).astype(float)))))).astype(float)))),  (((data["v12"] + (3.71879))/2.0))) +
                    np.minimum( ((1.0/(1.0 + np.exp(- data["v75"])))),  (((data["v66"] < (np.maximum( ((data["v75"] + data["v50"])),  (np.abs((1.414214 * data["v50"])))) + -3.0)).astype(float)))) +
                    (np.cos(np.maximum( (np.maximum( (np.maximum( (data["v40"]),  (np.maximum( (np.floor(np.maximum( ((data["v14"] * data["v66"])),  ((data["v75"] * data["v66"]))))),  ((data["v72"] / 2.0)))))),  (data["v40"]))),  (1.584910))) / 2.0) +
                    np.minimum( (((data["v75"] <= np.maximum( ((-(data["v24"]))),  (np.minimum( (data["v55"]),  (data["v113"]))))).astype(float))),  (((0.094340 * 2.0) * (-(np.sin(np.minimum( ((data["v125"] * 2.0)),  ((data["v21"] * 2.0))))))))) +
                    ((5.428570 <= np.maximum( (np.maximum( (np.ceil((data["v88"] + data["v75"]))),  (((data["v41"] + data["v15"]) - data["v75"])))),  (np.abs(np.floor(((np.cos(data["v88"]) + np.sinh(data["v14"]))/2.0)))))).astype(float)) +
                    np.floor(np.sin(np.maximum( (((((1.0/(1.0 + np.exp(- (data["v41"] - ((data["v32"] < ((data["v99"] + data["v118"])/2.0)).astype(float)))))) + data["v41"])/2.0) - data["v18"])),  (((data["v3"] + data["v104"])/2.0))))) +
                    np.ceil(np.sin(np.minimum( (np.minimum( (data["v21"]),  (np.minimum( (np.minimum( (data["v21"]),  (data["v23"]))),  ((5.200000 - np.sinh(np.round(data["v117"])))))))),  ((((-(data["v120"])) > ((2.302585 > data["v23"]).astype(float))).astype(float)))))) +
                    np.minimum( (((-(((data["v113"] >= np.ceil(np.minimum( (np.cos(data["v90"])),  (np.cos(data["v49"]))))).astype(float)))) * np.abs(data["v83"]))),  (((np.cos(data["v6"]) - data["v6"]) * 0.058823))) +
                    (((data["v42"] > 1.584910).astype(float)) + ((-((np.ceil(np.sin(((data["v96"] + np.maximum( (20.750000),  (data["v96"])))/2.0))) * np.sinh(np.sinh(data["v60"]))))) * np.sinh(np.sinh(data["v126"])))) +
                    (((np.floor(np.floor(((np.floor(np.sin(np.sin(((data["v123"] + (((data["v72"] + data["v70"]) <= 5.200000).astype(float)))/2.0)))) * 2.0) * 2.0))) * 2.0) + ((data["v115"] >= 1.732051).astype(float)))/2.0) +
                    ((1.584910 <= ((data["v51"] + (data["v12"] * (data["v66"] * (-((data["v66"] * (-(np.cos((-(((data["v12"] >= np.tanh(data["v42"])).astype(float)))))))))))))/2.0)).astype(float)) +
                    (np.sin(np.floor(np.cos(np.minimum( (((data["v30"] + (((data["v50"] * data["v117"]) + data["v118"])/2.0))/2.0)),  ((data["v50"] * data["v118"])))))) / 2.0) +
                    ((np.sinh(np.maximum( (np.ceil((np.maximum( (data["v108"]),  (((data["v5"] == data["v83"]).astype(float)))) * 2.0))),  (2.409090))) < (1.570796 * np.floor(np.maximum( ((data["v6"] * 2.0)),  (data["v50"]))))).astype(float)) +
                    (((((data["v58"] < data["v62"]).astype(float)) < data["v50"]).astype(float)) * ((((1.414214 < np.minimum( (data["v37"]),  (data["v3"]))).astype(float)) + ((data["v29"] <= np.minimum( ((data["v55"] + data["v3"])),  (data["v43"]))).astype(float)))/2.0)) +
                    ((((data["v79"] <= ((-((1.0/(1.0 + np.exp(- (1.0/(1.0 + np.exp(- np.round(np.sin(np.maximum( (data["v30"]),  (((0.840000 >= data["v99"]).astype(float)))))))))))))) * 2.0)).astype(float)) + np.cos(np.maximum( (np.maximum( (1.584910),  (data["v80"]))),  (data["v6"]))))/2.0) +
                    ((data["v59"] < (data["v67"] - (1.0/(1.0 + np.exp(- (((np.maximum( (data["v107"]),  (((data["v40"] >= data["v59"]).astype(float)))) > data["v8"]).astype(float)) + (((1.584910 > data["v62"]).astype(float)) + data["v16"]))))))).astype(float)) +
                    (np.round(data["v79"]) * ((((np.abs(data["v45"]) - data["v14"]) + np.minimum( (np.cos(data["v71"])),  ((((data["v114"] + np.cos(data["v114"]))/2.0) * 2.0))))/2.0) * 0.058823)) +
                    ((math.sin(0.434294) >= (1.584910 + (data["v39"] * ((np.minimum( (np.cos(data["v62"])),  (np.minimum( (np.cos(data["v9"])),  (data["v78"])))) / 2.0) * ((0.434294 < np.sin(data["v86"])).astype(float)))))).astype(float)) +
                    np.floor(np.cos(np.sinh((np.minimum( (data["v113"]),  (np.ceil(data["v29"]))) * np.minimum( (data["v129"]),  (np.maximum( (data["v29"]),  (np.maximum( ((data["v107"] * (1.0/(1.0 + np.exp(- data["v66"]))))),  (((data["v66"] + 2.212120)/2.0))))))))))) +
                    (np.floor((((np.sin((np.abs(data["v90"]) + np.cos(np.maximum( (data["v21"]),  (0.720430))))) == ((data["v31"] <= (data["v21"] + math.sin(math.cos(0.720430)))).astype(float))).astype(float)) * 2.0)) * 2.0) +
                    np.round((np.minimum( (np.floor(np.tanh((data["v7"] * ((data["v45"] >= data["v97"]).astype(float)))))),  ((data["v7"] * data["v127"]))) - (data["v97"] * (np.minimum( (data["v119"]),  ((data["v19"] / 2.0))) / 2.0)))) +
                    (0.094340 * np.tanh(((data["v119"] - np.floor(data["v127"])) + (data["v66"] * ((np.cos(np.abs(data["v24"])) - np.floor(np.abs(data["v24"]))) - np.abs(data["v24"])))))) +
                    ((np.abs(np.floor(np.cos(((data["v5"] + ((data["v16"] <= data["v39"]).astype(float)))/2.0)))) * 2.0) * (np.abs(np.floor(np.cos(((data["v5"] + ((data["v16"] <= data["v97"]).astype(float)))/2.0)))) * 2.0)) +
                    ((((np.cos((data["v108"] - ((data["v99"] > 0.636620).astype(float)))) >= ((np.round(data["v7"]) != np.cos((data["v108"] - np.cos(data["v127"])))).astype(float))).astype(float)) * 2.0) * 2.0) +
                    (((3.141593 <= (((np.maximum( (data["v16"]),  (data["v130"])) * data["v72"]) + (data["v112"] + ((data["v130"] <= ((2.409090 <= data["v72"]).astype(float))).astype(float))))/2.0)).astype(float)) + (-(((data["v5"] > 2.409090).astype(float))))) +
                    np.floor(np.cos((data["v97"] * np.minimum( (np.maximum( (data["v56"]),  (np.minimum( (data["v21"]),  ((data["v21"] * data["v83"])))))),  ((1.0/(1.0 + np.exp(- data["v13"])))))))) +
                    np.sinh(((data["v47"] * 0.138462) * np.cos(((data["v116"] * (((((0.058823 != data["v56"]).astype(float)) * 2.0) != np.ceil((data["v56"] * ((data["v62"] > data["v38"]).astype(float))))).astype(float))) * 2.0)))) +
                    (-(np.ceil(((3.0 < (data["v38"] * (data["v1"] + (3.0 * np.floor(np.maximum( (np.maximum( (((data["v99"] + data["v52"])/2.0)),  (data["v62"]))),  (data["v10"]))))))).astype(float))))) +
                    ((2.718282 < (np.minimum( (data["v27"]),  (np.maximum( (data["v52"]),  (data["v89"])))) + ((data["v6"] <= (np.round(data["v49"]) + ((data["v40"] + data["v27"])/2.0))).astype(float)))).astype(float)) +
                    ((np.cos((((data["v93"] * 2.0) * (data["v30"] * data["v109"])) / 2.0)) == (((((((data["v24"] * data["v109"]) != data["v93"]).astype(float)) >= data["v50"]).astype(float)) >= data["v126"]).astype(float))).astype(float)) +
                    np.round((((data["v106"] - (data["v35"] + (1.0/(1.0 + np.exp(- ((((data["v71"] > data["v16"]).astype(float)) > data["v116"]).astype(float))))))) / 2.0) / 2.0)) +
                    np.floor(np.floor(np.sin(np.abs(((data["v131"] + ((data["v18"] + (((((((data["v37"] > ((data["v110"] + data["v99"])/2.0)).astype(float)) + data["v110"])/2.0) < data["v32"]).astype(float)) * 2.0)) - data["v76"]))/2.0))))) +
                    np.floor((np.floor((data["v124"] * ((data["v21"] > ((8.0 + np.round(np.minimum( (data["v100"]),  (np.sin(np.sin(data["v124"])))))) / 2.0)).astype(float)))) * np.sinh(data["v21"]))) +
                    np.floor(np.cos(((data["v7"] + np.minimum( (((np.minimum( (((data["v10"] != ((data["v10"] == data["v98"]).astype(float))).astype(float))),  (np.abs(data["v10"]))) * (-(data["v117"]))) * data["v117"])),  (data["v10"])))/2.0))) +
                    (np.floor(np.cos(np.minimum( (((data["v83"] != data["v97"]).astype(float))),  (((np.tanh((data["v85"] * data["v81"])) * np.sinh(np.cos(data["v120"]))) * data["v99"]))))) * 2.0) +
                    ((2.675680 < np.maximum( (np.maximum( ((data["v117"] * ((data["v57"] + np.floor((np.cos(data["v89"]) * 2.0)))/2.0))),  ((((data["v17"] * data["v126"]) + ((data["v117"] <= data["v85"]).astype(float)))/2.0)))),  (data["v99"]))).astype(float)) +
                    (np.floor(np.floor(np.sin(np.maximum( (data["v116"]),  (np.cos((data["v121"] + np.maximum( (data["v30"]),  (np.maximum( (data["v66"]),  (((data["v40"] <= np.maximum( (np.cos(data["v109"])),  (data["v74"]))).astype(float))))))))))))) * 2.0) +
                    (((((data["v120"] + np.abs(data["v115"]))/2.0) * (((data["v120"] + np.abs(np.floor(data["v130"])))/2.0) * ((data["v95"] > 2.212120).astype(float)))) >= np.ceil(((data["v120"] + np.abs(data["v115"]))/2.0))).astype(float)) +
                    np.sinh(((np.round(np.minimum( (((data["v26"] >= data["v45"]).astype(float))),  (((data["v61"] <= np.round(((5.428570 <= data["v129"]).astype(float)))).astype(float))))) > np.cos(np.minimum( (np.maximum( (data["v86"]),  (np.sinh(data["v78"])))),  (data["v38"])))).astype(float))) +
                    (abs(2.718282) * (data["v127"] * ((2.718282 <= ((((np.abs(np.minimum( (data["v106"]),  (data["v30"]))) + np.abs(data["v28"]))/2.0) + data["v10"])/2.0)).astype(float)))) +
                    (np.floor(np.sin(np.maximum( (np.abs((data["v83"] - np.abs(np.maximum( (data["v114"]),  (np.abs(data["v80"]))))))),  ((data["v114"] * (data["v104"] + ((data["v90"] <= data["v71"]).astype(float)))))))) * 2.0) +
                    ((((data["v119"] > 2.0).astype(float)) > ((data["v94"] >= np.minimum( (data["v62"]),  (np.minimum( (np.minimum( (np.cos(data["v45"])),  (np.sin(data["v111"])))),  (np.maximum( (data["v103"]),  (np.sinh(data["v125"])))))))).astype(float))).astype(float)) +
                    (((((data["v54"] >= max( (2.0),  (2.0)))) * np.minimum( (data["v69"]),  (np.ceil(((data["v17"] - ((data["v47"] > data["v53"]).astype(float))) - ((data["v40"] >= data["v17"]).astype(float))))))) / 2.0) / 2.0) +
                    (((0.693147 * 2.0) < ((data["v117"] + (data["v20"] * np.minimum( (((((((math.floor(0.693147) == np.floor(data["v109"])).astype(float)) >= np.cos(data["v130"])).astype(float)) >= np.sin(data["v2"])).astype(float))),  (data["v47"]))))/2.0)).astype(float)) +
                    (-((((5.200000 <= (np.maximum( (data["v72"]),  (data["v34"])) * np.maximum( (np.maximum( (data["v101"]),  (np.maximum( (np.round(data["v99"])),  (data["v50"]))))),  ((np.maximum( (data["v39"]),  (data["v7"])) * data["v50"]))))).astype(float)) * 2.0))) +
                    ((((np.maximum( (data["v59"]),  (data["v2"])) < 1.570796).astype(float)) * (np.abs((np.floor(data["v97"]) * 2.0)) * data["v72"])) * ((2.302585 < data["v117"]).astype(float))) +
                    np.sinh(((np.tanh(np.maximum( (np.maximum( (data["v38"]),  (np.maximum( (np.maximum( (data["v123"]),  (np.ceil(data["v70"])))),  ((data["v55"] + data["v129"])))))),  ((np.floor(data["v38"]) + data["v89"])))) == np.ceil(data["v70"])).astype(float))) +
                    (3.141593 * ((((1.0/(1.0 + np.exp(- np.round(data["v94"])))) <= data["v33"]).astype(float)) * np.floor(np.cos(((data["v17"] + ((data["v83"] <= data["v49"]).astype(float)))/2.0))))) +
                    np.sinh(((np.abs(data["v126"]) < np.minimum( (data["v31"]),  (((data["v93"] < ((data["v15"] > (data["v24"] + (3.0 - (-(np.floor(data["v101"])))))).astype(float))).astype(float))))).astype(float))) +
                    np.sinh(((((data["v80"] + data["v118"])/2.0) > np.round(((((np.round(((3.0 + data["v80"])/2.0)) + 1.570796)/2.0) + (data["v126"] + ((data["v80"] != 3.0).astype(float))))/2.0))).astype(float))) +
                    np.minimum( ((data["v130"] * data["v69"])),  (((3.0 <= np.minimum( (data["v37"]),  ((((data["v117"] - data["v50"]) * ((np.abs(data["v81"]) <= ((data["v130"] > data["v117"]).astype(float))).astype(float))) + data["v83"])))).astype(float)))) +
                    ((2.212120 <= (data["v2"] - ((((np.ceil(((data["v2"] - data["v74"]) - data["v35"])) <= (2.212120 - ((np.floor(data["v57"]) + data["v81"])/2.0))).astype(float)) <= data["v120"]).astype(float)))).astype(float)) +
                    np.minimum( (((data["v85"] > data["v16"]).astype(float))),  (np.minimum( (np.cos(data["v96"])),  (((np.minimum( (data["v16"]),  ((-(data["v97"])))) >= ((data["v75"] > (data["v105"] * ((data["v56"] < (-(data["v56"]))).astype(float)))).astype(float))).astype(float)))))) +
                    (np.minimum( (np.round(data["v131"])),  (np.ceil(data["v72"]))) * np.minimum( (((data["v98"] >= 1.584910).astype(float))),  (((((data["v98"] >= 1.584910).astype(float)) >= data["v131"]).astype(float))))) +
                    (np.cos(data["v119"]) * np.minimum( (((1.732051 <= ((data["v82"] + ((data["v88"] >= np.maximum( (data["v52"]),  (((data["v58"] < np.round(np.cos(np.sinh(data["v27"])))).astype(float))))).astype(float)))/2.0)).astype(float))),  (np.cos(data["v52"])))) +
                    ((np.minimum( (data["v71"]),  (((((data["v80"] + data["v34"])/2.0) < (3.141593 - data["v28"])).astype(float)))) >= np.maximum( (data["v128"]),  ((((3.141593 - data["v28"]) >= data["v53"]).astype(float))))).astype(float)) +
                    (8.0 * np.minimum( (((((np.floor(data["v51"]) != np.cos(np.sinh(data["v40"]))).astype(float)) <= ((((data["v42"] != np.sin(data["v99"])).astype(float)) <= np.cos(np.cos(data["v7"]))).astype(float))).astype(float))),  (np.cos(data["v79"])))))

    return Outputs(predictions)


if __name__ == "__main__":
    print('Start')
    print('Importing Data')

    clipmin = .05
    clipmax = .99
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    print('Munge Data')
    train, test = MungeData(train, test)
    ss = StandardScaler()
    features = train.columns[2:]
    train[features] = np.round(ss.fit_transform(train[features].values), 6)
    test[features] = np.round(ss.transform(test[features].values), 6)

    mother = pd.DataFrame({'GP1': GPIndividual1(train),
                           'GP2': GPIndividual2(train),
                           'GP3': GPIndividual3(train),
                           'GP4': GPIndividual4(train),
                           'GP5': GPIndividual5(train),
                           'GP6': GPIndividual6(train)})

    targets = train.target.values
    predictions = mother.mean(axis=1)
    predictions.fillna(.76, inplace=True)
    mother.fillna(.76, inplace=True)

    print('Raw Log Loss: ', log_loss(targets, np.clip(predictions.values,
                                                      clipmin,
                                                      clipmax)))

    print('GP1: ', log_loss(targets, np.clip(mother.GP1.values,
                                             clipmin,
                                             clipmax)))
    print('GP2: ', log_loss(targets, np.clip(mother.GP2.values,
                                             clipmin,
                                             clipmax)))
    print('GP3: ', log_loss(targets, np.clip(mother.GP3.values,
                                             clipmin,
                                             clipmax)))
    print('GP4: ', log_loss(targets, np.clip(mother.GP4.values,
                                             clipmin,
                                             clipmax)))
    print('GP5: ', log_loss(targets, np.clip(mother.GP5.values,
                                             clipmin,
                                             clipmax)))
    print('GP6: ', log_loss(targets, np.clip(mother.GP6.values,
                                             clipmin,
                                             clipmax)))

    mother = pd.DataFrame({'GP1': GPIndividual1(test),
                           'GP2': GPIndividual2(test),
                           'GP3': GPIndividual3(test),
                           'GP4': GPIndividual4(test),
                           'GP5': GPIndividual5(test),
                           'GP6': GPIndividual6(test)})

    predictions = mother.mean(axis=1)
    submission = pd.DataFrame({'ID': test.ID,
                               'PredictedProb': np.clip(predictions.values,
                                                        clipmin,
                                                        clipmax)})
    print(submission.head())
    submission.to_csv('gpsubmission.csv', index=False)
    print('Finished')
