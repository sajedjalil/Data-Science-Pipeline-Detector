import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def MungeData(train, test):
    le = LabelEncoder()
    train.Product_Info_2.fillna('Z0', inplace=True)
    test.Product_Info_2.fillna('Z0', inplace=True)
    train.insert(0, 'Product_Info_2_N', train.Product_Info_2.str[1:])
    train.insert(0, 'Product_Info_2_C', train.Product_Info_2.str[0])
    train.drop('Product_Info_2', inplace=True, axis=1)
    test.insert(0, 'Product_Info_2_N', test.Product_Info_2.str[1:])
    test.insert(0, 'Product_Info_2_C', test.Product_Info_2.str[0])
    test.drop('Product_Info_2', inplace=True, axis=1)

    le.fit(list(train.Product_Info_2_C)+list(test.Product_Info_2_C))
    train.Product_Info_2_C = le.transform(train.Product_Info_2_C)
    test.Product_Info_2_C = le.transform(test.Product_Info_2_C)

    trainids = train.Id
    testids = test.Id
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    responses = train.Response.values
    train.drop('Response', inplace=True, axis=1)

    train = train.astype(float)
    test = test.astype(float)

    train.insert(0,
                 'SumMedicalKeywords',
                 train[train.columns[train.columns.str.contains('Medical_Keyword')]]
                 .sum(axis=1, skipna=True))
    test.insert(0,
                'SumMedicalKeywords',
                test[test.columns[test.columns.str.contains('Medical_Keyword')]]
                .sum(axis=1, skipna=True))
    train.insert(0,
                 'SumEmploymentInfo',
                 train[train.columns[train.columns.str.contains('InsuredInfo')]]
                 .sum(axis=1, skipna=True))
    test.insert(0,
                'SumEmploymentInfo',
                test[test.columns[test.columns.str.contains('InsuredInfo')]]
                .sum(axis=1, skipna=True))
    train.insert(0,
                 'SumMedicalHistory',
                 train[train.columns[train.columns.str.contains('Medical_History')]]
                 .sum(axis=1, skipna=True))
    test.insert(0,
                'SumMedicalHistory',
                test[test.columns[test.columns.str.contains('Medical_History')]]
                .sum(axis=1, skipna=True))

    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    features = train.columns
    ss = StandardScaler()
    train[features] = ss.fit_transform(train[features].values)
    test[features] = ss.transform(test[features].values)
    train['Response'] = responses
    train.insert(0, 'Id', trainids)
    test.insert(0, 'Id', testids)
    return train, test

def Response1(data):
    p = ((np.sin(np.cosh(data["Medical_History_23"])) + (((1.0/(1.0 + np.exp(- (data["Product_Info_4"] + data["Medical_History_4"])))) * 2.0) - np.maximum( (data["BMI"]),  (((((data["SumMedicalKeywords"] + (data["Ins_Age"] - data["Medical_History_40"]))/2.0) + data["Medical_Keyword_3"])/2.0))))) +
        (((((((data["Insurance_History_5"] - (data["Insurance_History_2"] + data["Medical_History_30"])) - np.maximum( (data["InsuredInfo_5"]),  (data["Medical_History_5"]))) + data["Medical_History_13"])/2.0) / 2.0) + ((np.minimum( (data["InsuredInfo_6"]),  ((data["Medical_History_15"] + data["Medical_History_4"]))) + data["Medical_History_15"])/2.0))/2.0) +
        np.tanh(np.minimum( (data["Medical_History_20"]),  ((np.minimum( (np.ceil((np.round(((data["Product_Info_4"] + data["Medical_History_7"])/2.0)) * 2.0))),  ((((data["Medical_History_27"] - ((data["Medical_Keyword_38"] + data["Product_Info_4"])/2.0)) + data["Medical_History_11"])/2.0))) - data["InsuredInfo_7"])))) +
        np.tanh(((data["Medical_History_31"] + (((((data["Medical_Keyword_15"] + (((data["Family_Hist_2"] + np.maximum( (data["Medical_Keyword_41"]),  (data["Medical_History_3"])))/2.0) - data["Medical_History_18"]))/2.0) + ((data["Medical_Keyword_25"] + data["Medical_History_1"])/2.0)) + np.sin(np.ceil(data["Product_Info_2_N"])))/2.0))/2.0)) +
        np.minimum( (np.minimum( ((data["InsuredInfo_2"] * data["Medical_History_35"])),  (np.floor(np.cos(np.minimum( (data["Medical_History_35"]),  (np.sinh((data["Medical_History_15"] * 5.4285697937))))))))),  (np.cos(np.maximum( (data["Ins_Age"]),  ((data["Medical_History_28"] * data["Employment_Info_3"])))))) +
        np.minimum( (np.ceil((data["Medical_Keyword_22"] + np.ceil((data["Medical_History_24"] + np.maximum( (0.0588234998),  ((np.floor(data["Product_Info_2_C"]) + data["Medical_Keyword_33"])))))))),  (np.minimum( (np.cos(np.ceil(data["Medical_History_11"]))),  (np.maximum( (data["SumMedicalKeywords"]),  (data["Medical_History_39"])))))) +
        (0.0588234998 * ((data["Medical_History_6"] + (((-(data["InsuredInfo_1"])) - (data["Product_Info_4"] + data["Insurance_History_8"])) + data["Medical_History_17"])) + (((data["InsuredInfo_6"] + data["Family_Hist_4"]) - data["Medical_History_40"]) + data["Medical_History_33"]))) +
        (-((((data["SumMedicalKeywords"] - np.cos(data["Medical_History_11"])) / 2.0) * (-(np.minimum( (np.cos((1.0/(1.0 + np.exp(- (data["SumMedicalKeywords"] - data["Medical_Keyword_3"])))))),  (np.round(((((data["Medical_Keyword_3"] + data["Medical_History_30"])/2.0) + data["Medical_History_5"])/2.0))))))))) +
        ((np.sinh(np.minimum( (0.1384620070),  (np.cos(data["Ins_Age"])))) + np.minimum( (np.tanh(np.minimum( ((data["Medical_Keyword_9"] * data["Medical_Keyword_37"])),  ((data["Medical_History_19"] * data["InsuredInfo_2"]))))),  (((data["Employment_Info_1"] + (1.0/(1.0 + np.exp(- data["Medical_History_33"]))))/2.0))))/2.0) +
        (((((((data["Medical_Keyword_34"] + data["Medical_Keyword_12"])/2.0) - (data["Product_Info_2_C"] + np.maximum( (0.0943396017),  (data["Medical_History_1"])))) + data["InsuredInfo_7"])/2.0) + (data["Medical_History_13"] * ((data["Medical_History_28"] + data["Medical_History_23"])/2.0))) * 0.0943396017))
    return p+4.5

def Response2(data):
    p = ((np.cos(((data["Medical_Keyword_3"] + (-(data["BMI"])))/2.0)) - ((data["BMI"] + ((((data["SumMedicalKeywords"] + data["InsuredInfo_5"])/2.0) - (np.cos(data["Medical_History_40"]) + np.minimum( (data["Product_Info_4"]),  (data["Medical_History_23"])))) - data["Medical_History_4"]))/2.0)) +
        np.round(np.tanh((((data["Medical_History_15"] - ((data["Insurance_History_2"] + data["InsuredInfo_7"])/2.0)) + np.ceil(np.minimum( (data["Product_Info_4"]),  (np.ceil(np.floor((data["Medical_History_15"] + np.round((data["Medical_History_27"] - np.tanh(data["Medical_History_15"]))))))))))/2.0))) +
        np.tanh((data["Medical_History_20"] + (((-(((((data["Medical_History_18"] + data["Medical_Keyword_3"])/2.0) + np.maximum( (data["Medical_History_5"]),  (data["Medical_History_30"])))/2.0))) + ((((data["InsuredInfo_6"] + data["SumMedicalKeywords"])/2.0) + ((data["Family_Hist_4"] + data["Medical_History_13"])/2.0))/2.0))/2.0))) +
        np.tanh(((((np.minimum( (data["Medical_History_11"]),  ((data["Medical_History_31"] + (data["Medical_History_28"] * data["SumEmploymentInfo"])))) + ((data["Insurance_History_5"] + (data["Medical_Keyword_41"] - data["Ins_Age"]))/2.0))/2.0) + (((data["Medical_History_40"] + data["InsuredInfo_5"])/2.0) - data["Medical_Keyword_38"]))/2.0)) +
        ((data["Medical_History_35"] * data["InsuredInfo_2"]) + np.minimum( (np.cos(data["Ins_Age"])),  ((((((((data["Medical_History_3"] + np.ceil(data["Medical_Keyword_15"]))/2.0) + np.minimum( (data["Medical_History_7"]),  (data["Medical_History_1"])))/2.0) / 2.0) + np.tanh(np.ceil(data["Medical_History_24"])))/2.0)))) +
        np.sin(np.minimum( (data["Medical_History_33"]),  (((data["Medical_Keyword_25"] + (((((np.floor(data["Ins_Age"]) * np.floor(((data["BMI"] * 2.0) + data["Medical_History_5"]))) + ((data["Medical_Keyword_22"] + data["Medical_History_17"])/2.0))/2.0) + np.cos(data["BMI"]))/2.0))/2.0)))) +
        np.sin((np.minimum( (data["Medical_History_11"]),  (np.minimum( (np.floor(np.maximum( (data["SumMedicalKeywords"]),  (np.maximum( (data["Medical_History_39"]),  (data["InsuredInfo_6"])))))),  ((0.0943396017 * (((data["InsuredInfo_6"] + data["BMI"]) + (data["Medical_History_1"] - data["InsuredInfo_1"]))/2.0)))))) * 2.0)) +
        (np.sin(np.maximum( ((data["Medical_History_5"] + np.sin((np.ceil(data["Medical_History_15"]) * 2.0)))),  (np.maximum( (np.sinh(data["Medical_Keyword_33"])),  (np.minimum( (0.1384620070),  (((data["Ins_Age"] + np.ceil(data["Product_Info_2_N"]))/2.0)))))))) / 2.0) +
        (np.minimum( (np.cos((data["Wt"] + np.round(data["Medical_History_4"])))),  (np.ceil(((1.0/(1.0 + np.exp(- data["Medical_Keyword_3"]))) - np.cosh(np.minimum( (data["Product_Info_4"]),  (np.sinh((data["InsuredInfo_2"] * data["Medical_Keyword_9"]))))))))) / 2.0) +
        (0.0588234998 * ((((np.ceil(data["Employment_Info_2"]) - data["Product_Info_4"]) + (np.maximum( (data["Family_Hist_2"]),  (data["Medical_History_21"])) + data["Family_Hist_5"])) + (data["BMI"] - data["Insurance_History_8"])) + np.minimum( (data["Medical_History_6"]),  (data["Product_Info_2_N"])))))
    return p+4.5

def Response3(data):
    p = (((data["Medical_History_4"] + (np.cos(data["BMI"]) - ((data["Medical_Keyword_3"] + (data["BMI"] - (1.6304299831 + np.minimum( (data["Product_Info_4"]),  (((data["Medical_History_40"] + data["Medical_History_13"])/2.0)))))) - np.ceil(data["Medical_History_23"]))))/2.0) +
        ((((data["Medical_History_15"] + ((data["InsuredInfo_6"] + ((data["Medical_History_20"] - data["Insurance_History_2"]) - np.maximum( (data["InsuredInfo_5"]),  (np.maximum( (data["Medical_History_30"]),  (data["Medical_History_5"]))))))/2.0))/2.0) + np.minimum( (np.ceil(data["Product_Info_4"])),  ((data["InsuredInfo_7"] * data["Medical_History_18"]))))/2.0) +
        (((((data["Family_Hist_2"] + (data["Medical_History_27"] - data["Medical_Keyword_38"]))/2.0) + ((((data["BMI"] * data["Medical_Keyword_3"]) - data["Ins_Age"]) + (data["Insurance_History_5"] + (data["Medical_Keyword_15"] + (data["BMI"] * data["Ins_Age"]))))/2.0))/2.0) / 2.0) +
        (np.tanh(data["Medical_History_11"]) + np.floor((np.sin((0.1384620070 * np.minimum( (np.round(((604) * np.minimum( (data["Medical_History_15"]),  (np.maximum( (data["Employment_Info_2"]),  ((-(data["Medical_History_28"]))))))))),  (0.1384620070)))) / 2.0))) +
        np.tanh((data["Medical_History_31"] + ((np.minimum( (np.ceil(data["Medical_History_24"])),  ((np.minimum( (data["Medical_History_4"]),  (data["Wt"])) * (-(np.cos(data["Ins_Age"])))))) + np.round(((data["Medical_Keyword_41"] + np.minimum( (data["Medical_History_7"]),  (data["Product_Info_4"])))/2.0)))/2.0))) +
        ((data["Medical_History_35"] * data["InsuredInfo_2"]) + np.sin(np.minimum( (data["Medical_History_17"]),  ((((((np.maximum( (data["Medical_History_3"]),  (data["Medical_History_1"])) + ((data["Medical_Keyword_25"] + data["Medical_History_33"])/2.0))/2.0) + np.ceil(np.minimum( (data["Medical_History_11"]),  (data["Product_Info_2_N"]))))/2.0) / 2.0))))) +
        np.minimum( ((0.1384620070 * ((data["Medical_History_6"] + (data["Family_Hist_3"] + ((((data["Medical_History_15"] - data["Medical_History_19"]) - data["Medical_Keyword_9"]) - (data["Medical_History_40"] - data["Family_Hist_4"])) + data["Medical_Keyword_33"])))/2.0))),  (np.cos(data["Ins_Age"]))) +
        np.minimum( ((np.floor(np.maximum( (data["Product_Info_2_C"]),  ((data["SumMedicalKeywords"] * np.round((((((data["Medical_Keyword_38"] + data["InsuredInfo_2"])/2.0) / 2.0) + (data["Medical_Keyword_3"] / 2.0))/2.0)))))) / 2.0)),  (np.sin(np.maximum( (data["BMI"]),  ((1.0/(1.0 + np.exp(- data["Medical_Keyword_3"])))))))) +
        (0.6029409766 - (1.0/(1.0 + np.exp(- ((((data["Product_Info_2_C"] + np.maximum( ((np.maximum( (data["Medical_Keyword_23"]),  (data["Medical_Keyword_37"])) + (data["Insurance_History_8"] + data["Medical_History_18"]))),  ((data["Medical_History_30"] + data["Medical_History_5"]))))/2.0) + np.maximum( (data["Medical_History_23"]),  (data["InsuredInfo_1"])))/2.0))))) +
        (((np.minimum( (data["Medical_History_32"]),  ((data["Employment_Info_1"] + np.maximum( (data["InsuredInfo_6"]),  (data["Medical_History_39"]))))) / 2.0) + np.minimum( (0.3183098733),  (((np.maximum( (data["Medical_History_21"]),  (data["Medical_History_1"])) + (1.0/(1.0 + np.exp(- np.minimum( (data["SumEmploymentInfo"]),  (data["InsuredInfo_6"]))))))/2.0))))/2.0))
    return p+4.5


if __name__ == "__main__":
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    train, test = MungeData(train, test)
    p1 = Response1(train)
    p2 = Response2(train)
    p3 = Response3(train)
    t1 = Response1(test)
    t2 = Response2(test)
    t3 = Response3(test)
    trainResponse = 0.32018*(p1-p3)+0.35505*(p2-p3)+p3
    testResponse = 0.32018*(t1-t3)+0.35505*(t2-t3)+t3
    previousCut = -1000
    cuts = np.array([2.781097, 3.846915, 4.294624, 4.994817, 5.540523, 6.221271, 6.574580])
    for i, cut in enumerate(cuts):
        trainResponse[(trainResponse > previousCut) & (trainResponse<cut)]=i+1
        testResponse[(testResponse > previousCut) & (testResponse<cut)]=i+1
        previousCut = cut
    trainResponse[(trainResponse > cuts[-1])] = 8
    testResponse[(testResponse > cuts[-1])] = 8
    gppythontrain = pd.DataFrame({"Id":train.Id.astype(int),"Response":train.Response.astype(int),"Prediction":trainResponse.astype(int)})
    gppythontrain.to_csv('gppythontrain.csv',index=False)
    gppythontest = pd.DataFrame({"Id":test.Id.astype(int),"Response":testResponse.astype(int)})
    gppythontest.to_csv('gppythontest.csv',index=False)

