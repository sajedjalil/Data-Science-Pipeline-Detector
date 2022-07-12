import numpy as np
import pandas as pd
import math
from sklearn.metrics import roc_auc_score


def Outputs(data):
    return 1./(1.+np.exp(-data))


def LeaveOneOut(data1, data2, columnName, useLOO=False):
    grpOutcomes = data1.groupby(columnName)['outcome'].mean().reset_index()
    grpCount = data1.groupby(columnName)['outcome'].count().reset_index()
    grpOutcomes['cnt'] = grpCount.outcome
    if(useLOO):
        grpOutcomes = grpOutcomes[grpOutcomes.cnt > 1]

    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['outcome'].values
    x = pd.merge(data2[[columnName, 'outcome']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['outcome']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
        x *= np.random.uniform(0.95, 1.05, x.shape[0])
    return x.fillna(x.mean())


def intersect(a, b):
    return list(set(a) & set(b))


def get_features(train, test):
    trainval = list(train.columns.values)
    testval = list(test.columns.values)
    output = intersect(trainval, testval)
    output.remove('people_id')
    output.remove('activity_id')
    return sorted(output)


def read_test_train(train, test, people):
    print("Process tables...")
    for table in [train, test]:
        table['year'] = table['date'].dt.year
        table['month'] = table['date'].dt.month
        table['day'] = table['date'].dt.day
        table['weekday'] = table['date'].dt.weekday
        table['weekend'] = \
            ((table.weekday == 0) | (table.weekday == 6)).astype(int)
        table['activity_category'] = \
            table['activity_category'].str.lstrip('type ').astype(np.int32)
        for i in range(1, 11):
            table['char_' + str(i)].fillna('type -999', inplace=True)
            table['char_' + str(i)] = \
                table['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    people['year'] = people['date'].dt.year
    people['month'] = people['date'].dt.month
    people['day'] = people['date'].dt.day
    people['weekday'] = people['date'].dt.weekday
    people['weekend'] = \
        ((people.weekday == 0) | (people.weekday == 6)).astype(int)
    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)
    for i in range(1, 10):
        people['char_' + str(i)] = \
            people['char_' + str(i)].str.lstrip('type ').astype(np.int32)
    for i in range(10, 38):
        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)

    print("Merge...")
    train = pd.merge(train,
                     people,
                     how='left',
                     suffixes=('_train', '_people'),
                     on='people_id', left_index=True)
    train.fillna(-999, inplace=True)
    test = pd.merge(test,
                    people,
                    how='left',
                    suffixes=('_train', '_people'),
                    on='people_id', left_index=True)
    test.fillna(-999, inplace=True)
    train['business_days_delta']  = \
      np.log10(1+np.busday_count(train['date_people'].values.astype('<M8[D]'), train['date_train'].values.astype('<M8[D]'))).astype(int)
    test['business_days_delta']  = \
    np.log10(1+np.busday_count(test['date_people'].values.astype('<M8[D]'), test['date_train'].values.astype('<M8[D]'))).astype(int)
    trainoutcomes = train.outcome
    trainoutcomes = train.outcome
    train.drop('outcome', inplace=True, axis=1)
    features = get_features(train, test)
    return train, trainoutcomes, test, features


def MungeData(train, test, people):
    train, trainoutcomes, test, features = read_test_train(train, test, people)
    return train, trainoutcomes, test, features


def GPIndividual(data):
    predictions = (np.tanh(((((31.006277 + 31.006277)/2.0) * np.minimum( ((data["weekend_people"] + data["group_1"])),  (np.minimum( (2.828570),  (((data["group_1"] + ((data["group_1"] + (1.222220 - (23.500000 * (0.318310 - ((data["group_1"] + np.minimum( (0.020833),  ((data["group_1"] + np.minimum( (0.872727),  ((((-1.0 + data["char_6_people"])/2.0) * 2.0)))))))/2.0)))))/2.0))/2.0)))))) * 2.0)) +
                   np.tanh(((((data["group_1"] * 2.0) - ((31.006277 * (np.tanh((0.970149 + np.tanh(((2.0 >= (31.006277 * (np.tanh((0.970149 + np.tanh(np.maximum( (0.575758),  (0.970149))))) - (data["group_1"] * 2.0)))).astype(float))))) - (data["group_1"] * 2.0))) * 2.0)) * 2.0) / 2.0)) +
                   np.tanh(((np.cos(((data["char_2_people"] - (-(((0.636620 <= np.minimum( (2.718282),  ((-(((data["char_7_people"] - 1.666670) + data["char_2_people"])))))).astype(float))))) * 2.0)) * (31.006277 - 0.367879)) * 2.0)) +
                   np.tanh((19.0 * ((data["char_2_people"] - data["year_train"]) * 2.0))) +
                   np.tanh((23.500000 * (((data["char_18"] * 2.0) - ((0.529412 >= data["char_9_people"]).astype(float))) * 2.0))) +
                   np.tanh((31.006277 * ((data["char_7_people"] * 2.0) - (0.529412 + (((data["char_7_people"] >= (31.006277 * ((data["char_14"] * 2.0) - ((0.529412 > ((((data["char_8_people"] > (np.cos(data["char_7_people"]) / 2.0)).astype(float)) > (data["char_7_people"] / 2.0)).astype(float))).astype(float))))).astype(float)) * 2.0))))) +
                   np.tanh(((19.0 * 2.0) * (((0.020833 + ((np.cos(data["char_21"]) * 2.0) - (((data["char_1_people"] <= np.cos(np.cos(2.718282))).astype(float)) + ((data["char_6_people"] <= np.cos((((data["char_34"] * 2.0) * 2.0) - (((data["char_1_people"] <= np.cos(np.cos(2.718282))).astype(float)) + np.sin(np.cos(np.cos(data["char_34"]))))))).astype(float)))))/2.0) * 2.0))) +
                   np.tanh((23.500000 * (0.529412 - ((0.0 >= np.tanh(np.minimum( ((23.500000 * (data["char_2_people"] - ((data["business_days_delta"] >= data["group_1"]).astype(float))))),  (((0.529412 + (data["group_1"] - ((data["business_days_delta"] >= ((((-3.0 >= np.cos(data["group_1"])).astype(float)) >= 23.500000).astype(float))).astype(float))))/2.0))))).astype(float))))) +
                   np.tanh((23.500000 * (data["char_8_people"] - np.maximum( ((((data["char_8_people"] * 2.0) <= ((data["char_8_people"] > data["char_7_people"]).astype(float))).astype(float))),  (((np.tanh((((data["char_7_people"] * 2.0) >= 0.0).astype(float))) + ((0.811111 <= ((0.367879 > data["char_7_people"]).astype(float))).astype(float)))/2.0)))))) +
                   np.tanh(((14.15272998809814453) * (1.0 - (((data["group_1"] < ((data["char_23"] + data["char_14"])/2.0)).astype(float)) * 2.0)))) +
                   np.tanh((9.869604 * np.minimum( ((data["char_2_people"] - np.cos(((np.cos(-2.0) + ((data["char_2_people"] * np.sin(data["char_6_people"])) * 23.500000))/2.0)))),  (0.636620)))) +
                   np.tanh((((((((((data["group_1"] - np.sin(data["char_4_people"])) * 2.0) - ((np.sin(data["group_1"]) < (((0.575758 > data["group_1"]).astype(float)) * np.sin(((data["group_1"] <= ((data["char_23"] > np.maximum( (data["group_1"]),  (((data["group_1"] - np.sin(data["char_4_people"])) * 2.0)))).astype(float))).astype(float))))).astype(float))) * 2.0) * 2.0) * 2.0) - np.sin(((data["group_1"] >= 0.0).astype(float)))) * 2.0) * 2.0)) +
                   np.tanh(((2.357140 * ((2.828570 * 4.166670) * (data["char_36"] - np.cos(np.maximum( ((2.428570 * data["char_14"])),  ((31.006277 * (data["char_36"] - (((((data["char_26"] + np.cos(np.cos(data["char_21"]))) / 2.0) > data["char_34"]).astype(float)) / 2.0))))))))) * 2.0)) +
                   np.tanh((data["weekend_train"] + (((0.318310 + ((23.500000 * data["group_1"]) * (np.maximum( (1.222220),  ((float(-3.0 >= 0.0)))) / 2.0)))/2.0) + (-((3.0 * (9.869604 * (np.tanh(data["char_20"]) - (3.0 * (data["group_1"] - data["char_8_people"])))))))))) +
                   np.tanh(((((data["char_16"] > np.maximum( (data["weekday_train"]),  ((np.maximum( (((data["char_10_people"] >= 0.0).astype(float))),  ((float(1.0 > -1.0)))) / 2.0)))).astype(float)) - (((1.0 <= ((1.414214 >= ((0.575758 + ((4.166670 * data["char_6_people"]) - np.tanh(np.cos((data["activity_category"] + 1.414214)))))/2.0)).astype(float))).astype(float)) * 2.0)) * 2.0)) +
                   np.tanh((np.maximum( (-3.0),  ((10.63567352294921875))) * np.minimum( (1.0),  ((np.maximum( (data["char_13"]),  (9.869604)) * (data["char_4_people"] + np.tanh((data["char_10_people"] - ((data["char_13"] < data["char_15"]).astype(float)))))))))) +
                   np.tanh((((((np.maximum( (data["char_6_people"]),  ((((data["char_12"] > data["char_2_people"]).astype(float)) + -2.0))) - ((((data["char_12"] > data["char_2_people"]).astype(float)) * 2.0) / 2.0)) * 2.0) * 3.141593) + -1.0) * 31.006277)) +
                   np.tanh((np.tanh(((data["char_2_people"] * (np.minimum( (data["char_2_people"]),  ((((float((3.141593 - 0.0) >= 0.0)) < (((((0.0 + data["char_6_people"])/2.0) * 2.0) * 2.0) * 2.0)).astype(float)))) * 2.0)) - data["char_20"])) * (19.0 - (((((3.141593 - data["year_people"]) >= 0.0).astype(float)) + np.minimum( (0.575758),  (19.0)))/2.0)))) +
                   np.tanh(((data["year_people"] - (10.20737457275390625)) - ((((((math.cos((23.500000 / 2.0)) >= data["char_7_people"]).astype(float)) * (((data["char_20"] >= data["char_7_people"]).astype(float)) * 1.531250)) <= data["char_7_people"]).astype(float)) * (-(np.maximum( (19.0),  (2.718282))))))) +
                   np.tanh(((((((data["char_33"] * np.minimum( (np.cos(9.869604)),  ((np.tanh((data["group_1"] + (-2.0 / 2.0))) * ((1.531250 + 31.006277) / 2.0))))) * 2.0) * 2.0) / 2.0) + (data["group_1"] * np.maximum( ((19.0 * (np.minimum( (19.0),  (data["group_1"])) - data["char_32"]))),  (-2.0)))) * 2.0)) +
                   np.tanh(((data["weekday_train"] - np.tanh(((data["char_2_people"] <= ((data["char_7_people"] < (data["char_6_people"] * data["char_2_people"])).astype(float))).astype(float)))) * (((data["char_2_people"] - ((1.414214 <= (((2.828570 - ((data["char_6_people"] * data["char_2_people"]) - (data["char_2_people"] * 19.0))) + 0.0)/2.0)).astype(float))) + (19.0 * 2.0))/2.0))) +
                   np.tanh((31.006277 * np.minimum( ((((data["group_1"] * 19.0) * 2.0) * 2.0)),  (np.minimum( (((data["group_1"] - np.cos(((data["char_11"] >= np.sin(((data["group_1"] < ((data["group_1"] < np.maximum( (((data["char_27"] * 2.0) / 2.0)),  (((np.minimum( (data["char_27"]),  (1.414214)) + (data["month_train"] + data["char_32"]))/2.0)))).astype(float))).astype(float)))).astype(float)))) * 2.0)),  ((data["char_28"] * 2.0))))))) +
                   np.tanh(((((data["group_1"] - np.cos(np.cos(data["char_19"]))) * 23.500000) - ((((data["group_1"] >= 9.869604).astype(float)) + (np.cos(((data["group_1"] * 23.500000) - ((((23.500000 >= data["group_1"]).astype(float)) + 2.718282) * 2.0))) * 2.0)) * 2.0)) * 2.0)) +
                   np.tanh((((0.0) + np.cos((-1.0 - ((data["group_1"] < (np.cos(np.sin(0.636620)) * ((1.261900 >= ((data["group_1"] - ((data["char_32"] + ((np.maximum( (data["char_1_people"]),  (data["char_30"])) > np.minimum( (data["month_train"]),  (data["group_1"]))).astype(float)))/2.0)) * (6.0))).astype(float)))).astype(float))))) * 31.006277)) +
                   np.tanh(((-((((np.maximum( (3.141593),  (data["char_15"])) * ((0.561404 - np.tanh(data["char_1_people"])) * 2.0)) * ((((((data["month_people"] + 1.531250) + (float(0.970149 >= 1.0)))/2.0) <= ((3.141593 > ((((data["char_23"] >= data["char_6_people"]).astype(float)) >= 0.0).astype(float))).astype(float))).astype(float)) * 2.0)) * 2.0))) - ((0.367879 > data["char_6_people"]).astype(float)))) +
                   np.tanh((((np.cos(((((np.maximum( (data["group_1"]),  ((((np.cos(((data["group_1"] * (-((1.732051 * 3.0)))) * 2.0)) * 2.0) <= (-(data["char_17"]))).astype(float)))) <= data["char_4_people"]).astype(float)) - 3.0) * 2.0)) * 2.0) * 2.0) * 2.0)) +
                   np.tanh((data["month_people"] - (data["char_20"] * (3.141593 - ((np.sin(np.cos(((-((8.0))) / 2.0))) + (9.869604 * np.cos((-((-((data["month_people"] * (data["month_people"] - ((-(19.0)) / 2.0))))))))))/2.0))))) +
                   np.tanh(((2.777780 * ((((-(data["char_24"])) - (-(np.cos(((((1.570796 + data["weekday_people"])/2.0) >= 0.0).astype(float)))))) >= 0.0).astype(float))) * (0.020833 - ((0.575758 < (((data["char_28"] <= ((np.sin(data["char_11"]) >= data["weekday_people"]).astype(float))).astype(float)) * np.minimum( (1.414214),  ((2.357140 * data["weekday_train"]))))).astype(float))))) +
                   np.tanh((2.0 * np.sin(np.tanh(((((((data["month_people"] - ((data["char_3_people"] < (((data["char_26"] >= 0.0).astype(float)) / 2.0)).astype(float))) >= 0.0).astype(float)) - ((((((data["char_26"] - ((data["char_3_people"] <= data["char_26"]).astype(float))) - data["char_22"]) >= 0.0).astype(float)) <= data["char_26"]).astype(float))) - data["char_26"]) - ((data["char_3_people"] < (((data["char_3_people"] < (((data["char_3_people"] >= 0.0).astype(float)) / 2.0)).astype(float)) / 2.0)).astype(float))))))) +
                   np.tanh((3.141593 * (3.141593 * np.sin((((((data["char_1_people"] * 2.828570) * 2.0) + np.maximum( (-1.0),  (data["char_9_people"]))) + np.cos(2.357140)) * ((1.261900 - ((data["char_9_people"] > data["char_13"]).astype(float))) * 2.0)))))) +
                   np.tanh((((data["activity_category"] - np.maximum( (((data["activity_category"] < ((data["activity_category"] < np.sin((-(np.cos(data["char_31"]))))).astype(float))).astype(float))),  (np.minimum( (np.sin(((data["year_train"] < np.sin(np.maximum( (np.tanh(0.529412)),  (0.318310)))).astype(float)))),  (((data["activity_category"] < np.sin(data["char_31"])).astype(float))))))) * 2.0) - ((data["year_train"] < np.sin(np.maximum( (np.tanh(0.529412)),  (0.0)))).astype(float)))) +
                   np.tanh((2.782610 * (data["char_2_people"] + (((np.sin((np.sin(((2.782610 + data["char_2_people"]) * 2.0)) * np.maximum( (31.006277),  (-2.0)))) * 2.0) * ((1.666670 + ((((np.sin(np.maximum( (2.357140),  (data["char_2_people"]))) >= 0.0).astype(float)) >= 0.0).astype(float)))/2.0)) * 2.0)))) +
                   np.tanh(((data["char_34"] + (12.34734821319580078)) * (((1.531250 * 1.570796) * (data["group_1"] - (np.cos(data["group_1"]) - ((data["group_1"] <= 0.367879).astype(float))))) - np.cos(((data["group_1"] >= 0.0).astype(float)))))) +
                   np.tanh((19.0 * ((-(((np.minimum( ((-(np.minimum( (data["char_16"]),  ((np.minimum( (data["char_7_people"]),  (2.718282)) - (data["char_26"] * 2.0))))))),  (((data["char_37"] >= data["char_7_people"]).astype(float)))) * 2.0) - data["char_16"]))) - (((0.367879 - np.maximum( (((data["char_6_people"] - (np.cos(((0.872727 >= (data["char_10_people"] / 2.0)).astype(float))) * 2.0)) / 2.0)),  (data["char_7_people"]))) >= 0.0).astype(float))))) +
                   np.tanh(((((data["char_16"] * 2.0) * 2.0) * (-(((data["char_5_people"] < ((((-((float(2.828570 <= 0.529412)))) < ((((((0.636620 + data["char_23"]) / 2.0) >= np.sin(data["char_16"])).astype(float)) > ((data["year_train"] >= np.sin(((data["char_16"] * 2.0) * 2.0))).astype(float))).astype(float))).astype(float)) / 2.0)).astype(float))))) * 2.0)) +
                   np.tanh(((np.maximum( (19.0),  (data["group_1"])) * ((np.maximum( (19.0),  (np.maximum( (19.0),  (2.357140)))) * ((np.maximum( (3.0),  (data["char_5_people"])) * (data["group_1"] - data["char_6_people"])) - data["char_6_people"])) - data["char_23"])) * np.minimum( (2.782610),  (((((np.maximum( (19.0),  (np.cos(1.414214))) >= 0.0).astype(float)) + ((2.828570 < data["group_1"]).astype(float)))/2.0))))) +
                   np.tanh((((data["group_1"] - ((data["char_3_people"] + np.sin(((data["char_22"] + np.sin((data["char_22"] / 2.0)))/2.0)))/2.0)) * 19.0) * 19.0)) +
                   np.tanh(((-((23.500000 - ((data["char_10_people"] >= data["month_people"]).astype(float))))) * ((data["char_8_people"] <= (data["char_23"] * ((((((data["char_10_people"] >= data["month_people"]).astype(float)) >= data["char_10_people"]).astype(float)) <= (data["month_people"] + data["char_8_people"])).astype(float)))).astype(float)))) +
                   np.tanh(((3.0 * np.minimum( ((np.sin(np.sin((((data["char_2_people"] - np.cos(0.970149)) * 2.0) * (23.500000 + 3.141593)))) * 2.0)),  ((data["char_19"] + (np.maximum( (data["char_2_people"]),  ((((data["char_2_people"] > np.tanh(np.tanh(data["char_18"]))).astype(float)) * 2.0))) - ((4.0) * data["char_18"])))))) * 2.0)) +
                   np.tanh(((14.18952274322509766) * (data["group_1"] - np.minimum( ((9.869604 * 2.0)),  (((data["char_16"] <= ((np.tanh(np.tanh(data["group_1"])) <= np.minimum( (np.maximum( (0.529412),  (((data["char_21"] >= (((0.970149 <= data["group_1"]).astype(float)) + 0.561404)).astype(float))))),  (data["char_9_people"]))).astype(float))).astype(float))))))) +
                   np.tanh((((np.minimum( (2.357140),  (((10.70516395568847656) * ((((data["group_1"] * 1.261900) - 1.414214) + (((data["group_1"] * 1.261900) - ((data["char_32"] <= ((float(1.261900 >= 1.261900)) * data["group_1"])).astype(float))) * 2.0))/2.0)))) / 2.0) * 2.0) * 2.0)) +
                   np.tanh((19.0 * ((19.0 * (data["group_1"] - ((1.414214 + (-(((data["char_26"] > data["char_20"]).astype(float)))))/2.0))) - (((19.0 * (data["group_1"] - ((data["char_20"] + ((0.575758 + (-(((np.sin((0.318310 / 2.0)) > (0.970149 + 1.732051)).astype(float)))))/2.0))/2.0))) + (-(((-1.0 > data["char_26"]).astype(float)))))/2.0)))) +
                   np.tanh(((np.minimum( (data["char_2_people"]),  ((data["char_2_people"] - ((data["char_17"] < np.minimum( (data["char_4_people"]),  (np.tanh(data["char_15"])))).astype(float))))) - ((data["char_2_people"] <= ((data["char_2_people"] < np.tanh((np.minimum( (np.minimum( (np.minimum( (np.minimum( ((13.72905063629150391)),  (data["char_17"]))),  (data["char_4_people"]))),  (np.tanh(data["char_4_people"])))),  (data["char_18"])) * 2.0))).astype(float))).astype(float))) * 19.0)) +
                   np.tanh((2.777780 * (-(((data["business_days_delta"] < (((((data["char_30"] + data["char_30"]) >= ((((0.984375 > data["business_days_delta"]).astype(float)) + data["char_28"])/2.0)).astype(float)) <= np.minimum( (3.0),  ((4.166670 * (0.872727 - (data["char_35"] * (((float(12.73873233795166016) >= 0.0)) - ((data["char_32"] < np.tanh(data["char_10_people"])).astype(float))))))))).astype(float))).astype(float)))))) +
                   np.tanh((0.811111 - ((1.666670 - ((data["group_1"] * ((((np.sin((data["char_8_people"] * ((data["group_1"] <= 0.984375).astype(float)))) * 2.0) < 0.811111).astype(float)) + ((((data["group_1"] + ((((data["group_1"] - data["char_26"]) >= 0.0).astype(float)) * ((data["group_1"] - 1.0) + np.cos(2.828570)))) >= 0.0).astype(float)) * 2.0))) * 2.0)) * 2.0))) +
                   np.tanh((((data["char_2_people"] - data["char_13"]) - ((math.cos(1.414214) >= (data["char_2_people"] - (((data["char_2_people"] * ((data["char_2_people"] >= np.sin(np.tanh(np.cos(data["char_7_people"])))).astype(float))) <= 0.020833).astype(float)))).astype(float))) * ((23.500000 + 0.970149) + ((data["char_2_people"] <= 0.020833).astype(float))))) +
                   np.tanh(((0.872727 * (((data["group_1"] + np.minimum( (2.782610),  ((data["group_1"] - (((data["group_1"] < np.tanh(((3.0 + data["group_1"])/2.0))).astype(float)) * 2.0))))) + ((data["char_34"] >= ((((data["group_1"] < 0.561404).astype(float)) <= np.tanh((data["group_1"] * 2.0))).astype(float))).astype(float))) * 2.0)) * np.maximum( (1.531250),  ((float(2.357140 >= (float(1.666670 >= 0.0)))))))) +
                   np.tanh(((1.0 - (((((data["char_21"] <= ((np.tanh(((0.636620 > np.minimum( (data["char_2_people"]),  (data["char_2_people"]))).astype(float))) + (float((float(1.0 < 3.0)) >= 0.0)))/2.0)).astype(float)) * 2.0) + ((((data["char_2_people"] > 0.636620).astype(float)) <= 2.777780).astype(float))) * np.tanh(((np.tanh((data["char_17"] * 2.357140)) > np.minimum( (data["char_2_people"]),  (2.357140))).astype(float))))) * 2.0)) +
                   np.tanh((((np.sin(data["char_9_people"]) / 2.0) + (((data["char_34"] * 2.0) + (np.minimum( ((data["char_9_people"] * (3.0 * (-(((data["char_28"] < ((2.782610 * ((data["char_9_people"] < (data["char_13"] * np.maximum( (2.0),  ((data["char_36"] / 2.0))))).astype(float))) / 2.0)).astype(float))))))),  ((-(data["char_36"])))) * 2.0)) * 2.0))/2.0)) +
                   np.tanh((((((((6.23677015304565430) * ((data["char_16"] > data["char_24"]).astype(float))) * (((data["char_22"] * 1.0) > data["char_16"]).astype(float))) * ((data["char_16"] > data["char_24"]).astype(float))) + (-((((data["weekend_people"] - np.maximum( (data["char_24"]),  (np.sin((6.23677015304565430))))) <= (((1.261900 + data["char_23"]) >= 1.666670).astype(float))).astype(float)))))/2.0) * 2.0)) +
                   np.tanh(((np.tanh((((((data["char_10_people"] >= data["char_19"]).astype(float)) / 2.0) >= data["char_34"]).astype(float))) + ((((8.0) >= data["char_17"]).astype(float)) * ((0.318310 < np.minimum( (data["char_19"]),  (((data["char_19"] >= np.tanh(np.cos(data["char_17"]))).astype(float))))).astype(float)))) - (data["month_people"] * (float((float(31.006277 > 2.0)) >= 0.0))))) +
                   np.tanh(((np.cos((((19.0 * data["char_2_people"]) - ((2.33206677436828613) + data["activity_category"])) * ((((((data["year_train"] * data["char_2_people"]) * ((0.561404 < data["char_2_people"]).astype(float))) * 2.0) > data["char_26"]).astype(float)) - ((2.428570 >= data["char_2_people"]).astype(float))))) - ((data["activity_category"] + data["char_26"])/2.0)) * 2.0)) +
                   np.tanh((19.0 * ((0.811111 * (data["group_1"] - (((((data["year_people"] + (2.357140 * data["char_36"]))/2.0) * (float(math.sin(-1.0) <= 2.828570))) <= ((math.tanh(2.428570) >= (1.570796 * (((data["group_1"] * data["group_1"]) >= ((data["char_10_people"] <= data["group_1"]).astype(float))).astype(float)))).astype(float))).astype(float)))) - ((data["group_1"] <= data["weekday_people"]).astype(float))))) +
                   np.tanh((((data["char_31"] >= (19.0 * ((data["char_8_people"] + (((((data["business_days_delta"] > data["char_5_people"]).astype(float)) > data["char_5_people"]).astype(float)) - data["char_1_people"])) / 2.0))).astype(float)) - ((((((data["char_5_people"] + data["char_1_people"]) - (((0.0) + ((((data["char_5_people"] + data["char_1_people"]) / 2.0) >= 0.0).astype(float)))/2.0)) > data["char_31"]).astype(float)) < np.maximum( (0.811111),  (data["char_5_people"]))).astype(float)))) +
                   np.tanh((((data["char_4_people"] <= data["year_train"]).astype(float)) - (data["weekday_people"] + np.sin((data["char_28"] - (((0.561404 < ((np.maximum( (data["char_4_people"]),  (((data["char_1_people"] < data["weekday_people"]).astype(float)))) <= data["weekday_people"]).astype(float))).astype(float)) * 2.0)))))) +
                   np.tanh((np.sin((data["group_1"] - np.cos((((data["char_23"] + ((data["char_35"] * 2.0) / 2.0))/2.0) - (np.cos(((data["group_1"] - (np.minimum( (np.maximum( (4.166670),  (19.0))),  (0.529412)) * np.minimum( ((data["char_26"] * 2.0)),  (((data["char_35"] * 2.0) / 2.0))))) * 23.500000)) * 2.0))))) * 23.500000)) +
                   np.tanh(((((data["day_people"] >= np.tanh(np.tanh((0.970149 - np.sin(data["char_32"]))))).astype(float)) - np.tanh((0.970149 - np.minimum( (data["char_32"]),  (((data["day_people"] >= np.tanh((0.970149 - np.minimum( (np.sin(data["char_32"])),  (np.sin(data["char_37"])))))).astype(float))))))) * 2.0)) +
                   np.tanh(((((data["group_1"] + (((((data["char_20"] + data["group_1"])/2.0) >= (data["char_20"] - (data["group_1"] * data["group_1"]))).astype(float)) - ((((data["year_people"] <= data["group_1"]).astype(float)) > data["char_31"]).astype(float)))) + ((0.636620 - ((((23.500000 > ((data["group_1"] > data["char_31"]).astype(float))).astype(float)) > data["group_1"]).astype(float))) - 0.811111)) * 2.0) * 2.0)) +
                   np.tanh((((data["char_2_people"] - (((data["char_2_people"] < np.sin(((np.tanh(np.sin(data["char_37"])) + ((np.tanh(((data["month_people"] + ((2.828570 <= data["char_23"]).astype(float)))/2.0)) > np.minimum( ((data["char_29"] * data["year_train"])),  ((0.529412 * 2.0)))).astype(float)))/2.0))).astype(float)) * 2.0)) * 2.0) - (((((data["char_14"] + data["char_2_people"])/2.0) < np.minimum( (3.141593),  (data["char_23"]))).astype(float)) * 2.0))) +
                   np.tanh(((23.500000 * 2.0) * np.cos((((data["group_1"] <= np.sin(((((data["group_1"] > ((math.sin((float(1.261900 >= 0.0))) < ((data["group_1"] >= 0.0).astype(float))).astype(float))).astype(float)) < np.minimum( (data["month_train"]),  (np.cos((((((data["char_24"] > (data["group_1"] * (4.166670 * 2.0))).astype(float)) * 2.0) - (((data["group_1"] <= 0.561404).astype(float)) * 2.0)) + data["group_1"]))))).astype(float)))).astype(float)) * 2.0)))) +
                   np.tanh((19.0 * np.sin((((data["char_16"] < ((data["char_6_people"] < (1.0 - np.tanh(np.sin(((data["char_14"] >= ((data["char_6_people"] >= data["char_9_people"]).astype(float))).astype(float)))))).astype(float))).astype(float)) - np.cos((np.maximum( (data["char_8_people"]),  (data["char_18"])) * ((data["weekend_people"] + np.tanh(19.0))/2.0))))))) +
                   np.tanh(((((data["day_people"] <= ((((((data["day_people"] <= ((0.318310 + np.maximum( (data["char_11"]),  ((data["char_26"] - (((((0.0 >= data["day_people"]).astype(float)) + 0.529412)/2.0) / 2.0)))))/2.0)).astype(float)) + np.maximum( (data["char_11"]),  ((data["char_29"] / 2.0))))/2.0) + np.maximum( (data["char_11"]),  ((data["char_26"] - data["business_days_delta"]))))/2.0)).astype(float)) * 2.0) - data["char_18"])) +
                   np.tanh(((((np.tanh((data["char_25"] - ((data["char_22"] < (-(np.sin(np.tanh((-((data["char_32"] + ((-2.0 >= data["char_25"]).astype(float)))))))))).astype(float)))) > np.minimum( (np.maximum( (((0.561404 <= data["char_3_people"]).astype(float))),  (0.367879))),  (((data["char_16"] <= data["char_10_people"]).astype(float))))).astype(float)) * 2.0) - ((data["char_16"] * 3.0) / 2.0))) +
                   np.tanh((((data["char_7_people"] > (0.367879 * (0.318310 + (((((((data["char_7_people"] > data["month_people"]).astype(float)) - data["char_7_people"]) > data["month_people"]).astype(float)) > np.tanh((0.318310 - data["char_7_people"]))).astype(float))))).astype(float)) - np.tanh((data["char_18"] * np.cos(((data["char_7_people"] > data["month_people"]).astype(float))))))))

    return Outputs(predictions*.1)


if __name__ == "__main__":
    print('Started!')
    INPUT_PATH = '../input/'
    train = pd.read_csv(INPUT_PATH + 'act_train.csv', parse_dates=['date'])
    train.sort_values(by='activity_id', inplace=True)
    train.reset_index(drop=True, inplace=True)
    test = pd.read_csv(INPUT_PATH + 'act_test.csv', parse_dates=['date'])
    test.sort_values(by='activity_id', inplace=True)
    test.reset_index(drop=True, inplace=True)
    people = pd.read_csv(INPUT_PATH + 'people.csv', parse_dates=['date'])
    train, trainoutcomes, test, features = MungeData(train, test, people)
    train['outcome'] = trainoutcomes
    test['outcome'] = 0

    subset = train.columns[1:-1]

    print('LOO train')
    lootrain = pd.DataFrame()
    for col in subset:
            lootrain[col] = LeaveOneOut(train, train, col, True).values

    traininv1 = GPIndividual(lootrain)
    print('ROC1:', roc_auc_score(train.outcome.values, traininv1))

    print('Part of the group date')
    datagroup = train[['group_1', 'date_train', 'outcome']].copy()
    x = datagroup.groupby(['group_1', 'date_train'])['outcome'].mean()
    x = x.reset_index(drop=False)
    visibletest = pd.merge(train, x, how='left', suffixes=('', '__grpdate'),
                           on=['group_1', 'date_train'], left_index=True)
    visibletest.sort_values(by='activity_id', inplace=True)
    visibletest.reset_index(drop=True, inplace=True)
    traininv1[visibletest.outcome__grpdate == 0.0] = 0.
    traininv1[visibletest.outcome__grpdate == 1.0] = 1.
    print('ROC & Heuristics:', roc_auc_score(train.outcome.values, traininv1))

    print('LOO test')
    lootest = pd.DataFrame()
    for col in subset:
        lootest[col] = LeaveOneOut(train, test, col, False).values

    testinv1 = GPIndividual(lootest)
    submission = pd.DataFrame({'activity_id': test.activity_id.values,
                               'outcome': testinv1.values})
    submission.sort_values(by='activity_id', inplace=True)
    submission.reset_index(drop=True, inplace=True)

    print('Part of the group date trick')

    datagroup = train[['group_1', 'date_train', 'outcome']].copy()
    x = datagroup.groupby(['group_1', 'date_train'])['outcome'].mean()
    x = x.reset_index(drop=False)
    visibletest = pd.merge(test, x, how='left', suffixes=('', '__grpdate'),
                           on=['group_1', 'date_train'], left_index=True)
    visibletest.sort_values(by='activity_id', inplace=True)
    visibletest.reset_index(drop=True, inplace=True)
    submission.loc[visibletest.outcome__grpdate == 0.0, 'outcome'] = 0.
    submission.loc[visibletest.outcome__grpdate == 1.0, 'outcome'] = 1.
    print('Saving File')
    submission.to_csv('gpsubmission.csv', index=False)
    print('Completed!')
