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
    trainoutcomes = train.outcome
    train.drop('outcome', inplace=True, axis=1)
    features = get_features(train, test)
    return train, trainoutcomes, test, features


def MungeData(train, test, people):
    train, trainoutcomes, test, features = read_test_train(train, test, people)
    return train, trainoutcomes, test, features


def GPIndividual(data):
    predictions = (np.tanh((((data["char_6_people"] + (((data["char_13"] * 2.0) * (data["char_2_people"] * 2.0)) - (((np.maximum( (((7.285710 + -3.0)/2.0)),  (0.692308)) * 2.0) >= 0.0).astype(float)))) * 2.0) * ((((31.006277 + ((0.634146 + np.tanh(data["char_2_people"]))/2.0))/2.0) + data["char_2_people"])/2.0))) +
                   np.tanh((2.466670 * (data["char_36"] + (np.sin((((data["char_4_train"] < (2.718282 * (data["char_7_people"] / 2.0))).astype(float)) - ((((data["char_1_people"] < np.sin(2.466670)).astype(float)) * ((data["day_train"] >= (data["char_8_people"] * ((np.maximum( (data["weekend_people"]),  (data["char_8_people"])) > np.cos(((data["char_4_train"] < (2.718282 * (data["char_7_people"] / 2.0))).astype(float)))).astype(float)))).astype(float))) * 2.0))) * 2.0)))) +
                   np.tanh(((31.006277 * 31.006277) * (((((0.276923 <= ((data["char_34"] + ((data["char_13"] >= np.tanh(((0.147059 + ((data["char_34"] > ((2.57123780250549316) * (-(((data["char_34"] >= np.tanh(data["char_1_people"])).astype(float)))))).astype(float)))/2.0))).astype(float)))/2.0)).astype(float)) * 2.0) + (-(np.cos((1.0 + data["char_1_people"])))))/2.0))) +
                   np.tanh((((9.869604 * (np.minimum( (1.787230),  ((np.minimum( (1.787230),  ((((np.maximum( (2.0),  (0.692308)) + data["char_6_people"])/2.0) + (data["char_38"] - 1.666670)))) * 2.0))) * 2.0)) - 1.787230) * 2.0)) +
                   np.tanh((-3.0 - ((9.869604 + (data["group_1"] * (9.869604 * (-((((data["char_10_train"] + (-3.0 - ((data["activity_category"] + (data["group_1"] * (9.869604 * (-((((1.787230 + (((((1.0 * 0.439394) < data["char_28"]).astype(float)) < np.cos(((1.492310 < data["char_10_train"]).astype(float)))).astype(float)))/2.0) * 2.0))))))/2.0)))/2.0) * 2.0))))))/2.0))) +
                   np.tanh((0.318310 + (31.006277 * ((-(((1.492310 + (-2.0 + ((data["char_36"] <= (((data["char_25"] <= ((9.869604 * ((1.492310 + (-2.0 + ((data["char_25"] <= (((data["char_17"] <= (0.367879 + (np.tanh(0.439394) - np.tanh(data["char_1_people"])))).astype(float)) / 2.0)).astype(float))))/2.0)) / 2.0)).astype(float)) / 2.0)).astype(float))))/2.0))) * 2.0)))) +
                   np.tanh((((1.666670 - data["char_21"]) * 2.0) - ((31.006277 * np.cos(np.maximum( ((np.maximum( (np.cos(np.minimum( ((((2.718282 * data["group_1"]) >= data["group_1"]).astype(float))),  ((14.20264434814453125))))),  (data["group_1"])) * 2.0)),  ((2.718282 * data["group_1"]))))) * 2.0))) +
                   np.tanh((31.006277 * (data["char_2_people"] - 0.367879))) +
                   np.tanh((((7.0) * (data["char_9_people"] + np.minimum( ((((7.0) * (data["char_9_people"] + np.minimum( (data["char_6_people"]),  (data["char_9_people"])))) - (7.285710 * (((7.285710 * ((data["char_17"] > np.minimum( (data["char_6_people"]),  (data["char_9_people"]))).astype(float))) > data["char_9_people"]).astype(float))))),  ((float(0.920000 > 0.634146)))))) - (7.285710 * data["char_36"]))) +
                   np.tanh(((np.cos(((-(((data["group_1"] <= np.maximum( (np.cos((float(0.147059 >= 0.0)))),  (data["char_25"]))).astype(float)))) * 2.0)) * 9.869604) * 2.0)) +
                   np.tanh((9.869604 * (((np.tanh((data["char_9_people"] + np.sin((data["char_7_people"] - ((data["char_10_people"] <= np.cos(np.maximum( (np.maximum( ((-(1.732051))),  ((data["char_1_people"] + data["char_7_people"])))),  ((data["char_29"] + data["char_7_people"]))))).astype(float)))))) * 2.0) * 2.0) * 2.0))) +
                   np.tanh((((-((np.sin(data["char_4_people"]) - ((((14.37779235839843750) - (3.0 * 2.0)) * (data["group_1"] - data["char_9_people"])) * data["group_1"])))) * 3.115380) * ((data["char_3_train"] + ((data["char_34"] + (3.115380 + 1.414214)) + 3.115380)) + (((0.920000 <= data["char_31"]).astype(float)) * 2.0)))) +
                   np.tanh(((3.0) + ((data["char_7_people"] * 2.0) - ((9.0) * ((((-(np.sin(np.tanh(data["char_1_people"])))) + ((((1.115940 > (data["char_13"] + np.minimum( (0.855072),  (data["char_7_people"])))).astype(float)) > 0.276923).astype(float))) >= ((data["char_32"] + (data["char_7_people"] - (-((data["char_10_train"] - 0.367879)))))/2.0)).astype(float)))))) +
                   np.tanh(((-2.0 + ((-1.0 - (-((-3.0 + np.maximum( ((data["char_38"] * 1.414214)),  ((((((data["char_38"] > data["char_26"]).astype(float)) * np.maximum( (-3.0),  (((((data["char_38"] >= np.tanh(data["char_14"])).astype(float)) * 2.0) * 2.0)))) * 2.0) * 2.0))))))) + 0.634146)) * 2.0)) +
                   np.tanh(((((2.718282 * ((data["char_38"] * 2.0) - data["char_25"])) - data["char_25"]) * 7.285710) - np.maximum( (((data["char_38"] * 2.0) - np.tanh(np.sin(((data["char_25"] + np.minimum( (data["char_14"]),  (data["char_38"])))/2.0))))),  (np.maximum( (3.115380),  ((((data["char_38"] * 2.0) + 1.732051)/2.0))))))) +
                   np.tanh(((((((((((data["char_10_train"] - ((data["day_people"] <= ((data["char_12"] >= data["char_2_people"]).astype(float))).astype(float))) * 2.0) * 1.570796) - ((data["char_38"] <= 0.636620).astype(float))) * 2.0) - ((data["char_1_people"] > np.cos(1.115940)).astype(float))) * 2.0) * 2.0) - (float(3.115380 <= 7.285710))) * 2.0)) +
                   np.tanh((2.022220 * ((7.0) * ((((data["group_1"] + ((((data["group_1"] - np.sin(data["char_5_people"])) * 2.0) * 2.0) - np.minimum( (2.022220),  (((0.333333 >= (-(((data["char_5_people"] >= (0.367879 - (12.83187294006347656))).astype(float))))).astype(float)))))) / 2.0) * 2.0) - data["char_34"])))) +
                   np.tanh(((((31.006277 + data["char_2_people"])/2.0) * (0.147059 + (data["char_2_people"] - (((0.920000 / 2.0) < ((data["char_5_people"] >= data["char_2_people"]).astype(float))).astype(float))))) * 2.0)) +
                   np.tanh((((-(9.869604)) * np.tanh(9.869604)) * np.minimum( (data["char_7_train"]),  ((0.439394 - ((((((3.0) * ((data["group_1"] / 2.0) - (np.cos(0.439394) - data["group_1"]))) + ((0.439394 * (np.sin(data["char_35"]) * data["char_14"])) / 2.0)) * np.sin(data["group_1"])) > data["char_21"]).astype(float))))))) +
                   np.tanh(((data["char_9_people"] - np.minimum( (2.466670),  (np.sin((((data["char_34"] <= (data["char_9_people"] * 2.466670)).astype(float)) + (31.006277 * (((np.sin(data["char_8_people"]) * data["char_8_people"]) >= np.sin(data["char_17"])).astype(float)))))))) * ((np.tanh((0.367879 + ((31.006277 + 1.666670) - 7.285710))) * 2.0) * 2.0))) +
                   np.tanh(((31.006277 - (((data["char_2_people"] - ((data["char_2_people"] < data["char_4_train"]).astype(float))) >= 0.0).astype(float))) * (((7.285710 + data["char_2_people"]) * (data["char_2_people"] - data["char_5_people"])) - data["char_5_people"]))) +
                   np.tanh((31.006277 * ((-2.0 + ((data["char_10_train"] * 2.0) + np.cos(((data["weekday_people"] / 2.0) - (((data["char_12"] >= np.tanh(np.minimum( (3.0),  (data["char_16"])))).astype(float)) + (float(31.006277 < (float(-3.0 > 2.718282)))))))))/2.0))) +
                   np.tanh((np.maximum( (data["char_11"]),  (2.718282)) * (((data["char_10_train"] - np.cos(np.cos((data["weekday_people"] - np.cos(((data["month_people"] >= data["char_7_train"]).astype(float))))))) * 2.0) * 2.0))) +
                   np.tanh(((-2.0 + (data["group_1"] * np.minimum( (data["group_1"]),  (-2.0)))) + (31.006277 * np.minimum( (((np.minimum( (2.466670),  (np.sin((data["group_1"] - data["char_8_people"])))) >= 0.0).astype(float))),  ((data["group_1"] - data["char_37"])))))) +
                   np.tanh((((((((data["group_1"] + np.cos(2.718282)) + (((0.920000 * data["group_1"]) >= data["char_14"]).astype(float))) + np.cos(2.718282)) + (((0.920000 * data["group_1"]) >= data["char_32"]).astype(float))) * 2.0) * 2.0) * 2.0)) +
                   np.tanh((((((31.006277 * (4.0)) + (data["group_1"] - (0.571429 * ((0.147059 + (data["group_1"] - (((float(0.634146 >= 0.0)) <= (-(((data["group_1"] - data["char_20"]) - data["group_1"])))).astype(float))))/2.0))))/2.0) * (data["group_1"] - (0.571429 * ((data["char_20"] + ((0.147059 + (4.0))/2.0))/2.0)))) / 2.0)) +
                   np.tanh(((7.87782144546508789) * ((((2.718282 + (np.cos(np.maximum( (0.571429),  (0.0))) + 31.006277)) * np.minimum( ((data["group_1"] - data["char_10_people"])),  (data["group_1"]))) - ((np.maximum( ((data["group_1"] + data["char_33"])),  (((data["group_1"] - data["char_34"]) * 2.0))) > (((((data["group_1"] >= data["char_34"]).astype(float)) * 2.0) * 2.0) / 2.0)).astype(float))) * 2.0))) +
                   np.tanh((9.869604 * (data["group_1"] - np.maximum( (((data["group_1"] < (0.439394 * np.minimum( (data["group_1"]),  (data["char_4_people"])))).astype(float))),  (np.sin(((data["group_1"] < np.sin(np.maximum( (data["char_4_people"]),  (np.maximum( ((-(((((data["group_1"] >= 0.0).astype(float)) * 2.0) / 2.0)))),  (np.sin(((data["group_1"] < np.sin(np.maximum( (data["char_4_people"]),  (((0.439394 >= np.sin(data["year_train"])).astype(float)))))).astype(float))))))))).astype(float)))))))) +
                   np.tanh(((((((data["char_6_people"] - np.sin(np.cos(2.022220))) < (data["group_1"] * (0.276923 + (data["group_1"] - np.cos((data["group_1"] * 2.0)))))).astype(float)) / 2.0) - ((0.692308 < np.cos(data["group_1"])).astype(float))) * ((11.47381019592285156) + np.tanh(data["group_1"])))) +
                   np.tanh(((((3.115380 * (((3.115380 * np.tanh(data["char_2_people"])) * 2.0) + (data["day_people"] - (3.115380 + np.cos((np.tanh(data["weekday_train"]) * 2.0)))))) * 2.0) * 2.0) + ((-((3.115380 * (data["char_2_people"] - ((data["char_31"] + ((3.0 <= (data["day_people"] / 2.0)).astype(float)))/2.0))))) / 2.0))) +
                   np.tanh((-(((0.367879 + (31.006277 * ((np.sin(np.maximum( ((((data["char_2_people"] * np.minimum( (data["char_2_people"]),  (np.tanh(31.006277)))) >= data["char_2_people"]).astype(float))),  ((1.300000 - (((((float(0.636620 >= 0.0)) / 2.0) * 2.0) * data["day_people"]) * 3.0))))) + (data["char_15"] - data["char_2_people"]))/2.0)))/2.0)))) +
                   np.tanh((0.0 + ((-((((np.tanh((((data["char_10_train"] * 31.006277) + data["activity_category"])/2.0)) * 31.006277) + data["char_15"])/2.0))) + (data["char_2_people"] * np.maximum( (((3.0 * (data["char_10_train"] * 31.006277)) + (data["char_10_train"] * 31.006277))),  (0.367879)))))) +
                   np.tanh(((-3.0 - 31.006277) - 31.006277)) +
                   np.tanh((7.285710 * ((-(1.300000)) + (((1.492310 < (-2.0 + (3.115380 * (((-(1.787230)) + ((data["group_1"] * 2.0) * 2.0)) - (np.tanh(data["char_7_people"]) * (((((1.666670 < (np.sin(data["group_1"]) * data["group_1"])).astype(float)) * 2.0) >= 0.0).astype(float))))))).astype(float)) * 2.0)))) +
                   np.tanh((31.006277 * ((data["char_2_people"] * (3.0 * (data["char_10_train"] * ((((7.285710 * (data["char_2_people"] - data["char_13"])) > ((data["char_1_train"] >= data["char_32"]).astype(float))).astype(float)) * 2.0)))) - data["weekday_people"]))) +
                   np.tanh(((12.24423313140869141) * ((12.24423313140869141) * (((-(np.sin(((data["char_38"] <= ((np.sin(((data["char_38"] <= ((data["char_1_people"] + ((data["char_1_people"] + ((float(0.571429 * 1.509090) >= 0.0))) / 2.0)) / 2.0)).astype(float))) + (data["char_17"] + (((0.571429 * data["char_38"]) >= 0.0).astype(float)))) / 2.0)).astype(float))))) * 2.0) + 0.333333)))) +
                   np.tanh((-2.0 + ((((7.285710 * (7.285710 * (data["group_1"] - np.sin(np.cos((data["char_34"] - data["group_1"])))))) / 2.0) * 2.0) + (-2.0 + (data["char_30"] * (data["group_1"] + (data["group_1"] - ((3.115380 > (((-1.0 - 31.006277) + ((data["group_1"] >= 0.0).astype(float)))/2.0)).astype(float))))))))) +
                   np.tanh((data["char_37"] - ((((1.666670 - ((data["char_6_people"] > np.cos(((0.439394 <= data["char_34"]).astype(float)))).astype(float))) * 2.0) - ((1.570796 + np.tanh(np.tanh(((float(0.920000 >= 0.0)) + (float(0.855072 >= 0.0))))))/2.0)) * 2.0))) +
                   np.tanh((((((np.minimum( (((data["char_6_people"] + ((((data["char_38"] * 2.0) >= ((2.022220 >= (((data["char_38"] * 2.0) >= 0.571429).astype(float))).astype(float))).astype(float)) - (float(0.439394 <= 1.570796)))) * (4.0))),  (((((data["char_8_people"] + (1.570796 / 2.0))/2.0) <= data["char_38"]).astype(float)))) * 2.0) - np.tanh(2.718282)) * 2.0) * 2.022220) - np.cos(0.367879))) +
                   np.tanh(((((data["char_10_train"] - (((data["char_20"] > data["char_3_people"]).astype(float)) / 2.0)) * 2.0) * data["year_train"]) + (((data["char_3_people"] - (((data["char_8_people"] > (data["month_train"] * data["char_3_people"])).astype(float)) + (((data["char_20"] > data["char_6_people"]).astype(float)) - data["char_10_train"]))) * 2.0) * 2.0))) +
                   np.tanh(np.minimum( ((data["char_6_people"] - ((np.cos(((1.300000 + ((data["char_6_people"] <= np.sin(((np.cos(((((data["char_6_people"] * 2.0) * np.sin(3.0)) >= 0.0).astype(float))) >= data["char_6_people"]).astype(float)))).astype(float)))/2.0)) >= data["month_train"]).astype(float)))),  (((data["char_25"] - ((data["char_18"] > (data["char_6_people"] * 2.0)).astype(float))) - ((math.sin(3.0) > data["char_6_people"]).astype(float)))))) +
                   np.tanh(((((((0.439394 * 2.0) - ((data["char_1_people"] > (data["char_38"] - ((data["char_30"] < (((data["char_6_people"] * 2.0) < data["char_7_train"]).astype(float))).astype(float)))).astype(float))) * 2.0) - ((data["char_1_people"] > data["char_38"]).astype(float))) * 2.0) - np.cos(np.cos(np.sin(1.0))))) +
                   np.tanh((((((((9.0) * np.sin((data["group_1"] - (np.sin(((0.72523373365402222) - data["group_1"])) + (((float((0.72523373365402222) >= (3.0))) >= (np.sin(((0.72523373365402222) - data["group_1"])) + ((data["weekend_people"] >= np.minimum( (3.0),  (3.0))).astype(float)))).astype(float)))))) * 2.0) * 2.0) * 2.0) - data["group_1"]) * 2.0)) +
                   np.tanh(((4.84525060653686523) * (((4.84525060653686523) * (data["group_1"] - np.minimum( (data["char_10_train"]),  (data["char_9_people"])))) + (-(((data["year_train"] < data["char_9_train"]).astype(float))))))) +
                   np.tanh((((np.cos((((data["char_7_train"] - np.sin(((np.maximum( (data["char_14"]),  ((9.0))) >= 0.0).astype(float)))) <= 9.869604).astype(float))) + ((data["char_2_people"] - (((data["char_2_people"] - ((data["char_2_people"] - (((data["char_25"] >= data["char_7_people"]).astype(float)) * 2.0)) * 2.0)) >= 0.0).astype(float))) * 2.0)) * 2.0) - (((((data["char_28"] >= data["char_7_people"]).astype(float)) * 2.0) + np.sin((float(1.509090 >= 0.0))))/2.0))) +
                   np.tanh(((-(((((float(3.0 <= (float(0.920000 > 3.141593)))) + (((data["char_37"] <= data["char_6_people"]).astype(float)) + (((((data["char_37"] < data["char_7_people"]).astype(float)) * 9.869604) <= ((((np.maximum( (data["char_7_people"]),  ((2.466670 * 2.0))) > np.tanh(0.276923)).astype(float)) >= ((0.318310 + data["char_3_train"])/2.0)).astype(float))).astype(float))))/2.0) + ((data["char_7_people"] <= data["char_6_people"]).astype(float))))) * 2.0)) +
                   np.tanh(np.minimum( ((0.692308 * 2.022220)),  ((np.minimum( ((data["char_15"] + (-(((data["year_train"] <= (np.cos(data["year_train"]) * (data["char_6_people"] + ((min( (0.692308),  (0.439394)) >= data["year_train"]).astype(float))))).astype(float)))))),  (((np.cos(data["char_11"]) >= data["year_train"]).astype(float)))) * 2.022220)))) +
                   np.tanh((((data["char_7_people"] + ((-(((((np.sin(np.sin(np.maximum( (0.692308),  (((data["char_7_people"] * (31.006277 + data["char_7_people"])) - ((data["char_25"] < 1.115940).astype(float))))))) / 2.0) - ((data["char_7_people"] > np.sin(np.tanh(data["char_21"]))).astype(float))) >= 0.0).astype(float)))) * 2.0)) * 2.0) / 2.0)) +
                   np.tanh((31.006277 + (-(((-((data["group_1"] + (np.maximum( (31.006277),  (3.0)) * (np.maximum( (31.006277),  (31.006277)) * (data["group_1"] - (data["char_1_people"] + ((data["group_1"] <= (data["char_38"] - (((0.276923 > (((data["group_1"] - np.minimum( (data["char_30"]),  (3.0))) >= 0.0).astype(float))).astype(float)) * 2.0))).astype(float))))))))) * 2.0))))) +
                   np.tanh((np.minimum( (((data["char_17"] * 2.0) - ((0.333333 <= ((data["activity_category"] > ((data["char_1_people"] <= data["month_people"]).astype(float))).astype(float))).astype(float)))),  (3.141593)) - ((((data["char_18"] < data["char_17"]).astype(float)) <= ((0.333333 >= (data["char_25"] + ((data["char_12"] >= (-1.0 / 2.0)).astype(float)))).astype(float))).astype(float)))) +
                   np.tanh(((((((((data["char_38"] > 0.276923).astype(float)) - np.maximum( ((data["char_12"] + ((0.636620 < data["char_1_people"]).astype(float)))),  (data["char_32"]))) * 2.0) - data["char_1_people"]) * 2.0) * 2.0) - np.maximum( (0.636620),  (np.cos(data["char_38"]))))) +
                   np.tanh((-(((((data["char_25"] - 0.147059) >= data["char_38"]).astype(float)) * 9.869604)))) +
                   np.tanh(np.sin(((np.sin(((((-(data["char_5_people"])) - (np.tanh(np.maximum( (data["char_5_train"]),  (((((-((data["char_25"] - 0.692308))) - (np.tanh(np.maximum( (data["char_5_train"]),  ((1.509090 * ((-(data["char_5_people"])) * 2.0))))) / 2.0)) * 2.0) * 2.0)))) / 2.0)) * 2.0) * 2.0)) * 2.0) * 2.0))) +
                   np.tanh((((9.869604 * (3.0 * (-(((data["char_6_people"] <= ((np.tanh(((data["char_6_people"] <= (np.cos(data["char_10_train"]) / 2.0)).astype(float))) / 2.0) / 2.0)).astype(float)))))) + ((data["char_6_people"] >= (((2.0 <= data["char_24"]).astype(float)) * 2.0)).astype(float))) * ((np.tanh(data["char_6_people"]) <= np.maximum( (0.318310),  ((((0.276923 >= data["char_8_people"]).astype(float)) / 2.0)))).astype(float)))) +
                   np.tanh((((data["char_6_people"] < (data["char_23"] * 1.509090)).astype(float)) - ((data["char_28"] <= ((data["activity_category"] + ((((data["char_22"] <= ((np.sin(data["char_6_people"]) + (((data["char_22"] - data["char_28"]) > np.sin((-3.0 - (-(data["activity_category"]))))).astype(float)))/2.0)).astype(float)) + ((data["char_28"] > (data["char_5_people"] + (-(data["char_28"])))).astype(float)))/2.0))/2.0)).astype(float)))) +
                   np.tanh((((data["char_15"] >= np.cos(((data["char_7_train"] < data["char_17"]).astype(float)))).astype(float)) - np.maximum( (data["char_1_people"]),  (((((((data["char_8_people"] >= data["day_people"]).astype(float)) > 0.634146).astype(float)) > ((data["char_17"] >= np.cos(((0.439394 < data["year_train"]).astype(float)))).astype(float))).astype(float)))))) +
                   np.tanh((((np.tanh(data["group_1"]) + np.minimum( ((np.cos(data["group_1"]) * 2.0)),  ((((data["char_29"] * 2.0) + ((-((((((data["group_1"] * data["group_1"]) * 2.0) * 2.0) <= data["group_1"]).astype(float)))) * 2.0)) * 2.0)))) * 2.0) * 2.0)) +
                   np.tanh(((((data["year_people"] - ((((data["char_1_train"] / 2.0) >= data["char_38"]).astype(float)) - ((((data["char_38"] <= (data["year_people"] - 0.147059)).astype(float)) < (-(((data["char_38"] / 2.0) - ((((data["char_8_people"] > data["char_38"]).astype(float)) * ((data["char_38"] - (-(np.tanh(((data["char_11"] >= 0.0).astype(float)))))) / 2.0)) / 2.0))))).astype(float)))) - data["char_36"]) * 2.0) * 2.0)) +
                   np.tanh((3.115380 * (data["char_38"] - np.tanh(np.cos((np.cos(((data["char_28"] <= data["char_38"]).astype(float))) * (3.115380 * (data["char_9_people"] - np.maximum( (0.318310),  (((data["char_37"] <= np.minimum( (((2.022220 - ((data["char_9_people"] <= 3.115380).astype(float))) / 2.0)),  (((data["char_38"] >= (data["char_9_people"] / 2.0)).astype(float))))).astype(float)))))))))))) +
                   np.tanh(((((0.692308 - (2.466670 * ((data["group_1"] <= ((data["char_38"] + ((((np.tanh((((((data["group_1"] - (0.147059 * (((((-(0.147059)) + data["char_9_train"])/2.0) >= 0.0).astype(float)))) * 2.0) - data["char_36"]) * 2.0) * 2.0)) - np.maximum( (data["char_21"]),  ((-((0.318310 / 2.0)))))) * 2.0) >= 0.0).astype(float)))/2.0)).astype(float)))) * 2.0) * 2.0) * 2.0)) +
                   np.tanh((1.509090 * (np.sin(np.cos(((31.006277 * (-(((data["char_2_people"] < np.maximum( (data["weekend_train"]),  ((((((1.0 + data["weekday_people"])/2.0) <= data["char_34"]).astype(float)) * 2.0)))).astype(float))))) / 2.0))) + ((data["char_2_people"] - ((data["char_2_people"] >= 0.0).astype(float))) - (data["char_20"] + ((data["char_2_people"] < np.minimum( (data["weekend_train"]),  (((data["char_17"] >= 0.0).astype(float))))).astype(float))))))) +
                   np.tanh((((2.0 * (((((data["group_1"] * 2.0) - ((1.492310 > np.sin((data["group_1"] + (0.439394 * 2.0)))).astype(float))) + (np.tanh(((data["year_train"] < (data["group_1"] + (-(0.439394)))).astype(float))) + -1.0)) * 2.0) - np.minimum( (0.439394),  (data["char_4_train"])))) * 2.0) - np.minimum( (data["char_22"]),  (1.0)))) +
                   np.tanh((((np.tanh(data["char_15"]) / 2.0) - (data["char_9_people"] * ((np.minimum( (data["activity_category"]),  (data["char_9_people"])) >= data["char_15"]).astype(float)))) - (7.285710 * ((np.minimum( (data["activity_category"]),  (((np.minimum( (data["activity_category"]),  ((data["char_9_people"] * ((np.minimum( (data["activity_category"]),  (data["char_15"])) >= np.tanh(data["char_15"])).astype(float))))) >= np.tanh(data["char_15"])).astype(float)))) >= data["char_18"]).astype(float))))) +
                   np.tanh(np.sin((-((((3.115380 * (((data["char_35"] / 2.0) > np.sin(0.276923)).astype(float))) * ((data["char_27"] * (((((((((data["char_7_train"] >= 3.141593).astype(float)) <= data["char_4_train"]).astype(float)) >= 0.0).astype(float)) >= 0.0).astype(float)) * 2.0)) - ((data["char_7_train"] <= (3.115380 * (2.0 * 2.0))).astype(float)))) * 2.0))))) +
                   np.tanh((((13.28466606140136719) * np.cos((np.sin(data["group_1"]) - ((((1.492310 * 2.0) * ((data["group_1"] * (1.732051 + ((data["group_1"] * (data["group_1"] * 2.0)) * 2.0))) - 1.509090)) + np.cos(np.cos(3.0)))/2.0)))) * 2.0)) +
                   np.tanh(np.minimum( (1.666670),  (((np.minimum( (0.920000),  (np.minimum( ((10.0)),  ((np.maximum( (1.666670),  (1.666670)) * (-3.0 * ((((data["char_10_train"] * (1.666670 * data["char_38"])) * data["char_10_train"]) <= (((0.439394 + (np.sin(((data["char_10_train"] + 0.571429) - (7.285710 + 0.439394))) * 2.0)) >= 0.0).astype(float))).astype(float)))))))) * 2.0) * 2.0)))) +
                   np.tanh(np.tanh(((data["char_10_train"] > np.sin(np.maximum( (data["char_23"]),  (((np.maximum( ((0.692308 * data["char_23"])),  (((data["char_1_people"] >= np.tanh(np.tanh(((data["char_8_people"] < np.cos(np.cos(((1.300000 <= data["char_10_train"]).astype(float))))).astype(float))))).astype(float)))) >= np.tanh(np.tanh(((data["char_10_train"] >= ((data["day_train"] <= 0.367879).astype(float))).astype(float))))).astype(float)))))).astype(float)))) +
                   np.tanh((((((2.81389904022216797) - (2.81389904022216797)) - np.sin(((data["group_1"] - np.sin(np.tanh(np.tanh(((data["char_11"] >= 0.0).astype(float)))))) * (7.285710 + 2.466670)))) * (7.285710 + 2.466670)) + ((0.692308 < (np.tanh((data["group_1"] * 2.0)) - (data["group_1"] - np.sin(np.tanh(np.tanh((float(9.869604 >= 0.0)))))))).astype(float)))) +
                   np.tanh((data["char_30"] - (((((data["char_15"] / 2.0) >= np.minimum( (data["char_10_train"]),  (((data["char_10_train"] >= np.sin(((data["char_25"] <= np.tanh(data["day_people"])).astype(float)))).astype(float))))).astype(float)) > np.minimum( (data["char_30"]),  ((((data["char_30"] - np.cos(-1.0)) > np.maximum( (data["char_10_train"]),  (((((1.787230 < data["char_10_train"]).astype(float)) >= 0.0).astype(float))))).astype(float))))).astype(float)))) +
                   np.tanh(((((-((((-(data["char_38"])) >= 0.0).astype(float)))) * 2.0) * 2.0) * 2.0)) +
                   np.tanh(((np.minimum( (data["char_19"]),  (data["char_1_people"])) * 2.0) - np.maximum( (data["char_34"]),  (np.maximum( ((((0.276923 * 2.0) < np.maximum( (data["char_34"]),  (data["char_1_people"]))).astype(float))),  (np.maximum( (data["char_34"]),  (np.maximum( (data["char_32"]),  ((-(((((data["activity_category"] > 3.0).astype(float)) >= data["char_32"]).astype(float)))))))))))))) +
                   np.tanh(np.minimum( (np.tanh((((data["char_7_people"] <= (0.571429 * data["year_train"])).astype(float)) - np.tanh((data["char_22"] - np.tanh(np.maximum( (data["char_3_train"]),  (((data["char_31"] < 1.492310).astype(float)))))))))),  ((2.718282 * (2.718282 * (data["char_2_people"] - data["year_train"])))))) +
                   np.tanh((((((((data["group_1"] * 2.0) - ((data["weekend_train"] <= np.cos((np.sin(data["group_1"]) + np.sin((((3.0) * data["group_1"]) * 2.0))))).astype(float))) * 2.0) * 2.0) - ((data["group_1"] >= np.minimum( (0.147059),  (np.cos((-((3.141593 * 2.0))))))).astype(float))) * 2.0) - ((data["group_1"] <= (-3.0 * 2.0)).astype(float)))) +
                   np.tanh((-2.0 * (((data["char_7_people"] >= ((-((((data["char_38"] >= np.tanh(np.sin(np.sin(((data["char_38"] > (((-2.0 * ((data["char_7_people"] > data["char_38"]).astype(float))) >= (float((-2.0 * 31.006277) >= 0.276923))).astype(float))).astype(float)))))).astype(float)) - 1.570796))) / 2.0)).astype(float)) / 2.0))) +
                   np.tanh(np.minimum( (data["char_2_people"]),  (((np.minimum( ((np.minimum( (((math.cos(1.300000) > np.minimum( (((data["char_2_people"] < (4.0)).astype(float))),  (((data["char_2_people"] - 0.571429) * 2.0)))).astype(float))),  ((data["char_2_people"] - 0.571429))) * 2.0)),  (np.minimum( (1.0),  (data["char_2_people"])))) * 2.0) * 2.0)))) +
                   np.tanh(((data["weekend_people"] - np.sin(np.tanh(np.cos((((0.276923 > data["char_8_people"]).astype(float)) - (data["char_27"] + (data["char_27"] - np.tanh(np.cos((data["char_16"] + ((2.466670 > (np.sin((np.sin(((-3.0 - np.cos(0.692308)) / 2.0)) / 2.0)) / 2.0)).astype(float)))))))))))) * 2.0)) +
                   np.tanh((((9.0) * 2.718282) * (data["group_1"] + np.minimum( (np.minimum( ((3.0 - 3.141593)),  (((1.492310 + (((np.minimum( (data["group_1"]),  (((9.0) * (np.cos(((9.0) * (data["group_1"] * 2.0))) * 2.0)))) * 2.022220) / 2.0) / 2.0)) * 2.0)))),  (np.cos(0.855072)))))) +
                   np.tanh(((7.285710 * ((data["group_1"] - ((((((2.718282 > ((data["char_3_train"] >= data["char_12"]).astype(float))).astype(float)) / 2.0) < data["char_7_people"]).astype(float)) / 2.0)) - ((((1.666670 + ((((9.869604 * 2.0) >= data["char_7_people"]).astype(float)) + np.maximum( (data["char_15"]),  (((data["char_29"] >= data["group_1"]).astype(float))))))/2.0) / 2.0) / 2.0))) * np.maximum( (7.285710),  (3.0)))) +
                   np.tanh((((((data["group_1"] - np.cos(((((math.cos(0.318310) <= data["group_1"]).astype(float)) >= ((data["char_4_train"] < ((0.571429 > np.maximum( (data["char_14"]),  (np.sin((data["group_1"] + (1.115940 + -2.0)))))).astype(float))).astype(float))).astype(float)))) * 2.0) * 2.0) * 2.0) * 2.0)) +
                   np.tanh((((np.sin(((data["group_1"] * np.maximum( (3.0),  (np.cos(data["group_1"])))) - np.sin((((((data["group_1"] * 2.0) - ((np.tanh(data["char_3_people"]) >= (data["group_1"] * 3.0)).astype(float))) * 2.0) * np.maximum( ((10.0)),  ((data["group_1"] - np.maximum( (data["char_27"]),  (np.tanh(data["char_1_people"]))))))) / 2.0)))) * 2.0) * 2.0) * 2.0)) +
                   np.tanh((-(((((((data["weekend_train"] > data["char_25"]).astype(float)) - data["char_10_people"]) < ((np.tanh(0.439394) - data["char_36"]) * 2.0)).astype(float)) * 2.0)))) +
                   np.tanh((3.0 * ((4.25551128387451172) * (((data["char_27"] / 2.0) / 2.0) - np.tanh(np.sin(np.cos((data["group_1"] * (9.869604 + ((data["char_25"] < ((-(data["group_1"])) * data["group_1"])).astype(float))))))))))) +
                   np.tanh((((((data["char_7_people"] <= 0.147059).astype(float)) <= 0.147059).astype(float)) - np.maximum( (((data["char_7_people"] > 0.276923).astype(float))),  (((data["char_7_people"] <= 0.147059).astype(float)))))) +
                   np.tanh((((data["group_1"] - np.cos(((data["group_1"] * 7.285710) * 2.0))) - np.cos(((data["char_2_train"] <= (((((1.732051 * data["group_1"]) * 2.0) * 2.0) - 0.634146) * 2.0)).astype(float)))) * ((((2.94066381454467773) * (2.718282 * 2.0)) * data["group_1"]) + ((1.492310 - ((1.115940 < np.tanh(-1.0)).astype(float))) * 2.0)))) +
                   np.tanh((0.367879 + (((data["char_38"] * 0.636620) - 1.492310) + (((((((0.636620 > data["char_38"]).astype(float)) >= 1.787230).astype(float)) < (data["char_38"] * ((0.636620 > ((((0.636620 > data["char_38"]).astype(float)) <= ((data["char_38"] > np.maximum( (((data["char_6_people"] <= 0.367879).astype(float))),  ((np.minimum( (data["char_9_people"]),  (1.570796)) * 2.0)))).astype(float))).astype(float))).astype(float)))).astype(float)) * 2.0)))) +
                   np.tanh(((data["group_1"] - np.sin((np.cos((data["group_1"] * (31.006277 + np.tanh(data["char_34"])))) * ((data["char_1_people"] > 0.276923).astype(float))))) * (31.006277 - ((data["weekend_people"] - 0.147059) * 2.0)))) +
                   np.tanh(np.cos(((1.732051 * (data["char_10_train"] + data["char_10_train"])) * (data["char_10_train"] + ((data["char_10_train"] + ((((data["char_10_train"] + np.sin(((1.732051 * (data["char_10_train"] + data["char_10_train"])) * (data["char_10_train"] + np.minimum( (0.636620),  (data["char_10_train"])))))) >= 0.0).astype(float)) / 2.0)) / 2.0))))) +
                   np.tanh(((np.minimum( (data["group_1"]),  ((data["group_1"] - ((((np.cos(3.0) + data["group_1"]) + (data["group_1"] + (((0.692308 + (((data["char_14"] > 1.666670).astype(float)) * (-(((((0.636620 + data["char_20"])/2.0) >= 0.0).astype(float))))))/2.0) - np.tanh(((data["char_21"] <= 0.571429).astype(float)))))) <= np.sin(((float(1.666670 >= 0.0)) / 2.0))).astype(float))))) * 2.0) * 2.0)) +
                   np.tanh((9.869604 * ((np.minimum( (np.minimum( (data["char_10_people"]),  (((data["char_4_people"] > np.sin((data["char_38"] / 2.0))).astype(float))))),  (data["char_38"])) < np.minimum( (data["char_10_people"]),  ((np.tanh(np.sin(np.minimum( (data["char_38"]),  (data["char_38"])))) * 2.0)))).astype(float)))) +
                   np.tanh(np.cos(((data["char_5_people"] + np.maximum( (((data["char_1_people"] >= np.minimum( ((1.115940 - ((data["char_5_people"] < ((data["char_5_people"] + data["char_34"]) / 2.0)).astype(float)))),  ((((data["char_23"] <= (((np.tanh((data["char_5_people"] + np.tanh(((data["char_5_people"] < data["char_13"]).astype(float))))) <= np.sin(((data["char_5_people"] >= 0.0).astype(float)))).astype(float)) / 2.0)).astype(float)) / 2.0)))).astype(float))),  ((-(data["char_15"]))))) * 2.0))) +
                   np.tanh((1.115940 * ((((((data["char_2_people"] * data["char_7_train"]) > data["char_38"]).astype(float)) - ((np.cos(np.sin(data["char_2_people"])) < (np.tanh(((np.cos(np.sin(data["char_2_people"])) >= ((data["char_7_train"] >= data["char_7_train"]).astype(float))).astype(float))) * 2.0)).astype(float))) * 2.0) * 2.0))) +
                   np.tanh(((((data["year_people"] <= data["char_26"]).astype(float)) * (3.0)) * np.tanh((np.maximum( (data["year_train"]),  (data["char_26"])) - (((0.920000 > ((np.tanh(np.minimum( (np.cos(((data["char_10_people"] > (-(data["char_20"]))).astype(float)))),  (data["char_17"]))) * ((0.439394 < data["year_train"]).astype(float))) * 2.0)).astype(float)) * 2.0))))) +
                   np.tanh((((((0.571429 - ((np.tanh((np.tanh(1.509090) * (((((data["group_1"] * (6.0)) * 2.0) + (-(((((1.732051 <= np.minimum( (np.sin(3.115380)),  (np.sin((data["group_1"] * 2.0))))).astype(float)) <= data["group_1"]).astype(float))))) * 2.0) * (-(data["group_1"]))))) >= 0.0).astype(float))) * 2.0) * 2.0) * 2.0) * 2.0)) +
                   np.tanh((((data["char_2_people"] + (-(np.sin(np.sin(((data["char_11"] <= (0.634146 - ((data["char_2_people"] + (-(np.sin(((((data["char_2_people"] + (-(np.sin(((data["char_2_people"] >= data["char_35"]).astype(float)))))) * 2.0) < np.cos(np.sin((float(0.634146 < 2.718282))))).astype(float)))))) * 2.0))).astype(float))))))) * 2.0) * 2.0)) +
                   np.tanh(((((((((data["group_1"] - (((((2.0 - data["group_1"]) >= 0.0).astype(float)) <= ((((data["group_1"] - data["char_3_people"]) / 2.0) >= 0.0).astype(float))).astype(float))) * 2.0) * 2.0) + (data["group_1"] - np.tanh(data["char_38"]))) * 2.0) * 2.0) * 2.0) * 2.0)) +
                   np.tanh(((np.minimum( ((0.571429 + ((1.509090 + data["char_22"])/2.0))),  (((((data["group_1"] > ((0.439394 <= ((data["group_1"] + (data["char_6_train"] - ((np.cos(data["group_1"]) / 2.0) * 2.0))) * 2.0)).astype(float))).astype(float)) - np.cos(np.maximum( ((data["group_1"] * 2.0)),  (np.sin(0.920000))))) * 2.0))) + (data["group_1"] - 0.571429)) * 2.0)) +
                   np.tanh((data["char_21"] - ((0.0 >= (np.sin((np.cos(2.466670) + ((data["year_people"] < np.sin((((((data["char_21"] > 0.571429).astype(float)) + data["weekend_people"]) >= (0.920000 * (data["char_21"] - (data["char_21"] - data["char_6_people"])))).astype(float)))).astype(float)))) * (data["char_21"] - (data["char_6_people"] * 2.0)))).astype(float)))) +
                   np.tanh(((((np.sin(np.sin((0.920000 - data["month_train"]))) > ((data["char_38"] + np.maximum( (0.276923),  ((data["char_15"] * np.sin(np.sin((0.0 - data["month_train"])))))))/2.0)).astype(float)) - (np.minimum( (data["month_train"]),  (np.sin(np.sin(((data["char_18"] * (10.0)) / 2.0))))) - np.minimum( (data["month_train"]),  (((data["char_6_people"] <= 0.367879).astype(float)))))) * 2.0)) +
                   np.tanh(((data["char_14"] * 2.0) * (((data["char_14"] * 2.0) * (0.276923 - (((np.maximum( (data["char_32"]),  (-1.0)) * 1.115940) > np.maximum( (0.692308),  (((-(data["char_32"])) / 2.0)))).astype(float)))) - np.minimum( (np.minimum( (data["char_14"]),  ((((np.tanh(data["char_21"]) <= data["char_32"]).astype(float)) * 2.0)))),  (data["char_32"]))))) +
                   np.tanh(((data["char_2_people"] - (((3.115380 + (0.0 - np.sin(np.minimum( (data["char_2_people"]),  (((3.0 <= (data["char_28"] * np.maximum( (((3.115380 + np.minimum( (0.318310),  (((1.414214 <= np.cos((31.006277 * 2.0))).astype(float)))))/2.0)),  ((5.12667894363403320))))).astype(float)))))))/2.0) - ((data["char_2_people"] < (data["char_2_people"] * 2.0)).astype(float)))) * 2.0)) +
                   np.tanh((-((np.minimum( (data["char_9_people"]),  (((((data["char_1_people"] > np.tanh(data["char_32"])).astype(float)) >= 0.0).astype(float)))) * ((np.sin(data["activity_category"]) > np.minimum( (data["char_8_people"]),  (data["char_1_people"]))).astype(float)))))) +
                   np.tanh((3.0 * ((data["group_1"] - ((data["group_1"] <= np.tanh((0.276923 + np.maximum( (1.492310),  (0.147059))))).astype(float))) + (data["group_1"] - ((np.sin(np.tanh((np.tanh(data["group_1"]) + (((data["char_2_train"] + (data["group_1"] - ((data["group_1"] <= 0.920000).astype(float)))) * 2.0) - ((np.sin(np.tanh(0.367879)) >= 0.0).astype(float)))))) >= 0.0).astype(float)))))) +
                   np.tanh(np.minimum( (np.sin(((data["year_people"] < np.minimum( (1.115940),  ((data["char_26"] - ((0.855072 >= ((np.cos((data["char_2_people"] * 2.0)) * 2.0) * 2.0)).astype(float)))))).astype(float)))),  (((6.07185411453247070) * ((data["char_2_people"] - np.cos(0.920000)) * 2.0))))) +
                   np.tanh((((data["char_2_people"] - (1.570796 + np.cos(((((3.70186781883239746) + 3.141593) + (data["char_7_people"] * ((-((-((3.141593 / 2.0))))) - (((np.sin(((data["char_36"] / 2.0) * 2.0)) < (1.115940 - 0.634146)).astype(float)) * 3.141593)))) / 2.0)))) * 2.0) * 2.0)) +
                   np.tanh(((data["char_26"] > np.cos(np.maximum( (((1.666670 < data["char_32"]).astype(float))),  ((((3.0 + (data["char_7_people"] + (1.509090 * ((data["weekday_train"] <= ((data["char_14"] < (data["char_9_people"] * data["char_28"])).astype(float))).astype(float)))))/2.0) / 2.0))))).astype(float))) +
                   np.tanh((((data["char_1_people"] >= data["char_2_people"]).astype(float)) * (-3.0 * 2.0))) +
                   np.tanh(((((((0.333333 <= data["char_2_people"]).astype(float)) * 2.0) >= 1.115940).astype(float)) - (((1.300000 >= ((data["char_10_train"] <= data["char_18"]).astype(float))).astype(float)) + np.cos((np.sin(((np.minimum( (np.tanh(np.sin(data["char_2_people"]))),  (data["char_2_people"])) > data["char_10_train"]).astype(float))) * 2.0))))) +
                   np.tanh((-(((9.869604 * 2.0) * ((data["char_8_train"] > data["char_2_people"]).astype(float)))))) +
                   np.tanh(((data["char_5_train"] < (0.333333 + ((data["char_30"] > np.cos(np.sin((data["day_people"] + (((data["char_18"] * (((((((0.0 / 2.0) + (0.0 + ((0.692308 + np.cos(np.sin(data["month_train"])))/2.0)))/2.0) >= data["month_train"]).astype(float)) >= data["month_train"]).astype(float))) <= ((data["char_30"] + ((float(2.022220 < 2.466670)) / 2.0))/2.0)).astype(float)))))).astype(float)))).astype(float))) +
                   np.tanh(((((data["char_29"] + ((data["char_10_train"] >= (data["char_4_train"] / 2.0)).astype(float)))/2.0) - np.minimum( (data["char_37"]),  ((((3.0 * (7.285710 * data["char_10_train"])) >= 0.0).astype(float))))) - np.minimum( (data["char_29"]),  ((3.0 * (data["char_29"] - ((math.cos(-1.0) <= ((np.minimum( (data["char_29"]),  (data["char_29"])) <= (data["char_10_train"] - np.minimum( (data["char_29"]),  (data["char_10_train"])))).astype(float))).astype(float)))))))) +
                   np.tanh((((np.sin(np.sin(9.869604)) + ((data["group_1"] + (np.sin(np.sin(((9.0) * (data["group_1"] - 0.855072)))) * 2.0)) * 2.0)) * 2.0) + (((np.sin(3.141593) + ((data["char_20"] + (((data["weekday_people"] >= 0.0).astype(float)) / 2.0)) * 2.0)) * 2.0) - 0.634146))) +
                   np.tanh(((-(1.492310)) + (data["char_25"] + (((0.276923 < np.maximum( (data["char_10_train"]),  ((-(2.0))))).astype(float)) + (data["char_25"] - ((np.maximum( (data["char_10_train"]),  (data["char_25"])) >= np.minimum( ((data["char_29"] * 2.0)),  ((np.cos(data["char_10_train"]) * data["char_7_train"])))).astype(float))))))) +
                   np.tanh((3.141593 * (((data["group_1"] > (2.022220 * (((((0.692308 > data["group_1"]).astype(float)) <= data["char_3_people"]).astype(float)) - ((data["group_1"] > np.cos((0.634146 / 2.0))).astype(float))))).astype(float)) - ((data["group_1"] <= np.sin((data["group_1"] * 2.0))).astype(float))))) +
                   np.tanh(((data["char_2_people"] - np.sin(np.sin(((data["char_1_people"] < data["char_29"]).astype(float))))) - (((0.83134907484054565) < np.sin(((data["char_2_people"] < ((0.333333 + np.sin((((np.minimum( (data["char_2_people"]),  ((data["char_2_people"] + (np.minimum( (data["char_2_people"]),  (data["char_2_people"])) - (float(0.0 >= 0.0)))))) - 0.692308) >= 0.0).astype(float))))/2.0)).astype(float)))).astype(float)))) +
                   np.tanh(np.sin(((data["char_38"] + (((((float(1.0 >= 1.300000)) > data["char_10_train"]).astype(float)) < ((data["char_7_train"] < ((data["char_2_people"] >= data["char_23"]).astype(float))).astype(float))).astype(float))) * 2.0))) +
                   np.tanh(((((((data["char_34"] + ((np.minimum( (data["char_34"]),  ((np.minimum( (data["char_25"]),  (data["char_25"])) + ((0.147059 >= ((0.0 + (data["char_34"] / 2.0))/2.0)).astype(float))))) > np.sin(data["char_38"])).astype(float)))/2.0) >= np.minimum( (data["char_38"]),  (data["char_38"]))).astype(float)) * 2.0) * ((0.147059 >= ((0.0 + np.minimum( (data["char_28"]),  ((0.692308 - data["month_train"]))))/2.0)).astype(float)))) +
                   np.tanh((31.006277 * np.sin(((data["char_10_train"] < ((data["char_7_people"] < np.tanh((np.minimum( (2.466670),  ((data["char_10_train"] * (data["char_25"] / 2.0)))) * 2.0))).astype(float))).astype(float))))) +
                   np.tanh(((data["char_3_people"] < (data["month_train"] * np.sin(np.tanh((data["char_29"] * 2.0))))).astype(float))) +
                   np.tanh((3.115380 * (7.285710 * np.sin(np.tanh(((np.maximum( (np.tanh(data["group_1"])),  (data["char_23"])) - ((data["group_1"] <= np.tanh(data["char_38"])).astype(float))) - np.tanh(data["char_38"]))))))) +
                   np.tanh(((data["char_38"] <= (data["char_24"] + np.sin(((-(np.minimum( (0.333333),  (((0.333333 + np.minimum( ((-(((np.maximum( ((0.333333 / 2.0)),  (data["char_38"])) > (1.732051 * data["char_1_train"])).astype(float))))),  (((0.333333 + ((-(np.minimum( (0.333333),  (((((data["char_38"] <= 0.333333).astype(float)) + 1.0)/2.0))))) / 2.0))/2.0))))/2.0))))) / 2.0)))).astype(float))) +
                   np.tanh((((data["char_38"] <= ((((data["char_5_people"] * 2.0) >= 0.0).astype(float)) + ((data["month_train"] * 2.0) - ((data["char_31"] >= ((data["char_38"] <= (np.maximum( (np.maximum( (0.634146),  (np.minimum( (((-(7.285710)) / 2.0)),  (3.0))))),  (0.636620)) / 2.0)).astype(float))).astype(float))))).astype(float)) - ((((data["char_24"] < 1.300000).astype(float)) + ((data["char_31"] <= 0.571429).astype(float)))/2.0))) +
                   np.tanh(np.minimum( (data["char_2_people"]),  ((data["char_2_people"] - (((((0.634146 > (((data["char_33"] * 2.0) - ((0.333333 <= np.cos((data["char_17"] + ((data["char_26"] > data["char_2_people"]).astype(float))))).astype(float))) * 1.0)).astype(float)) - (data["char_5_people"] * ((0.571429 <= np.cos(data["char_2_people"])).astype(float)))) <= 0.636620).astype(float)))))) +
                   np.tanh(((data["day_people"] >= data["weekend_people"]).astype(float))) +
                   np.tanh(((np.sin(((-((((1.570796 + ((1.414214 + data["group_1"])/2.0)) * 2.0) * (data["group_1"] - (((-(np.cos((data["group_1"] + ((0.318310 < np.cos(np.maximum( (-2.0),  (data["char_30"])))).astype(float)))))) * 2.0) * 2.0))))) * 2.0)) * 2.0) * 2.0)) +
                   np.tanh(np.cos(np.maximum( ((-(np.cos(1.732051)))),  (((((((np.tanh(np.minimum( (((data["char_38"] > (data["month_people"] * data["char_10_train"])).astype(float))),  (1.0))) < np.minimum( (data["char_28"]),  (data["char_38"]))).astype(float)) <= (-2.0 - (data["char_6_people"] - 3.0))).astype(float)) * 2.0) * 2.0))))) +
                   np.tanh((data["group_1"] + (((3.141593 * (data["group_1"] - ((data["group_1"] <= np.sin(((1.300000 <= data["group_1"]).astype(float)))).astype(float)))) - ((data["group_1"] <= np.sin(((((data["year_train"] + (0.855072 - np.minimum( (data["char_38"]),  (data["char_16"]))))/2.0) <= 0.318310).astype(float)))).astype(float))) * 2.0))) +
                   np.tanh(((((np.sin((((data["group_1"] > np.sin((data["group_1"] * 2.0))).astype(float)) - ((np.sin((data["group_1"] + 0.318310)) > data["group_1"]).astype(float)))) * 2.0) - data["group_1"]) * 2.0) * 2.0)) +
                   np.tanh(((((np.minimum( (0.439394),  (((np.minimum( (0.439394),  (((((float(3.141593 >= 2.022220)) * data["char_18"]) >= (0.636620 - ((data["char_38"] > (data["year_people"] + ((np.maximum( (np.cos(-2.0)),  (0.439394)) + ((float(3.141593 >= 2.022220)) * data["char_18"]))/2.0))).astype(float)))).astype(float)))) > data["year_people"]).astype(float)))) > data["year_people"]).astype(float)) * 2.0) * 2.0)))

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

    # print('Part of the group date trick')

    # datagroup = train[['group_1', 'date_train', 'outcome']].copy()
    # x = datagroup.groupby(['group_1', 'date_train'])['outcome'].mean()
    # x = x.reset_index(drop=False)
    # visibletest = pd.merge(test, x, how='left', suffixes=('', '__grpdate'),
    #                       on=['group_1', 'date_train'], left_index=True)
    # visibletest.sort_values(by='activity_id', inplace=True)
    # visibletest.reset_index(drop=True, inplace=True)
    # submission.loc[visibletest.outcome__grpdate == 0.0, 'outcome'] = 0.
    # submission.loc[visibletest.outcome__grpdate == 1.0, 'outcome'] = 1.
    # print('Saving File')
    submission.to_csv('gpsubmission.csv', index=False)
    print('Completed!')
