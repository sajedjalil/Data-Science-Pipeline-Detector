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
    grpOutcomes = grpOutcomes[grpOutcomes.cnt > 39]
    grpOutcomes.drop('cnt', inplace=True, axis=1)
    outcomes = data2['outcome'].values
    x = pd.merge(data2[[columnName, 'outcome']], grpOutcomes,
                 suffixes=('x_', ''),
                 how='left',
                 on=columnName,
                 left_index=True)['outcome']
    if(useLOO):
        x = ((x*x.shape[0])-outcomes)/(x.shape[0]-1)
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


def GPIndividual1(data):
    predictions = (np.tanh((38.0 * (((((7.0) * (((data["char_38"] * 5.888890) * data["char_6_people"]) - np.tanh((((11.80543136596679688) > np.minimum( (data["char_24"]),  (np.sin(((1.633330 + np.tanh((7.0)))/2.0))))).astype(float))))) > np.maximum( (0.302326),  (0.173913))).astype(float)) - np.tanh(np.maximum( (data["char_38"]),  (data["char_31"])))))) +
                   np.tanh(((((data["char_7_people"] * (data["char_2_people"] + (5.888890 * (((2.42440271377563477) * data["char_2_people"]) - np.cos((float(1.732051 < 1.264710))))))) - ((data["char_2_people"] < np.cos(np.sin(1.953490))).astype(float))) * 2.0) * (1.953490 - (np.cos((float(1.414214 > 31.006277))) - 1.414214)))) +
                   np.tanh(((((data["char_38"] - np.cos((((data["char_2_people"] * (38.0 + data["char_38"])) + (-3.0 * 2.0)) + np.minimum( (((data["char_31"] >= ((1.0 > (data["char_9_people"] / 2.0)).astype(float))).astype(float))),  (((data["char_9_people"] > data["char_23"]).astype(float))))))) * 2.0) * 2.0) * 2.0)) +
                   np.tanh(((1.773580 * (data["char_38"] - np.maximum( (-3.0),  ((4.238100 - (data["char_2_people"] * ((1.773580 * (data["char_38"] - (4.238100 - (data["char_2_people"] * np.maximum( ((9.05275726318359375)),  ((1.773580 - 1.125000))))))) * 2.0))))))) * 2.0)) +
                   np.tanh((3.0 * (3.033330 * (((3.033330 * ((data["char_38"] * 2.0) - data["char_17"])) * 2.0) - np.cos(0.800000))))) +
                   np.tanh((((data["char_6_people"] - ((data["char_2_people"] <= (((data["char_34"] <= (0.318310 - ((np.tanh(data["char_2_people"]) + (np.tanh((data["char_2_people"] - data["char_36"])) / 2.0))/2.0))).astype(float)) * 2.0)).astype(float))) * 9.869604) - data["char_37"])) +
                   np.tanh((5.888890 * ((data["char_2_people"] * (5.888890 * (data["char_37"] - (-(((2.0 / 2.0) * (data["char_2_people"] - ((data["char_38"] < (np.sin(1.732051) / 2.0)).astype(float))))))))) - np.cos((-(((data["char_9_people"] + (data["char_2_people"] * 2.0))/2.0))))))) +
                   np.tanh(((data["char_7_people"] * (38.0 * data["char_2_people"])) - ((9.869604 + np.minimum( ((np.cos((10.66818141937255859)) * 2.0)),  (np.sin(data["char_2_people"]))))/2.0))) +
                   np.tanh((38.0 * (-((4.238100 * np.cos((np.maximum( (np.minimum( (data["char_33"]),  (data["char_36"]))),  (data["char_9_people"])) + ((1.0 >= (((2.718282 + ((-3.0 + data["char_36"]) * 2.0))/2.0) + ((float((2.718282 + 1.633330)/2.0) <= 0.367879)))).astype(float))))))))) +
                   np.tanh(((-((((4.14308404922485352) * (np.cos(((4.14308404922485352) * (data["char_2_people"] - ((1.018520 <= (((data["char_2_people"] - data["char_6_people"]) * 2.0) - ((4.238100 <= data["char_2_people"]).astype(float)))).astype(float))))) * 2.0)) * 2.0))) + ((-((((0.511628 - data["char_2_people"]) >= data["char_2_people"]).astype(float)))) * 2.0))) +
                   np.tanh(((data["char_38"] * 2.0) - (-2.0 - (((data["char_2_people"] - np.sin((((data["char_31"] + data["char_38"]) + np.sin(data["char_2_people"])) * 2.0))) * 2.0) * (4.238100 * 2.0))))) +
                   np.tanh(((9.0) + np.minimum( (data["char_22"]),  ((((31.006277 / 2.0) * ((5.888890 * (np.tanh(data["char_38"]) - 0.511628)) - ((data["char_7_people"] <= data["char_38"]).astype(float)))) / 2.0))))) +
                   np.tanh((0.302326 - (((((((math.sin(38.0) >= data["char_2_people"]).astype(float)) - ((data["char_10_train"] + (-(((data["char_34"] >= data["char_2_people"]).astype(float)))))/2.0)) * 2.0) * 2.0) * 2.0) * 2.0))) +
                   np.tanh((((((data["char_6_people"] * np.maximum( (1.732051),  (38.0))) * (data["char_2_people"] / 2.0)) - 2.0) - (0.800000 / 2.0)) * ((7.0) + 0.800000))) +
                   np.tanh((((data["char_38"] - np.cos(((data["char_33"] < (data["char_7_people"] * 2.0)).astype(float)))) * 9.869604) * 2.0)) +
                   np.tanh((((6.82127714157104492) * 31.006277) * np.cos((((-((data["char_13"] + data["char_7_people"]))) + (1.570796 + (np.maximum( ((14.88385009765625000)),  (((data["char_8_people"] < data["char_17"]).astype(float)))) / 2.0))) * np.cos((((float(1.0 > 5.888890)) >= ((data["char_16"] + 1.633330)/2.0)).astype(float))))))) +
                   np.tanh((((31.006277 + ((0.750000 < np.minimum( (data["char_2_people"]),  (9.869604))).astype(float)))/2.0) * (np.sin((1.018520 * (data["char_2_people"] - np.minimum( ((((float(6.0) >= 0.302326)) + ((-1.0 < np.sin(((1.414214 + data["char_34"])/2.0))).astype(float)))),  ((((((data["char_16"] <= (data["char_9_people"] - data["char_2_people"])).astype(float)) >= data["date_people"]).astype(float)) + 0.302326)))))) * 2.0))) +
                   np.tanh(((((2.0 * ((np.minimum( (np.sin(((data["char_28"] + data["char_9_people"])/2.0))),  (data["char_7_people"])) - ((0.511628 * 2.0) / 2.0)) * 2.0)) * 2.0) * 2.0) * 2.0)) +
                   np.tanh((5.888890 - (31.006277 * np.tanh(((data["char_38"] <= ((0.511628 + (((data["char_25"] >= ((data["char_38"] + np.sin(np.cos(((data["char_38"] <= (np.maximum( (np.cos((data["date_people"] - 1.125000))),  (np.maximum( (np.minimum( (data["char_38"]),  ((1.732051 + ((np.sin(1.018520) > 0.0).astype(float)))))),  (1.264710)))) / 2.0)).astype(float)))))/2.0)).astype(float)) * 2.0))/2.0)).astype(float)))))) +
                   np.tanh(((1.570796 - 31.006277) * (1.0 + (0.588889 - (data["char_38"] + (((-((data["char_6_people"] * 2.0))) < np.cos(np.minimum( (-2.0),  (data["char_11"])))).astype(float))))))) +
                   np.tanh((np.maximum( (38.0),  ((-(data["date_people"])))) * (np.tanh(data["char_2_people"]) - (((data["char_2_people"] * np.tanh(data["char_18"])) <= (((data["char_18"] - data["char_18"]) >= ((data["char_22"] >= ((0.302326 + (data["char_2_people"] - (np.maximum( (data["char_18"]),  (np.sin(data["char_38"]))) / 2.0)))/2.0)).astype(float))).astype(float))).astype(float))))) +
                   np.tanh((4.238100 * (((-((np.cos(data["char_22"]) - np.tanh((data["char_7_people"] * np.cos((((data["char_20"] > ((data["char_21"] + data["char_20"])/2.0)).astype(float)) * data["char_22"]))))))) * 2.0) + np.minimum( (data["char_34"]),  (5.888890))))) +
                   np.tanh(((1.953490 + 9.869604) * ((((((np.maximum( (((14.37075042724609375) * data["char_38"])),  ((np.cos(0.574713) * 2.0))) * 2.0) + 0.0)/2.0) + (np.tanh(3.033330) + 0.588889))/2.0) * (data["char_38"] - ((0.173913 * ((((np.tanh(data["char_6_people"]) <= data["char_19"]).astype(float)) + data["char_33"])/2.0)) + 0.588889))))) +
                   np.tanh((31.006277 * np.minimum( ((0.511628 - np.minimum( (1.953490),  (((data["char_2_people"] <= ((((data["char_7_people"] - np.minimum( (1.953490),  (((data["char_2_people"] <= ((1.125000 <= ((data["char_14"] <= data["char_2_people"]).astype(float))).astype(float))).astype(float))))) * 2.0) <= data["char_2_people"]).astype(float))).astype(float)))))),  (38.0)))) +
                   np.tanh((((data["char_2_people"] + (((9.869604 * (data["char_2_people"] - data["char_20"])) + (data["char_2_people"] - np.maximum( ((data["char_23"] + (-(np.maximum( (1.732051),  (3.141593)))))),  (((data["char_14"] <= np.tanh(((((np.minimum( (data["char_2_people"]),  (0.750000)) > data["char_2_people"]).astype(float)) >= (data["char_8_people"] - 0.511628)).astype(float)))).astype(float)))))) * 2.0)) * 2.0) * 2.0)) +
                   np.tanh(((np.sin((np.minimum( (((0.511628 >= data["char_2_people"]).astype(float))),  ((-((np.minimum( (1.773580),  (((np.cos(((0.636620 > np.sin(((0.588889 * np.cos(data["char_21"])) + (data["char_6_people"] / 2.0)))).astype(float))) + (data["char_38"] - (((1.732051 * 2.0) < ((np.tanh((2.51669239997863770)) < 0.636620).astype(float))).astype(float))))/2.0))) * 2.0))))) * 2.0)) * 2.0) * 2.0)) +
                   np.tanh(((31.006277 * np.sin(((0.574713 - (9.869604 * (data["char_38"] + (3.141593 + 0.367879)))) + np.sin((1.0 + np.sin(((3.033330 + (((((0.750000 >= data["char_38"]).astype(float)) + data["char_27"])/2.0) + ((data["char_15"] * 31.006277) / 2.0)))/2.0))))))) * 2.0)) +
                   np.tanh((((1.953490 + (float(0.318310 <= 1.018520)))/2.0) + (((((data["char_38"] - ((np.cos((data["char_38"] * 1.732051)) >= data["char_7_people"]).astype(float))) * 2.0) - 0.800000) * 2.0) * 2.0))) +
                   np.tanh((-((((3.0 - 0.173913) * (((((((data["char_37"] < data["char_7_people"]).astype(float)) + (((data["char_18"] + data["char_38"]) < 0.173913).astype(float)))/2.0) + data["char_38"]) <= np.tanh((3.0 + 38.0))).astype(float))) * 2.0)))) +
                   np.tanh((38.0 * np.sin((data["char_2_people"] - ((4.238100 - np.maximum( ((-((-(data["char_22"]))))),  (((data["char_8_people"] >= ((data["char_2_people"] - (data["char_2_people"] - data["char_33"])) - ((data["char_2_people"] - data["char_37"]) - ((0.173913 > data["char_6_people"]).astype(float))))).astype(float))))) * 2.0))))) +
                   np.tanh(((np.minimum( (1.0),  (2.718282)) - (((31.006277 * ((data["char_17"] >= np.minimum( (data["char_2_people"]),  (np.maximum( ((-((38.0 + 1.633330)))),  (data["char_10_train"]))))).astype(float))) - (np.maximum( (9.869604),  (data["char_35"])) * data["char_2_people"])) + np.cos(data["char_17"]))) - ((np.tanh(data["char_2_people"]) + np.tanh(data["char_34"]))/2.0))) +
                   np.tanh((31.006277 * ((data["char_23"] + (data["char_38"] - (float(2.718282 > 1.633330)))) - ((0.302326 / 2.0) + (((np.minimum( (data["char_18"]),  (((((data["date_people"] <= data["char_17"]).astype(float)) + np.sin(np.maximum( (0.367879),  (((data["char_38"] <= np.minimum( (data["char_38"]),  (np.tanh(0.511628)))).astype(float))))))/2.0))) > data["char_15"]).astype(float)) * 2.0))))))

    return Outputs(predictions*.1)


def GPIndividual2(data):
    predictions = (np.tanh((60.0 * np.sin((0.950617 * (data["char_2_people"] - (np.cos(((data["date_people"] > ((data["char_2_people"] <= np.sin((data["char_33"] - (((((data["char_2_people"] * 2.0) <= 1.054050).astype(float)) > (-2.0 - 8.666670)).astype(float))))).astype(float))).astype(float))) * data["char_20"])))))) +
                   np.tanh(((-(2.718282)) * ((0.870968 + (data["char_33"] - ((np.minimum( (1.0),  (((data["char_32"] + data["char_6_people"])/2.0))) * (data["char_2_people"] * ((31.006277 + np.minimum( (31.006277),  ((-(data["char_2_people"])))))/2.0))) / 2.0))) * 2.0))) +
                   np.tanh((data["char_2_people"] - ((5.0) + (-((0.870968 + (data["char_27"] + ((8.666670 * (((8.666670 - (np.maximum( (0.705882),  (1.570796)) * data["char_2_people"])) * np.minimum( (data["char_2_people"]),  (data["char_7_people"]))) - 0.621622)) * 2.0)))))))) +
                   np.tanh((data["char_28"] - ((np.tanh(9.869604) - data["char_10_train"]) + (((0.597015 - ((((data["char_31"] + np.sin(np.maximum( (data["char_7_people"]),  (-2.0))))/2.0) > (0.950617 + (-(data["char_9_people"])))).astype(float))) * 2.0) * 2.0)))) +
                   np.tanh(((((31.006277 * ((4.0) * (data["char_7_people"] - ((data["char_2_people"] <= (data["char_19"] * np.cos(np.cos(((((data["char_33"] < (9.869604 * 2.0)).astype(float)) > data["char_38"]).astype(float)))))).astype(float))))) - 2.724140) * data["char_2_people"]) + (-((6.80385351181030273))))) +
                   np.tanh((8.666670 * (np.minimum( (3.0),  (np.sin((data["char_38"] + np.sin((data["char_2_people"] - np.cos(data["char_6_people"]))))))) * 2.0))) +
                   np.tanh((data["char_38"] - (((np.tanh(np.sin((((data["char_38"] - (((-(np.cos(data["char_38"]))) < 2.718282).astype(float))) + np.minimum( (-3.0),  ((data["char_36"] + 1.0)))) + data["char_6_people"]))) * 2.0) * 2.0) * (np.tanh(np.cos(np.sin(data["char_13"]))) + (31.006277 - np.cos(data["char_12"])))))) +
                   np.tanh(((31.006277 * ((((data["char_2_people"] * 2.0) - data["date_people"]) * 2.0) - np.sin(((np.maximum( (60.0),  ((-(0.621622)))) > (0.621622 - 0.259259)).astype(float))))) * 2.0)) +
                   np.tanh(((data["char_8_people"] - (np.sin(((60.0 * ((data["char_17"] > np.tanh(((data["char_19"] < data["char_2_people"]).astype(float)))).astype(float))) - 8.666670)) * 2.0)) + ((((data["char_8_people"] - ((0.621622 >= np.sin(3.0)).astype(float))) + (((data["char_38"] - np.sin(np.tanh(((data["char_2_people"] <= data["char_11"]).astype(float))))) * 2.0) * 2.0)) * 2.0) * 2.0))) +
                   np.tanh(((((60.0 * (data["char_6_people"] - (1.054050 - np.minimum( (data["char_8_people"]),  (np.tanh(60.0)))))) * 2.0) + (-3.0 * ((data["char_19"] < data["char_11"]).astype(float)))) * 2.0)) +
                   np.tanh(((((((data["char_34"] > 0.857143).astype(float)) - (np.minimum( (0.950617),  (1.414214)) - (((((np.minimum( (data["char_7_people"]),  ((9.869604 / 2.0))) - ((data["char_2_people"] <= data["char_32"]).astype(float))) * 2.0) * 2.0) * 2.0) * 2.0))) * 2.0) - ((1.570796 + np.sin(0.0))/2.0)) * 2.0)) +
                   np.tanh((-(((data["char_12"] + (60.0 * np.tanh((3.0 * (((-(data["char_6_people"])) + np.cos(((data["char_24"] < (((data["char_36"] * 0.826087) >= ((1.0 - data["char_19"]) / 2.0)).astype(float))).astype(float))))/2.0)))))/2.0)))) +
                   np.tanh((((((((1.0 + data["char_10_train"]) <= 3.0).astype(float)) + (data["char_10_people"] + (31.006277 / 2.0))) * data["char_34"]) * (((data["char_15"] + (31.006277 / 2.0)) * data["char_9_people"]) - np.maximum( ((8.39196205139160156)),  ((8.39196205139160156))))) - np.maximum( ((8.39196205139160156)),  ((data["char_10_people"] + ((-(data["char_21"])) / 2.0)))))) +
                   np.tanh((60.0 * (data["char_38"] - np.cos(((31.006277 * data["char_2_people"]) - np.cos(((1.414214 >= (31.006277 * data["char_2_people"])).astype(float)))))))) +
                   np.tanh(((6.0) * (((1.74467003345489502) * (((0.0 + ((data["char_38"] - (1.032970 - np.minimum( (0.820000),  (np.sin(data["char_2_people"]))))) * 2.0)) - (np.cos((np.maximum( (data["char_2_people"]),  ((data["char_7_people"] + data["char_2_people"]))) * 2.0)) - data["char_23"])) * np.cos(data["char_7_people"]))) - ((data["char_2_people"] < np.sin(0.259259)).astype(float))))) +
                   np.tanh((31.006277 * ((np.tanh(np.minimum( (data["char_31"]),  (np.minimum( (data["char_2_people"]),  (data["char_2_people"]))))) + data["char_38"]) - (((0.636620 < ((2.718282 >= np.minimum( (data["char_16"]),  (np.minimum( (((0.589744 < np.cos(1.0)).astype(float))),  (((data["char_16"] >= data["char_2_people"]).astype(float))))))).astype(float))).astype(float)) - (data["char_22"] / 2.0))))) +
                   np.tanh((1.013330 + (((1.0 - ((((float(np.tanh(1.032970) >= 1.013330)) >= ((0.636620 < ((data["char_6_people"] + (data["char_19"] + (((((data["char_9_people"] <= (float((13.42087173461914062) <= 0.636620))).astype(float)) * -2.0) * 2.0) * 2.0)))/2.0)).astype(float))).astype(float)) * 2.0)) * 2.0) * 2.0))) +
                   np.tanh((np.maximum( ((data["char_2_people"] / 2.0)),  (60.0)) * ((data["char_2_people"] - (((data["char_37"] >= np.tanh(data["char_2_people"])).astype(float)) / 2.0)) / 2.0))) +
                   np.tanh((8.666670 * ((9.869604 * np.minimum( (data["char_34"]),  (((data["char_38"] - (((data["char_25"] * ((0.589744 < (1.570796 + np.minimum( (1.414214),  (np.cos((1.49743115901947021)))))).astype(float))) / 2.0) * 2.0)) * 2.0)))) - ((data["char_18"] + np.sin(((-3.0 + (data["char_38"] * 2.0))/2.0)))/2.0)))) +
                   np.tanh((((9.869604 * 2.0) + (float(0.0 <= (2.0 / 2.0)))) * (data["char_2_people"] - ((((data["date_people"] > np.cos(np.sin((data["char_19"] - (10.13461208343505859))))).astype(float)) + np.sin(np.cos(np.sin((-(((((0.02423644624650478) <= (0.950617 * (np.minimum( (0.820000),  (data["char_31"])) / 2.0))).astype(float)) * 0.622222)))))))/2.0)))) +
                   np.tanh((np.maximum( (3.141593),  (31.006277)) * (np.maximum( (3.141593),  (0.200000)) * ((data["char_7_people"] + ((2.0 * (data["char_38"] + ((-1.0 + ((data["char_2_people"] * 2.0) - (((data["char_2_people"] * 2.0) > np.cos(np.minimum( ((6.61050224304199219)),  ((data["char_7_people"] + (data["char_38"] * 2.0)))))).astype(float))))/2.0))) + -1.0))/2.0)))) +
                   np.tanh((-1.0 + ((7.30152225494384766) * (data["char_22"] + (-1.0 + (-((0.857143 - (data["char_2_people"] * (((5.0) + ((data["date_people"] + (((np.tanh((((1.032970 - 0.705882) + ((np.cos(((0.0 <= data["char_2_people"]).astype(float))) * 2.0) / 2.0)) * 2.0)) / 2.0) >= data["char_25"]).astype(float))) * 2.0))/2.0)))))))))) +
                   np.tanh((9.869604 * (data["char_10_train"] - ((data["char_38"] < ((data["char_20"] + 0.857143)/2.0)).astype(float))))) +
                   np.tanh((8.666670 * (((6.74188041687011719) * ((data["char_6_people"] * np.tanh(1.732051)) * data["char_2_people"])) - 0.950617))) +
                   np.tanh(((60.0 * ((31.006277 * np.tanh(((data["char_7_people"] + (data["char_2_people"] - ((data["char_23"] > (((data["char_38"] - np.sin(data["char_23"])) * 2.0) / 2.0)).astype(float))))/2.0))) - (float(60.0 >= 60.0)))) / 2.0)) +
                   np.tanh(((2.0 + 3.0) * ((3.141593 * (((data["char_38"] * ((data["char_38"] * (data["char_38"] + (((float(31.006277 > 0.597015)) + 3.141593)/2.0))) * 2.0)) * 2.0) - (1.054050 * (data["char_12"] * 8.666670)))) - (1.054050 * (1.054050 + 3.0))))) +
                   np.tanh(((1.032970 - (((data["char_6_people"] < ((data["char_37"] + ((data["char_9_people"] <= ((data["char_2_people"] < 0.597015).astype(float))).astype(float))) / 2.0)).astype(float)) * 2.0)) * 2.0)) +
                   np.tanh((((data["char_38"] - ((data["char_14"] >= (data["char_38"] * np.cos(((data["char_19"] >= (((-(31.006277)) >= data["char_16"]).astype(float))).astype(float))))).astype(float))) * 2.0) * 2.0)) +
                   np.tanh(((-2.0 - ((data["char_2_people"] - np.maximum( (60.0),  (9.869604))) * np.sin((((data["char_2_people"] + (np.minimum( (data["char_2_people"]),  (data["char_7_people"])) - ((np.cos(data["char_2_people"]) < (data["char_34"] - ((8.666670 - 31.006277) * 2.0))).astype(float)))) * 2.0) * 0.820000)))) * 2.0)) +
                   np.tanh((1.570796 - (((((data["char_38"] < np.tanh((np.minimum( (data["char_20"]),  (2.724140)) * 2.0))).astype(float)) * 2.0) * 2.0) * 2.0))) +
                   np.tanh((3.0 - (9.869604 * np.maximum( (((np.sin(np.cos(data["date_people"])) >= data["char_38"]).astype(float))),  (np.minimum( (((0.950617 > ((math.cos((0.0)) < data["char_9_people"]).astype(float))).astype(float))),  (((((data["char_25"] > (0.200000 - np.maximum( (60.0),  ((1.032970 - np.maximum( (data["char_38"]),  (0.636620))))))).astype(float)) < data["char_37"]).astype(float))))))))) +
                   np.tanh((-((60.0 * (((((data["char_2_people"] - (((data["char_36"] < (1.570796 * 0.0)).astype(float)) / 2.0)) >= ((data["char_2_people"] - (((data["char_7_people"] < (data["char_15"] - ((0.705882 <= data["char_8_people"]).astype(float)))).astype(float)) / 2.0)) * 2.0)).astype(float)) >= np.tanh(((data["char_2_people"] - (0.597015 / 2.0)) * 2.0))).astype(float)))))))

    return Outputs(predictions*.1)


def GPIndividual3(data):
    predictions = (np.tanh(((9.15296173095703125) * ((np.minimum( (((min( ((math.cos(1.342110) * 2.0)),  (0.176471)) <= data["char_6_people"]).astype(float))),  ((data["char_2_people"] - (np.sin((13.750000 * 2.0)) / 2.0)))) / 2.0) * 2.0))) +
                   np.tanh((13.750000 * (data["char_7_people"] - ((2.718282 + ((np.minimum( (np.tanh((-(np.maximum( (data["char_27"]),  ((4.97198677062988281))))))),  (np.tanh((0.542373 * ((7.444440 >= ((data["char_32"] > 0.318310).astype(float))).astype(float)))))) * (data["char_17"] * 2.0)) - (data["char_9_people"] * 2.0)))/2.0)))) +
                   np.tanh(((np.minimum( ((((2.67980289459228516) + (data["char_38"] - 0.318310))/2.0)),  (((np.minimum( ((data["char_38"] - 0.318310)),  (2.722220)) > ((data["char_32"] - np.tanh(((data["char_19"] + np.cos(31.006277))/2.0))) / 2.0)).astype(float)))) - 0.318310) * (13.750000 + ((3.600000 * 2.0) * 2.0)))) +
                   np.tanh(((13.750000 * ((data["char_2_people"] * data["char_2_people"]) - ((data["char_2_people"] <= (0.176471 * 3.141593)).astype(float)))) - ((((-((10.86678314208984375))) * 2.0) >= data["char_10_people"]).astype(float)))) +
                   np.tanh(((1.069770 * ((9.0) * (0.542373 + (-(((data["char_9_people"] >= (31.006277 * (data["char_7_people"] + (data["char_38"] + (-(((1.208330 + data["char_31"])/2.0))))))).astype(float))))))) * 2.0)) +
                   np.tanh(((data["char_22"] + ((9.869604 * (np.tanh(((data["char_9_people"] - np.cos(((-(data["char_10_train"])) * 2.0))) * 2.0)) * 2.0)) / 2.0)) * 2.0)) +
                   np.tanh((((data["char_38"] - data["char_25"]) * (13.750000 - ((data["char_38"] - (float(13.750000 <= 0.542373))) * (float(0.630952 > 1.732051))))) + (3.600000 * data["char_38"]))) +
                   np.tanh((31.006277 * (((((((np.cos(data["char_2_people"]) <= np.sin(((data["date_people"] > (float(0.630952 >= 2.0))).astype(float)))).astype(float)) / 2.0) + (data["char_2_people"] / 2.0))/2.0) - (0.931035 - np.sin(data["char_2_people"]))) * 2.0))) +
                   np.tanh((1.382350 - (31.006277 * np.sin(np.cos(np.maximum( (((data["char_33"] + data["date_people"])/2.0)),  (np.maximum( (data["char_12"]),  ((np.tanh((np.minimum( (data["char_22"]),  (data["char_6_people"])) * 2.0)) * 2.0)))))))))) +
                   np.tanh(((np.cos(np.sin(np.tanh(data["char_8_people"]))) * 2.0) - (31.006277 * (np.tanh(3.600000) - ((np.tanh(1.844440) - (31.006277 * (np.tanh(1.844440) - (data["char_38"] + data["char_7_people"])))) + data["char_7_people"]))))) +
                   np.tanh((13.750000 * ((((0.318310 - (1.0 / 2.0)) + (np.minimum( (np.maximum( (data["date_people"]),  (((0.542373 < data["char_38"]).astype(float))))),  (((-(np.tanh(np.tanh((0.630952 + (-(((data["char_2_people"] > ((0.636620 >= (np.minimum( (data["char_12"]),  (13.750000)) * 2.0)).astype(float))).astype(float))))))))) * 2.0))) * 2.0))/2.0) * 2.0))) +
                   np.tanh((31.006277 * (data["char_38"] - (((((0.367879 + 0.318310) >= data["char_2_people"]).astype(float)) >= ((-((-(data["char_6_people"])))) - (-((-((((data["char_7_people"] + data["char_14"])/2.0) / 2.0))))))).astype(float))))) +
                   np.tanh((13.750000 * (data["char_38"] - (((data["char_38"] <= (-((data["char_38"] - np.cos(data["char_7_people"]))))).astype(float)) * ((1.844440 > data["char_38"]).astype(float)))))) +
                   np.tanh(((9.0) * (data["char_38"] + ((-(((data["char_2_people"] <= (((0.194444 > ((data["char_7_people"] * 2.0) - (1.414214 - (data["char_2_people"] + data["char_38"])))).astype(float)) * ((31.006277 - ((0.0 * 0.318310) / 2.0)) / 2.0))).astype(float)))) * (data["char_32"] - np.sin((-(((1.382350 > data["char_38"]).astype(float)))))))))) +
                   np.tanh((data["char_32"] - ((np.cos((data["char_2_people"] * 31.006277)) * 7.444440) * 2.0))) +
                   np.tanh((((0.542373 + 9.869604)/2.0) * (((np.maximum( (data["char_2_people"]),  (2.718282)) * (data["char_2_people"] * ((data["char_6_people"] + np.sin(2.722220)) / 2.0))) - data["char_14"]) * 2.0))) +
                   np.tanh((-((3.0 - (((data["char_2_people"] <= 2.722220).astype(float)) + ((((31.006277 + (float(31.006277 <= (float(1.208330 < 3.141593))))) * (-(np.cos(((data["char_2_people"] * 1.069770) * 3.600000))))) - 13.750000) * data["char_7_people"])))))) +
                   np.tanh((31.006277 * ((data["char_38"] - np.tanh(((2.722220 + np.cos(0.176471))/2.0))) + data["char_6_people"]))) +
                   np.tanh(((data["char_38"] * 2.0) + ((((data["char_38"] * 2.0) - (0.989583 + ((np.minimum( (data["char_7_people"]),  (data["char_21"])) < ((data["char_37"] + (data["char_31"] - (data["char_9_people"] * (((data["char_38"] * 2.0) - (((data["char_2_people"] * 2.0) >= ((0.0 + -2.0)/2.0)).astype(float))) * 2.0))))/2.0)).astype(float)))) * 2.0) * 2.0))) +
                   np.tanh((((data["char_2_people"] - (0.363636 + (-((-(np.maximum( (1.570796),  (((data["char_31"] > np.sin((0.0 + 7.444440))).astype(float)))))))))) - (3.600000 + (((((0.367879 + data["char_10_train"])/2.0) - (np.sin(((data["char_2_people"] < data["char_31"]).astype(float))) * 2.0)) * 2.0) * ((-(9.869604)) - 1.414214)))) * 2.0)) +
                   np.tanh((((-(((10.0) * ((0.176471 > (data["char_2_people"] * (((data["char_7_people"] >= np.maximum( (0.318310),  (-2.0))).astype(float)) * 2.0))).astype(float))))) * data["char_16"]) + (-3.0 * np.cos((data["char_2_people"] * (data["char_16"] - (((0.0 + -3.0)/2.0) + (np.maximum( ((8.0)),  (data["char_2_people"])) * 2.0)))))))) +
                   np.tanh((7.444440 * np.minimum( ((7.444440 - (-(np.cos(data["char_24"]))))),  ((data["char_10_train"] - np.cos(((0.676056 < data["char_8_people"]).astype(float)))))))) +
                   np.tanh(((-(((np.maximum( (13.750000),  (-3.0)) * (0.636620 - ((np.tanh(1.208330) + ((data["char_2_people"] * data["char_10_train"]) * 2.0))/2.0))) * 2.0))) * 2.0)) +
                   np.tanh((np.minimum( ((8.0)),  ((((((((data["char_38"] * 2.0) - (np.sin(np.cos(7.444440)) + 0.989583)) + data["char_7_people"])/2.0) >= (data["char_35"] / 2.0)).astype(float)) + (((data["char_38"] - (((3.0 + ((0.0 + data["char_38"])/2.0))/2.0) / 2.0)) * 2.0) * 2.0)))) * 2.0)) +
                   np.tanh((7.444440 * (np.maximum( (data["char_38"]),  (data["char_2_people"])) - ((((data["char_17"] >= np.minimum( (data["char_2_people"]),  (np.maximum( (data["char_7_people"]),  (((np.maximum( (-2.0),  ((9.869604 + np.maximum( (data["char_12"]),  (2.0))))) < data["char_10_people"]).astype(float))))))).astype(float)) * 2.0) * 2.0)))) +
                   np.tanh(((((11.51005172729492188) * 2.0) * np.maximum( ((data["char_38"] - ((13.750000 + 0.0)/2.0))),  (((np.maximum( (data["char_35"]),  (np.tanh(data["char_2_people"]))) < ((0.676056 <= data["char_38"]).astype(float))).astype(float))))) - 13.750000)) +
                   np.tanh(((data["char_2_people"] - np.minimum( ((float(0.630952 < (3.600000 * 0.630952)))),  (((0.630952 <= ((data["char_2_people"] <= ((0.367879 > data["char_7_people"]).astype(float))).astype(float))).astype(float))))) * 13.750000)) +
                   np.tanh(np.minimum( ((3.64932155609130859)),  (((1.414214 + np.maximum( ((8.66946983337402344)),  (0.046154))) * ((1.414214 * (data["char_2_people"] * 2.0)) - np.minimum( ((np.maximum( (0.0),  ((((1.0) - ((data["char_38"] + ((data["char_7_people"] * 2.0) * np.minimum( (data["char_2_people"]),  ((0.542373 + data["char_12"])))))/2.0)) * 2.0))) * 2.0)),  ((1.414214 + data["char_18"])))))))) +
                   np.tanh((((31.006277 + np.minimum( (data["char_7_people"]),  (np.sin((-(((data["char_2_people"] <= (((9.869604 + 31.006277) <= data["date_people"]).astype(float))).astype(float))))))))/2.0) * ((data["char_38"] - (((data["char_20"] + 0.989583)/2.0) + ((data["char_11"] > (data["char_9_people"] + (data["char_9_people"] + np.minimum( (2.722220),  (np.sin((data["char_2_people"] - data["char_10_people"]))))))).astype(float)))) * 2.0))) +
                   np.tanh(((5.01674795150756836) * (((data["char_2_people"] - 0.046154) - ((data["char_2_people"] <= np.maximum( (((0.194444 <= np.sin(((((data["char_7_people"] < 1.570796).astype(float)) - (9.869604 * (-1.0 * ((data["char_2_people"] * (data["char_7_people"] * ((8.93964385986328125) + (5.01674795150756836)))) * 2.0)))) - (14.10326576232910156)))).astype(float))),  (np.minimum( (0.636620),  (data["char_2_people"]))))).astype(float))) / 2.0))) +
                   np.tanh(((np.cos(data["char_10_train"]) - ((0.0 > np.minimum( (data["char_10_train"]),  (np.cos((data["char_38"] * 2.0))))).astype(float))) * (-(((31.006277 + 13.750000)/2.0))))) +
                   np.tanh(((data["char_38"] - (9.869604 * (((1.208330 >= np.tanh((data["char_20"] * (-(((1.844440 <= (7.444440 - (data["char_10_people"] * 2.0))).astype(float))))))).astype(float)) + np.minimum( (((data["date_people"] / 2.0) * 2.0)),  (((((data["char_14"] - data["char_2_people"]) * 2.0) * 2.0) * 2.0)))))) * 2.0)))

    return Outputs(predictions*.1)


if __name__ == "__main__":
    print('Started!')
    print('Group_1 is for wimps!')
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

    subset = ['char_2_people', 'char_38', 'char_8_people', 'char_9_people',
              'char_7_people', 'char_34', 'char_13', 'char_36', 'char_22',
              'char_37',
              'char_25', 'char_21', 'char_17', 'char_32', 'char_28', 'char_19',
              'char_23', 'char_16', 'char_15', 'char_10_people', 'char_20',
              'char_6_people', 'char_10_train', 'char_11', 'char_31',
              'char_14',
              'char_24', 'char_27', 'char_18', 'char_12', 'date_people',
              'char_35',
              'char_33']

    print('LOO train')
    lootrain = pd.DataFrame()
    for col in subset:
            lootrain[col] = LeaveOneOut(train, train, col, True).values

    traininv1 = GPIndividual1(lootrain)
    traininv2 = GPIndividual2(lootrain)
    traininv3 = GPIndividual3(lootrain)

    print('ROC1:', roc_auc_score(train.outcome.values, traininv1))
    print('ROC2:', roc_auc_score(train.outcome.values, traininv2))
    print('ROC3:', roc_auc_score(train.outcome.values, traininv3))
    ari = (traininv1 +
           traininv2 +
           traininv3)/3.

    print('ROC:', roc_auc_score(train.outcome.values, ari))

    print('Part of the group date trick plus char_38 heuristic')

    datagroup = train[['group_1', 'date_train', 'outcome']].copy()
    x = datagroup.groupby(['group_1', 'date_train'])['outcome'].mean()
    x = x.reset_index(drop=False)
    visibletest = pd.merge(train, x, how='left', suffixes=('', '__grpdate'),
                           on=['group_1', 'date_train'], left_index=True)
    visibletest.sort_values(by='activity_id', inplace=True)
    visibletest.reset_index(drop=True, inplace=True)
    ari[visibletest.outcome__grpdate == 0.0] = 0.
    ari[visibletest.outcome__grpdate == 1.0] = 1.
    ari[train.char_38.values < 40] = 0.  # Risky!!
    print('ROC & Heuristics:', roc_auc_score(train.outcome.values, ari))

    print('LOO test')
    lootest = pd.DataFrame()
    for col in subset:
        lootest[col] = LeaveOneOut(train, test, col, False).values

    testinv1 = GPIndividual1(lootest)
    testinv2 = GPIndividual2(lootest)
    testinv3 = GPIndividual3(lootest)
    ari = (testinv1 +
           testinv2 +
           testinv3)/3.
    submission = pd.DataFrame({'activity_id': test.activity_id.values,
                               'outcome': ari.values})
    submission.sort_values(by='activity_id', inplace=True)
    submission.reset_index(drop=True, inplace=True)

    print('Part of the group date trick plus char_38 heuristic')

    datagroup = train[['group_1', 'date_train', 'outcome']].copy()
    x = datagroup.groupby(['group_1', 'date_train'])['outcome'].mean()
    x = x.reset_index(drop=False)
    visibletest = pd.merge(test, x, how='left', suffixes=('', '__grpdate'),
                           on=['group_1', 'date_train'], left_index=True)
    visibletest.sort_values(by='activity_id', inplace=True)
    visibletest.reset_index(drop=True, inplace=True)
    submission.loc[visibletest.outcome__grpdate == 0.0, 'outcome'] = 0.
    submission.loc[visibletest.outcome__grpdate == 1.0, 'outcome'] = 1.
    submission.loc[test.char_38.values < 40, 'outcome'] = 0.  # Risky!!

    print('Saving File')

    submission.to_csv('gpsubmission.csv', index=False)

    print('Completed!')
