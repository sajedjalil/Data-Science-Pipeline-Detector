import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler


def Outputs(data):
    return 1.-(1./(1.+np.exp(-data)))


def GPIndividual1(data):
    predictions = (((data["action_type_Jump Shot"] + np.maximum( ((data["action_type_Jump Shot"] + data["action_type_Layup Shot"])),  (((((data["action_type_Jump Shot"] + data["action_type_Tip Shot"])/2.0) + np.maximum( (((data["last_moments_1"] + (data["action_type_Jump Shot"] + data["action_type_Jump Shot"]))/2.0)),  ((-(data["shot_zone_basic_Restricted Area"])))))/2.0))))/2.0) +
                    (((-((((data["action_type_Slam Dunk Shot"] - data["distance_45"]) + ((np.maximum( (data["action_type_Driving Dunk Shot"]),  (data["action_type_Alley Oop Dunk Shot"])) + data["action_type_Driving Dunk Shot"])/2.0))/2.0))) + ((data["season_2015-16"] > (-(data["season_2015-16"]))).astype(float)))/2.0) +
                    ((((((data["action_type_Hook Shot"] >= (-(data["distance_29"]))).astype(float)) >= np.cos(data["action_type_Reverse Layup Shot"])).astype(float)) + (-(((data["action_type_Hook Shot"] < np.maximum( (np.maximum( (np.maximum( (data["action_type_Running Jump Shot"]),  (data["distance_1"]))),  (data["action_type_Pullup Jump shot"]))),  (data["action_type_Jump Bank Shot"]))).astype(float)))))/2.0) +
                    (0.058823 * (data["period_4"] - ((data["action_type_Reverse Dunk Shot"] + ((data["action_type_Running Hook Shot"] + data["action_type_Driving Slam Dunk Shot"]) - data["distance_31"])) + (data["shot_zone_range_16-24 ft."] + data["shot_zone_area_Right Side Center(RC)"])))) +
                    (((data["away_False"] * ((data["action_type_Dunk Shot"] > data["action_type_Fadeaway Bank shot"]).astype(float))) + ((data["away_False"] <= (data["action_type_Finger Roll Shot"] + (-((((data["away_False"] * data["season_2014-15"]) > (data["action_type_Fadeaway Jump Shot"] + data["action_type_Turnaround Jump Shot"])).astype(float)))))).astype(float)))/2.0) +
                    ((data["action_type_Fadeaway Bank shot"] * data["action_type_Turnaround Bank shot"]) + ((data["distance_40"] - ((data["action_type_Driving Finger Roll Layup Shot"] >= (-(np.maximum( (data["action_type_Dunk"]),  (data["action_type_Running Bank shot"]))))).astype(float))) + (data["action_type_Step Back Jump shot"] * ((data["minutes_remaining_7"] >= data["action_type_Fadeaway Bank shot"]).astype(float))))) +
                    np.tanh((data["action_type_Driving Jump shot"] * (data["opponent_NYK"] - (data["shot_zone_area_Left Side(L)"] - (data["last_moments_0"] - np.maximum( (data["season_1997-98"]),  (np.maximum( (np.maximum( (data["opponent_OKC"]),  (data["distance_28"]))),  (data["minutes_remaining_0"]))))))))) +
                    ((((data["action_type_Jump Shot"] >= ((data["action_type_Driving Finger Roll Shot"] >= data["shot_zone_range_Less Than 8 ft."]).astype(float))).astype(float)) + (data["action_type_Jump Shot"] * ((data["opponent_PHI"] >= (((data["shot_zone_range_Less Than 8 ft."] != data["action_type_Jump Shot"]).astype(float)) * (data["action_type_Driving Slam Dunk Shot"] * data["action_type_Driving Finger Roll Shot"]))).astype(float))))/2.0) +
                    (data["distance_42"] + (((np.maximum( (np.minimum( (data["action_type_Layup Shot"]),  (np.sinh(data["opponent_PHI"])))),  (data["distance_36"])) + np.minimum( (np.sin(((data["opponent_PHI"] < data["away_True"]).astype(float)))),  (((data["distance_41"] < data["action_type_Running Jump Shot"]).astype(float)))))/2.0) + data["distance_41"])) +
                    ((((data["shot_zone_basic_In The Paint (Non-RA)"] - (data["action_type_Hook Shot"] - data["distance_23"])) - data["distance_39"]) - data["opponent_HOU"]) * np.minimum( (((((data["action_type_Hook Shot"] >= data["opponent_BKN"]).astype(float)) >= data["minutes_remaining_7"]).astype(float))),  (data["opponent_BKN"]))))
                   
    return Outputs(predictions)


def GPIndividual2(data):
    predictions = (((data["action_type_Jump Shot"] + np.maximum( (((data["action_type_Jump Shot"] + ((((data["action_type_Jump Shot"] + ((data["action_type_Jump Shot"] + np.sin(np.tanh(data["opponent_OKC"])))/2.0)) + data["action_type_Tip Shot"]) + data["last_moments_1"])/2.0))/2.0)),  ((data["action_type_Layup Shot"] - data["shot_zone_area_Center(C)"]))))/2.0) +
                    (((data["shot_zone_area_Right Side Center(RC)"] * data["action_type_Driving Dunk Shot"]) + (data["distance_45"] - ((data["action_type_Slam Dunk Shot"] + ((((np.minimum( (((data["shot_zone_range_16-24 ft."] + data["action_type_Driving Dunk Shot"])/2.0)),  ((-(data["action_type_Hook Shot"])))) + data["action_type_Jump Bank Shot"])/2.0) + data["action_type_Driving Dunk Shot"])/2.0))/2.0)))/2.0) +
                    np.sin(np.maximum( (data["action_type_Reverse Layup Shot"]),  (((data["distance_1"] + ((data["action_type_Running Jump Shot"] * data["shot_zone_range_16-24 ft."]) + ((data["action_type_Reverse Layup Shot"] * (-(data["distance_7"]))) + ((data["season_2015-16"] > ((data["action_type_Reverse Layup Shot"] < data["season_2015-16"]).astype(float))).astype(float)))))/2.0)))) +
                    (-((((data["action_type_Alley Oop Dunk Shot"] * ((((data["action_type_Driving Finger Roll Shot"] + data["action_type_Pullup Jump shot"])/2.0) + (data["action_type_Reverse Dunk Shot"] + (data["action_type_Fadeaway Bank shot"] + data["action_type_Pullup Jump shot"])))/2.0)) <= np.minimum( (data["action_type_Running Jump Shot"]),  (np.sinh((data["away_False"] + data["opponent_NOH"]))))).astype(float)))) +
                    (((data["action_type_Driving Slam Dunk Shot"] * data["distance_23"]) + np.tanh(np.maximum( (data["distance_31"]),  (((data["distance_5"] + ((data["season_2014-15"] + np.maximum( (data["period_4"]),  (data["distance_6"])))/2.0))/2.0)))))/2.0) +
                    (data["action_type_Running Bank shot"] * np.maximum( (np.maximum( (data["distance_25"]),  (data["distance_21"]))),  (np.maximum( (np.maximum( (data["opponent_SAC"]),  (np.maximum( (np.maximum( (np.maximum( (data["action_type_Driving Finger Roll Layup Shot"]),  (data["action_type_Running Hook Shot"]))),  (data["season_2005-06"]))),  (np.maximum( (data["action_type_Turnaround Bank shot"]),  (data["distance_24"]))))))),  (data["action_type_Dunk"]))))) +
                    (((np.maximum( (data["action_type_Finger Roll Shot"]),  (data["distance_31"])) > (((data["distance_42"] * data["action_type_Layup"]) + (data["action_type_Driving Jump shot"] * data["opponent_BKN"]))/2.0)).astype(float)) + (data["season_2014-15"] * ((data["action_type_Fadeaway Bank shot"] + ((0.720430 == data["action_type_Finger Roll Shot"]).astype(float)))/2.0))) +
                    ((((data["action_type_Running Jump Shot"] >= np.maximum( ((5.428570 + np.sinh(data["shot_zone_basic_Mid-Range"]))),  (data["action_type_Running Jump Shot"]))).astype(float)) + (data["action_type_Dunk Shot"] * np.minimum( (np.sin((data["away_False"] + np.sinh((1.0/(1.0 + np.exp(- 1.197370))))))),  (data["last_moments_0"]))))/2.0) +
                    np.maximum( (data["distance_36"]),  (np.maximum( (np.maximum( (data["distance_39"]),  (np.maximum( (data["distance_41"]),  ((data["opponent_SAC"] * data["action_type_Alley Oop Layup shot"])))))),  (((data["distance_40"] > np.tanh((-((data["season_2011-12"] * ((data["action_type_Fadeaway Jump Shot"] >= data["action_type_Alley Oop Layup shot"]).astype(float))))))).astype(float)))))) +
                    ((data["action_type_Fadeaway Jump Shot"] * data["away_True"]) * (-((((data["action_type_Turnaround Jump Shot"] > data["distance_21"]).astype(float)) + np.minimum( (data["action_type_Floating Jump shot"]),  (data["action_type_Fadeaway Jump Shot"])))))))
                    
    return Outputs(predictions)


def GPIndividual3(data):
    predictions = ((((data["last_moments_0"] <= data["action_type_Layup Shot"]).astype(float)) + ((np.tanh(data["action_type_Jump Shot"]) + ((-(data["action_type_Slam Dunk Shot"])) + np.tanh((((data["action_type_Layup Shot"] + (data["action_type_Tip Shot"] + data["action_type_Jump Shot"]))/2.0) + data["action_type_Jump Shot"]))))/2.0)) +
                    ((np.tanh((((data["action_type_Turnaround Jump Shot"] >= np.sin(np.sin(data["action_type_Fadeaway Jump Shot"]))).astype(float)) + data["season_2015-16"])) + (data["action_type_Driving Dunk Shot"] * np.minimum( (data["shot_zone_range_16-24 ft."]),  ((data["action_type_Hook Shot"] * -2.0)))))/2.0) +
                    ((0.138462 * ((data["period_4"] + ((data["distance_45"] + data["shot_zone_area_Left Side(L)"]) + data["action_type_Turnaround Fadeaway shot"]))/2.0)) - np.sinh(((data["action_type_Driving Finger Roll Layup Shot"] > (data["distance_45"] - ((data["action_type_Alley Oop Dunk Shot"] + data["action_type_Driving Slam Dunk Shot"])/2.0))).astype(float)))) +
                    ((np.minimum( ((data["action_type_Dunk"] * data["shot_zone_basic_Left Corner 3"])),  (((data["action_type_Fadeaway Bank shot"] + np.maximum( (data["action_type_Reverse Dunk Shot"]),  ((data["shot_zone_range_16-24 ft."] + data["distance_1"])))) * data["shot_zone_basic_Left Corner 3"]))) + ((data["action_type_Reverse Layup Shot"] >= np.sin(data["action_type_Driving Finger Roll Shot"])).astype(float)))/2.0) +
                    ((((data["season_2014-15"] >= (data["action_type_Finger Roll Shot"] * data["action_type_Tip Shot"])).astype(float)) + (data["away_True"] * ((np.tanh(data["action_type_Tip Shot"]) <= (data["action_type_Driving Jump shot"] - ((data["action_type_Running Jump Shot"] <= (-(data["action_type_Pullup Jump shot"]))).astype(float)))).astype(float))))/2.0) +
                    ((((data["away_False"] * ((data["shot_zone_area_Back Court(BC)"] < data["action_type_Dunk Shot"]).astype(float))) + ((data["shot_zone_area_Back Court(BC)"] > (-(np.tanh(data["distance_31"])))).astype(float))) + ((data["action_type_Slam Dunk Shot"] > data["distance_41"]).astype(float))) - ((data["action_type_Driving Finger Roll Shot"] > data["action_type_Driving Finger Roll Layup Shot"]).astype(float))) +
                    ((-1.0 + np.maximum( (((data["shot_zone_area_Left Side Center(LC)"] > data["action_type_Jump Bank Shot"]).astype(float))),  (np.maximum( (data["distance_42"]),  (((data["action_type_Pullup Jump shot"] <= (data["shot_zone_basic_Mid-Range"] * ((data["action_type_Jump Bank Shot"] >= (data["action_type_Running Hook Shot"] * data["action_type_Pullup Jump shot"])).astype(float)))).astype(float)))))))/2.0) +
                    np.tanh((data["action_type_Running Bank shot"] * (data["distance_25"] + ((data["period_1"] - (data["action_type_Driving Jump shot"] - (data["shot_zone_area_Right Side Center(RC)"] - data["distance_7"]))) + (data["distance_21"] - np.maximum( (data["shot_zone_area_Right Side(R)"]),  (data["action_type_Driving Jump shot"]))))))) +
                    ((data["distance_24"] * ((data["action_type_Jump Bank Shot"] >= (data["season_1999-00"] * ((np.maximum( (data["action_type_Running Jump Shot"]),  (data["action_type_Driving Layup Shot"])) >= data["action_type_Driving Slam Dunk Shot"]).astype(float)))).astype(float))) + (data["action_type_Step Back Jump shot"] * ((data["minutes_remaining_7"] > (data["action_type_Jump Bank Shot"] * data["action_type_Slam Dunk Shot"])).astype(float)))) +
                    np.maximum( ((data["action_type_Alley Oop Layup shot"] * data["opponent_SAC"])),  (((data["distance_31"] > (((data["season_2002-03"] * np.minimum( (data["season_2010-11"]),  (data["distance_22"]))) + ((data["action_type_Layup Shot"] <= np.maximum( (data["opponent_CLE"]),  ((data["distance_36"] * data["distance_36"])))).astype(float)))/2.0)).astype(float)))))
                   
    return Outputs(predictions)


def GPIndividual4(data):
    predictions = (((data["action_type_Jump Shot"] + np.maximum( ((((-(data["shot_zone_basic_Restricted Area"])) + np.maximum( (((data["action_type_Jump Shot"] + data["action_type_Layup Shot"])/2.0)),  ((((data["action_type_Tip Shot"] + data["season_2015-16"]) + data["last_moments_1"])/2.0))))/2.0)),  ((data["action_type_Jump Shot"] + data["action_type_Layup Shot"]))))/2.0) +
                    (data["distance_45"] - ((((((data["action_type_Alley Oop Dunk Shot"] + np.maximum( (data["action_type_Jump Bank Shot"]),  (data["action_type_Running Jump Shot"])))/2.0) + data["action_type_Driving Dunk Shot"])/2.0) + ((data["action_type_Slam Dunk Shot"] + ((data["action_type_Driving Dunk Shot"] + data["action_type_Pullup Jump shot"])/2.0))/2.0))/2.0)) +
                    np.tanh((data["action_type_Running Bank shot"] * ((data["last_moments_0"] + (((data["shot_zone_range_16-24 ft."] + (data["action_type_Driving Slam Dunk Shot"] - data["distance_31"])) + data["shot_zone_area_Right Side Center(RC)"]) - ((data["action_type_Reverse Layup Shot"] > data["action_type_Reverse Layup Shot"]).astype(float)))) - data["action_type_Hook Shot"]))) +
                    np.sin(((data["away_True"] * ((((np.tanh(data["action_type_Reverse Dunk Shot"]) >= (data["action_type_Pullup Jump shot"] * ((data["action_type_Reverse Dunk Shot"] + data["action_type_Running Jump Shot"])/2.0))).astype(float)) >= (1.0/(1.0 + np.exp(- data["away_True"])))).astype(float))) - ((data["action_type_Fadeaway Bank shot"] > (-(data["action_type_Reverse Dunk Shot"]))).astype(float)))) +
                    ((np.tanh(np.maximum( (data["action_type_Reverse Layup Shot"]),  ((data["action_type_Finger Roll Shot"] + np.maximum( (data["opponent_OKC"]),  (((data["distance_40"] > (data["distance_28"] * data["season_2014-15"])).astype(float)))))))) + np.sin((-((data["distance_1"] * data["period_4"])))))/2.0) +
                    (-(((((data["action_type_Driving Finger Roll Layup Shot"] + ((data["action_type_Dunk"] > (data["action_type_Running Hook Shot"] * data["action_type_Driving Finger Roll Shot"])).astype(float)))/2.0) >= (data["action_type_Driving Slam Dunk Shot"] * ((((data["action_type_Floating Jump shot"] * data["away_False"]) * np.sinh(data["action_type_Dunk Shot"])) + data["action_type_Turnaround Bank shot"])/2.0))).astype(float)))) +
                    (0.094340 * (np.maximum( (data["distance_42"]),  ((((data["shot_zone_range_Less Than 8 ft."] + data["distance_31"]) + data["shot_zone_area_Left Side(L)"])/2.0))) - (data["action_type_Driving Jump shot"] * ((data["away_False"] + (-(data["distance_39"])))/2.0)))) +
                    (data["season_2013-14"] * np.maximum( (data["distance_23"]),  (np.maximum( (data["distance_21"]),  (np.maximum( (data["season_2005-06"]),  (np.maximum( (np.maximum( (data["distance_24"]),  (np.maximum( (data["distance_24"]),  (np.maximum( (data["distance_24"]),  (((data["opponent_TOR"] >= data["action_type_Fadeaway Bank shot"]).astype(float))))))))),  (data["action_type_Reverse Dunk Shot"]))))))))) +
                    (np.minimum( (data["opponent_NYK"]),  (((data["opponent_SAC"] >= data["period_2"]).astype(float)))) * np.minimum( (np.minimum( (data["opponent_SAC"]),  ((data["action_type_Hook Shot"] * (-(((data["shot_zone_area_Center(C)"] <= data["distance_7"]).astype(float)))))))),  (np.cos(data["minutes_remaining_0"])))) +
                    (((data["distance_35"] + data["distance_36"]) * (data["distance_25"] + (1.0/(1.0 + np.exp(- data["season_2006-07"]))))) + (data["away_False"] * np.sin(np.sinh((data["season_2007-08"] * np.sin(((data["distance_40"] <= data["action_type_Fadeaway Jump Shot"]).astype(float)))))))))
                    
    return Outputs(predictions)


def GPIndividual5(data):
    predictions = (((((data["action_type_Jump Shot"] + (-((data["action_type_Slam Dunk Shot"] - ((data["season_2015-16"] + (data["last_moments_1"] + np.maximum( (data["action_type_Tip Shot"]),  ((1.0/(1.0 + np.exp(- data["action_type_Jump Shot"])))))))/2.0)))))/2.0) + (data["action_type_Jump Shot"] * np.sin(np.cos(data["action_type_Layup Shot"]))))/2.0) +
                    (data["action_type_Driving Dunk Shot"] * (np.minimum( (data["action_type_Driving Layup Shot"]),  ((((data["shot_zone_area_Center(C)"] - (data["shot_zone_range_8-16 ft."] - data["action_type_Driving Dunk Shot"])) - data["action_type_Fadeaway Jump Shot"]) - (data["action_type_Hook Shot"] - (1.0/(1.0 + np.exp(- data["action_type_Driving Layup Shot"]))))))) + data["action_type_Alley Oop Dunk Shot"])) +
                    ((data["distance_45"] * ((data["action_type_Driving Slam Dunk Shot"] - data["period_4"]) + (np.maximum( (data["action_type_Jump Bank Shot"]),  (data["action_type_Driving Slam Dunk Shot"])) + data["action_type_Dunk"]))) - ((data["action_type_Reverse Dunk Shot"] > (data["away_True"] * ((data["action_type_Running Jump Shot"] > data["shot_zone_area_Back Court(BC)"]).astype(float)))).astype(float))) +
                    (((data["action_type_Driving Finger Roll Shot"] - np.maximum( (data["season_2014-15"]),  (data["distance_31"]))) * data["action_type_Driving Finger Roll Layup Shot"]) - ((((data["action_type_Dunk Shot"] < data["action_type_Driving Finger Roll Shot"]).astype(float)) <= (data["action_type_Pullup Jump shot"] * ((np.sin(data["away_False"]) > data["action_type_Driving Finger Roll Shot"]).astype(float)))).astype(float))) +
                    (np.tanh(data["distance_29"]) * (((data["distance_1"] - data["shot_zone_range_Less Than 8 ft."]) - np.maximum( (data["distance_42"]),  (data["distance_7"]))) - np.maximum( (data["action_type_Layup Shot"]),  (data["action_type_Turnaround Jump Shot"])))) +
                    np.sinh((data["distance_39"] - ((((((data["action_type_Driving Finger Roll Layup Shot"] > (data["action_type_Running Hook Shot"] * data["action_type_Slam Dunk Shot"])).astype(float)) > (data["action_type_Running Bank shot"] * ((data["action_type_Turnaround Bank shot"] + data["action_type_Fadeaway Bank shot"])/2.0))).astype(float)) > ((data["period_1"] >= data["action_type_Slam Dunk Shot"]).astype(float))).astype(float)))) +
                    np.sin(np.maximum( (data["distance_21"]),  (np.maximum( (((data["action_type_Finger Roll Shot"] > (data["action_type_Driving Jump shot"] * (data["away_True"] * data["distance_28"]))).astype(float))),  ((-((data["minutes_remaining_0"] * ((((data["action_type_Finger Roll Shot"] < data["shot_zone_area_Left Side(L)"]).astype(float)) + data["distance_28"])/2.0))))))))) +
                    ((data["distance_36"] + (((((data["season_1997-98"] > (((data["action_type_Fadeaway Jump Shot"] > data["distance_41"]).astype(float)) - data["distance_26"])).astype(float)) - ((data["opponent_NYK"] > (data["away_True"] * ((data["action_type_Fadeaway Jump Shot"] > data["action_type_Running Bank shot"]).astype(float)))).astype(float))) + data["distance_40"])/2.0))/2.0) +
                    ((data["distance_44"] + ((data["distance_41"] + data["distance_35"])/2.0)) * (data["action_type_Hook Shot"] - ((data["distance_5"] + (data["season_2012-13"] + data["season_2011-12"])) + (data["distance_6"] + data["opponent_OKC"])))) +
                    np.maximum( (((((data["season_2003-04"] > data["shot_zone_area_Left Side(L)"]).astype(float)) < np.minimum( (data["minutes_remaining_8"]),  (np.maximum( (data["season_2003-04"]),  (((data["distance_17"] >= np.sin(data["distance_13"])).astype(float))))))).astype(float))),  ((data["last_moments_1"] * ((data["opponent_NOH"] >= data["distance_37"]).astype(float))))))
                    
    return Outputs(predictions)


df = pd.read_csv("../input/data.csv")
df.drop(['game_event_id', 'game_id', 'lat', 'lon', 'team_id', 'team_name'],
        axis=1,
        inplace=True)
df.sort_values('game_date',  inplace=True)
mask = df['shot_made_flag'].isnull()

# Clean data
actiontypes = dict(df.action_type.value_counts())
df['type'] = \
    df.apply(lambda row: (row['action_type']
                          if actiontypes[row['action_type']] > 20
                          else row['combined_shot_type']), axis=1)
df.drop(['action_type', 'combined_shot_type'], axis=1, inplace=True)

df['away'] = df.matchup.str.contains('@')
df.drop('matchup', axis=1, inplace=True)

df['distance'] = df.apply(lambda row: row['shot_distance']
                          if row['shot_distance'] < 45 else 45, axis=1)

df['time_remaining'] = \
    df.apply(lambda x: x['minutes_remaining'] * 60 + x['seconds_remaining'],
             axis=1)
df['last_moments'] = \
    df.apply(lambda row: 1 if row['time_remaining'] < 3 else 0, axis=1)

data = pd.get_dummies(df['type'], prefix="action_type")

features = ["away", "period", "playoffs", "shot_type", "shot_zone_area",
            "shot_zone_basic", "season", "shot_zone_range", "opponent",
            "distance", "minutes_remaining", "last_moments"]
for f in features:
    data = pd.concat([data, pd.get_dummies(df[f], prefix=f), ], axis=1)
ss = StandardScaler()
train = data[~mask].copy()
features = train.columns
train[features] = np.round(ss.fit_transform(train[features]), 6)
train['shot_made_flag'] = df.shot_made_flag[~mask]
test = data[mask].copy()
test.insert(0, 'shot_id', df[mask].shot_id)
test[features] = np.round(ss.transform(test[features]), 6)
trainpredictions1 = GPIndividual1(train)
trainpredictions2 = GPIndividual2(train)
trainpredictions3 = GPIndividual3(train)
trainpredictions4 = GPIndividual4(train)
trainpredictions5 = GPIndividual5(train)
testpredictions1 = GPIndividual1(test)
testpredictions2 = GPIndividual2(test)
testpredictions3 = GPIndividual3(test)
testpredictions4 = GPIndividual4(test)
testpredictions5 = GPIndividual5(test)
predictions = (trainpredictions1 +
               trainpredictions2 +
               trainpredictions3 +
               trainpredictions4 +
               trainpredictions5)/5

print(log_loss(train.shot_made_flag.values, predictions.values))

predictions = (testpredictions1 +
               testpredictions2 +
               testpredictions3 +
               testpredictions4 +
               testpredictions5)/5

submission = pd.DataFrame({"shot_id": test.shot_id,
                           "shot_made_flag": predictions})
submission.sort_values('shot_id',  inplace=True)
submission.to_csv("arisubmission.csv", index=False)

predictions = np.power(trainpredictions1 *
                       trainpredictions2 *
                       trainpredictions3 *
                       trainpredictions4 *
                       trainpredictions5, 1./5)

print(log_loss(train.shot_made_flag.values, predictions.values))

predictions = np.power(testpredictions1 *
                       testpredictions2 *
                       testpredictions3 *
                       testpredictions4 *
                       testpredictions5, 1./5)

submission = pd.DataFrame({"shot_id": test.shot_id,
                           "shot_made_flag": predictions})
submission.sort_values('shot_id',  inplace=True)
submission.to_csv("geosubmission.csv", index=False)
