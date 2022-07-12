import numpy as np
import pandas as pd
import kagglegym


def GPTechnical20Prediction1(data):
    p = ((((data["technical_20_Row_Offset_1"] + ((((((data[
            "technical_20_Row_Offset_1"] +
        ((((((((((-(data[
                                "technical_30_Row_Offset_0"
                            ])) *
                            data[
                                "technical_30_Row_Offset_0"
                            ]
                        ) +
                        data[
                            "technical_30_Row_Offset_0"
                        ]) /
                    2.0) +
                data[
                    "technical_20_Row_Offset_1"
                ]) / 2.0) +
            data[
                "technical_20_Row_Offset_0"
            ]) / 2.0) + data[
            "technical_20_Row_Offset_1"
        ]) / 2.0)) / 2.0) + data[
        "technical_20_Row_Offset_1"]) / 2.0) + ((0.0 +
        data["technical_20_Row_Offset_0"]) / 2.0)) / 2.0)) / 2.0)))
    return p.values.clip(0.0, 0.0678482)


def GPTechnical20Prediction2(data):
    p = (((data["technical_20_Row_Offset_1"] + ((((((-(data[
        "technical_20_Row_Offset_0"])) * data[
        "technical_20_Row_Offset_1"]) + data[
        "technical_20_Row_Offset_0"]) / 2.0) + ((data[
        "technical_20_Row_Offset_1"] + ((((data[
        "technical_20_Row_Offset_1"
    ] + ((data[
        "technical_20_Row_Offset_0"
    ] + ((((data[
                "technical_20_Row_Offset_0"
            ] +
            data[
                "technical_30_Row_Offset_0"
            ]) /
        2.0) + data[
        "technical_20_Row_Offset_1"
    ]) / 2.0)) / 2.0)) / 2.0) + data[
        "technical_20_Row_Offset_1"]) / 2.0)) / 2.0)) / 2.0)) / 2.0))
    return p.values.clip(0.0, 0.0678482)


def GPTechnical30Prediction1(data):
    p = (((data["technical_30_Row_Offset_0"] + ((((((2.0) * data[
        "technical_30_Row_Offset_0"]) + data[
        "technical_30_Row_Offset_0"]) / 2.0) + (((
        4.36477041244506836) * (((
        5.41192293167114258) + (
        5.41191959381103516)) * (data[
        "technical_30_Row_Offset_0"] + data[
        "technical_30_Row_Offset_0"]))) * (-(data[
        "technical_30_Row_Offset_0"])))) / 2.0)) / 2.0))
    return p.values.clip(0.0, 0.0719502)


def GPTechnical30Prediction2(data):
    p = (((((data["technical_20_Row_Offset_1"] + ((data[
        "technical_30_Row_Offset_0"] - data[
        "technical_20_Row_Offset_1"]) + (data[
        "technical_20_Row_Offset_0"] - (-(data[
        "technical_20_Row_Offset_1"]))))) * (-((((
            13.70487880706787109) + (0.0 + (5.0))) +
        data["technical_30_Row_Offset_0"])))) + (
        0.79312223196029663)) * data["technical_30_Row_Offset_0"]))
    return p.values.clip(0.0, 0.0719502)


def GPTechnicalPrediction1(data, low_y_cut, high_y_cut):
    p = ((data["technical_20_Row_Offset_2"] + (((data[
        "technical_20_Row_Offset_2"] + (((-(data[
        "technical_30_Row_Offset_0"])) - (
        0.90638661384582520)) * data[
        "technical_30_Row_Offset_1"])) + data[
        "technical_30_Row_Offset_1"]) + (data[
        "technical_20_Row_Offset_1"] * (((data[
        "technical_30_Row_Offset_1"] - data[
        "technical_20_Row_Offset_1"]) - (
        0.90638661384582520)) - (0.90638661384582520))))))
    return p.values.clip(low_y_cut, high_y_cut)


def GPTechnicalPrediction2(data, low_y_cut, high_y_cut):
    p = ((((data["technical_20_Row_Offset_0"] - data[
        "technical_20_Row_Offset_2"]) + ((((data[
        "technical_30_Row_Offset_1"] + (-(((((data[
        "technical_30_Row_Offset_2"
    ] + data[
        "technical_20_Row_Offset_0"
    ]) - data[
        "technical_20_Row_Offset_1"
    ]) + data[
        "technical_20_Row_Offset_0"
    ]) / 2.0)))) / 2.0) + ((data[
        "technical_20_Row_Offset_0"] + data[
        "technical_20_Row_Offset_0"]) * ((
        14.14197921752929688) * (-(data[
        "technical_20_Row_Offset_1"]))))) / 2.0)) / 2.0))
    return p.values.clip(low_y_cut, high_y_cut)


def GP(data, low_y_cut, high_y_cut):
    data["technical_20_Row_Offset_2"] = \
        (.5*GPTechnical20Prediction1(data) +
         .5*GPTechnical20Prediction2(data))
    data["technical_30_Row_Offset_2"] = \
        (.5*GPTechnical30Prediction1(data) +
         .5*GPTechnical30Prediction2(data))
    yhat = (.5*GPTechnicalPrediction1(data, low_y_cut, high_y_cut) +
            .5*GPTechnicalPrediction2(data, low_y_cut, high_y_cut))
    return yhat


if __name__ == "__main__":
    print('Started')
    env = kagglegym.make()
    observation = env.reset()
    train = observation.train
    median_values = train.median(axis=0)
    low_y_cut = -0.086093
    high_y_cut = 0.093497
    y_is_above_cut = (train.y > high_y_cut)
    y_is_below_cut = (train.y < low_y_cut)
    y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
    defaulty = train.y[y_is_within_cut].median()
    defaultsids = dict(train.groupby(["id"])["y"].median())
    previousTechnical20 = {}
    previousTechnical30 = {}
    firstsids = []
    while True:
        yarray = np.zeros(observation.target.y.shape[0])
        observation.features.fillna(median_values, inplace=True)
        timestamp = observation.features["timestamp"][0]
        if timestamp % 100 == 0:
            print("Timestamp #{}".format(timestamp))
        allData = None
        for i in range(observation.target.y.shape[0]):
            sid = observation.features["id"].values[i]
            if(sid in previousTechnical20.keys()):
                data = np.zeros(shape=(1, 4))
                data[0, 0] = previousTechnical20[sid]
                data[0, 1] = observation.features["technical_20"][i]
                data[0, 2] = previousTechnical30[sid]
                data[0, 3] = observation.features["technical_30"][i]
                if(allData is None):
                    allData = data.copy()
                else:
                    allData = np.concatenate([allData, data])
            else:
                yarray[i] = -999999
                firstsids.append(sid)
            previousTechnical20[sid] = \
                observation.features["technical_20"][i]
            previousTechnical30[sid] = \
                observation.features["technical_30"][i]

        if(allData is not None):
            gpdata = pd.DataFrame({'technical_20_Row_Offset_0': allData[:, 0],
                                   'technical_20_Row_Offset_1': allData[:, 1],
                                   'technical_30_Row_Offset_0': allData[:, 2],
                                   'technical_30_Row_Offset_1': allData[:, 3]})

            yarray[yarray == 0] = GP(gpdata, low_y_cut, high_y_cut)
        index = 0
        for i in range(len(yarray)):
            if(yarray[i] == -999999):
                if(firstsids[index] in defaultsids.keys()):
                    yarray[i] = defaultsids[firstsids[index]]
                else:
                    yarray[i] = defaulty
                index += 1
        observation.target.y = yarray
        target = observation.target
        observation, reward, done, info = env.step(target)
        if done:
            break

    print(info)
    print('Finished')
