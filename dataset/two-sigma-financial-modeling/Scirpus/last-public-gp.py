import numpy as np
import pandas as pd
import kagglegym


def GPTechnicalPrediction1(data, low_y_cut, high_y_cut):
    p = (1.000000*((((data["technical_13_Prev"] + (((data["technical_20_Prev"] - data["technical_20_Cur"]) + ((((((((((data["technical_25_Cur"] * ((data["technical_20_Cur"] + (data["technical_25_Cur"] * 2.0))/2.0)) + ((data["technical_20_Prev"] + data["technical_30_Cur"])/2.0))/2.0) + (data["technical_30_Cur"] - data["technical_20_Cur"]))/2.0) - data["technical_20_Cur"]) + data["technical_20_Prev"])/2.0) + ((((data["technical_30_Cur"] - data["technical_13_Prev"]) + data["technical_30_Cur"])/2.0) - data["technical_30_Prev"]))/2.0))/2.0))/2.0) / 2.0)) +
         0.999600*(((((-((((((-1.0 / 2.0) + (data["fundamental_0_Cur"] / 2.0))/2.0) / 2.0) / 2.0))) * ((((-1.0 / 2.0) / 2.0) / 2.0) / 2.0)) - data["technical_20_Prev"]) * (data["technical_20_Prev"] + (data["technical_20_Cur"] + ((data["technical_20_Cur"] + (-((((((-((data["fundamental_0_Cur"] - (-((-1.0 / 2.0)))))) / 2.0) / 2.0) / 2.0) / 2.0))))/2.0))))))
    return p.clip(low_y_cut, high_y_cut)


def GPTechnicalPrediction2(data, low_y_cut, high_y_cut):
    p = (1.000000*(((((((((data["fundamental_0_Cur"] * (0.19517545402050018)) / 2.0) - (((data["technical_30_Cur"] - (data["technical_20_Cur"] - data["technical_30_Cur"])) - data["technical_20_Prev"]) - data["technical_20_Cur"])) * ((data["technical_30_Cur"] - data["technical_20_Prev"]) - data["technical_20_Prev"])) + data["technical_13_Prev"])/2.0) + ((((data["technical_20_Prev"] - data["technical_20_Cur"]) + (((data["technical_20_Prev"] - data["technical_20_Cur"]) + (data["technical_30_Cur"] / 2.0))/2.0))/2.0) / 2.0))/2.0)) +
         0.909200*((data["technical_13_Cur"] * (((((data["technical_13_Cur"] * 2.0) + data["technical_13_Cur"]) - (14.54311275482177734)) * ((data["technical_13_Cur"] - data["technical_30_Prev"]) * 2.0)) - (data["technical_44_Prev"] + ((((((((((14.54310894012451172) + data["technical_13_Prev"])/2.0) * data["technical_11_Prev"]) + (14.54311275482177734))/2.0) * data["technical_13_Prev"]) * 2.0) * 2.0) * 2.0))))))
    return p.clip(low_y_cut, high_y_cut)


def GP(data, low_y_cut, high_y_cut):
    yhat = (GPTechnicalPrediction1(data, low_y_cut, high_y_cut) +
            GPTechnicalPrediction2(data, low_y_cut, high_y_cut))/2.
    return yhat


if __name__ == "__main__":
    print('Started')
    low_y_cut = -0.086093
    high_y_cut = 0.093497
    env = kagglegym.make()
    observation = env.reset()
    train = observation.train
    print(train.y.mean())
    y_is_above_cut = (train.y > high_y_cut)
    y_is_below_cut = (train.y < low_y_cut)
    y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
    median_values = train[y_is_within_cut].median(axis=0)
    defaulty = train[y_is_within_cut].y.mean()
    defaultsids = dict(train[y_is_within_cut].groupby(["id"])["y"].median())
    previousTechnical11 = {}
    previousTechnical13 = {}
    previousTechnical20 = {}
    previousTechnical25 = {}
    previousTechnical30 = {}
    previousTechnical44 = {}
    previousFundamental0 = {}
    while True:
        firstsids = []
        yarray = np.zeros(observation.target.y.shape[0])
        observation.features.fillna(median_values, inplace=True)
        timestamp = observation.features["timestamp"][0]
        allData = None
        for i in range(observation.target.y.shape[0]):
            sid = observation.features["id"].values[i]
            if(sid in previousTechnical11.keys()):
                data = np.zeros(shape=(1, 14))
                data[0, 0] = previousTechnical11[sid]
                data[0, 1] = observation.features["technical_11"][i]
                data[0, 2] = previousTechnical13[sid]
                data[0, 3] = observation.features["technical_13"][i]
                data[0, 4] = previousTechnical20[sid]
                data[0, 5] = observation.features["technical_20"][i]
                data[0, 6] = previousTechnical25[sid]
                data[0, 7] = observation.features["technical_25"][i]
                data[0, 8] = previousTechnical30[sid]
                data[0, 9] = observation.features["technical_30"][i]
                data[0, 10] = previousTechnical44[sid]
                data[0, 11] = observation.features["technical_44"][i]
                data[0, 12] = previousFundamental0[sid]
                data[0, 13] = observation.features["fundamental_0"][i]
                if(allData is None):
                    allData = data.copy()
                else:
                    allData = np.concatenate([allData, data])
            else:
                yarray[i] = -999999
                firstsids.append(sid)

            previousTechnical11[sid] = \
                observation.features["technical_11"][i]
            previousTechnical13[sid] = \
                observation.features["technical_13"][i]
            previousTechnical20[sid] = \
                observation.features["technical_20"][i]
            previousTechnical25[sid] = \
                observation.features["technical_25"][i]
            previousTechnical30[sid] = \
                observation.features["technical_30"][i]
            previousTechnical44[sid] = \
                observation.features["technical_44"][i]
            previousFundamental0[sid] = \
                observation.features["fundamental_0"][i]
        if(allData is not None):
            gpdata = pd.DataFrame({'technical_11_Prev': allData[:, 0],
                                   'technical_11_Cur': allData[:, 1],
                                   'technical_13_Prev': allData[:, 2],
                                   'technical_13_Cur': allData[:, 3],
                                   'technical_20_Prev': allData[:, 4],
                                   'technical_20_Cur': allData[:, 5],
                                   'technical_25_Prev': allData[:, 6],
                                   'technical_25_Cur': allData[:, 7],
                                   'technical_30_Prev': allData[:, 8],
                                   'technical_30_Cur': allData[:, 9],
                                   'technical_44_Prev': allData[:, 10],
                                   'technical_44_Cur': allData[:, 11],
                                   'fundamental_0_Prev': allData[:, 12],
                                   'fundamental_0_Cur': allData[:, 13]})

            yarray[yarray == 0] = GP(gpdata, low_y_cut, high_y_cut)
        # index = 0
        # for i in range(len(yarray)):
        #     if(yarray[i] == -999999):
        #         if(firstsids[index] in defaultsids.keys()):
        #             yarray[i] = defaultsids[firstsids[index]]
        #         else:
        #             yarray[i] = defaulty
        #         index += 1
        yarray[yarray == -999999] = defaulty
        observation.target.y = yarray
        target = observation.target
        observation, reward, done, info = env.step(target)
        if((timestamp % 100 == 0)):
            print(timestamp)

        if done:
            break

    print(info)
    print('Finished')