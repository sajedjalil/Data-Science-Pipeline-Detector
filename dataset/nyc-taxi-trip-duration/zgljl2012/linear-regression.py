# coding=UTF-8

import pandas as pd
from sklearn.linear_model import LinearRegression
from functools import wraps
from datetime import datetime
import time

def timing(func):
    """
    计时装饰器
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        """
        装饰函数
        """
        start = time.clock()
        r = func(*args, **kwargs)
        end =time.clock()
        print('[' + func.__name__ + ']used:' + str(end - start))
        return r
    return wrapper

@timing
def dataset():
    """
    读取数据
    """
    train = pd.read_csv("../input/train.csv", index_col="id")
    test = pd.read_csv("../input/test.csv", index_col="id")
    return train, test

@timing
def build_model(data, labels):
    """
    训练模型
    """
    reg = LinearRegression()
    reg.fit(data, labels)
    return reg

@timing
def predict(model, id, data):
    """
    生成结果
    """
    res = model.predict(data)
    res = [int(i) if i>=0 else 0 for i in res]
    result = pd.DataFrame({'id': id, "trip_duration": res})
    result.index = result["id"]
    result = result.drop("id", axis=1)
    result.to_csv("result"+".csv")
    return result


if __name__ == "__main__":
    train, test = dataset()
    target = ["vendor_id", "passenger_count", "pickup_longitude", 
        "pickup_latitude", "dropoff_longitude", "dropoff_latitude"]
    data = train.loc[:, target]
    labels = train.iloc[:, -1]
    model = build_model(data, labels)
    data_test = test.loc[:, target]
    res = predict(model, test.index, data_test)
    print(res.head(5))
