import pandas as pd
import numpy as np
types2 = {
        'itemID': np.dtype(int),
        'categoryID': np.dtype(int),
        'title': np.dtype(str),
        'description': np.dtype(str),
        'images_array': np.dtype(str),
        'attrsJSON': np.dtype(str),
        'price': np.dtype(float),
        'locationID': np.dtype(int),
        'metroID': np.dtype(float),
        'lat': np.dtype(float),
        'lon': np.dtype(float),
    }
items = pd.read_csv("../input/ItemInfo_train.csv", dtype=types2)
items[0:10000].shape
pd.DataFrame(items[0:10000]).to_csv('sample.csv')
