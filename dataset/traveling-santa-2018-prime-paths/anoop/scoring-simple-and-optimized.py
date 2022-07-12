import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy.ntheory.primetest import isprime
# Input data files are available in the "../input/" directory.

cities = pd.read_csv('../input/cities.csv', index_col='CityId')

def getScore(submission_file_path):
    path = pd.read_csv(submission_file_path).values.reshape(-1)
    from_city = cities.iloc[path[:-1]].values
    to_city = cities.iloc[path[1:]].values
    distance = np.linalg.norm(from_city - to_city, axis=1)

    non_prime_factor = [1.1 if not isprime(city_id) else 1 for city_id in path[:-1][9::10] ]
    distance[9::10] *= non_prime_factor
    return distance.sum()

print(getScore(submission_file_path='../input/sample_submission.csv'))