# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

import os
from bs4 import BeautifulSoup
import pandas as pd
import requests

# Wikipedia
URL = 'https://en.wikipedia.org/wiki/List_of_cities_and_towns_in_Russia_by_population'

def scrape_wiki():
    wiki_dict = []
    
    r = requests.get(URL)

    # Soup
    bso = BeautifulSoup(r.text, 'lxml')
    
    # Table object
    tab = bso.body.find('table')
    
    # Rows
    rows = tab.find_all('tr')
    for row in rows[1:]:
        # Population
        city_name = row.find_all('td')[1].find_all('span')[0].text
    
        # Name (Russian)
        pop = int(row.find_all('td')[4].text.replace(',', ''))
        wiki_dict.append({'city': city_name, 'population': pop})
        print(city_name, pop)
    
    df = pd.DataFrame(data=wiki_dict)
    df = df.drop_duplicates('city')
    df.to_csv('city_population.csv')


if __name__ == '__main__':
    print('Uncomment the main function and run locally (no internet access within Kaggle)')
    # scrape_wiki()