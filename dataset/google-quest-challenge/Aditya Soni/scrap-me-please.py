# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import requests, re
from bs4 import BeautifulSoup as bs
from tqdm import tqdm

data_dir = '../input/google-quest-challenge/'
train = pd.read_csv(data_dir+"train.csv")
test = pd.read_csv(data_dir+"test.csv")

temp_train_stats = []
temp_test_stats = []

for url in tqdm(train["url"].values):
    
    response = requests.get(url)
    soup = bs(response.content, 'html.parser')
    body = soup.find('body')
    try:
        temp_train_stats.append([url, soup.find('div', { "class" : re.compile("js-voting-container")}).find_all("div")[0]["data-value"],\
              soup.find('div', { "class" : re.compile("js-favorite-count mt4")})["data-value"],\
              soup.find('div', {"class" : "grid--cell ws-nowrap mb8"})["title"],\
              soup.find("div", {"class": re.compile(r"subheader answers-subheader")}).h2["data-answercount"],\
              [i["href"].split("/")[-1] for i in soup.find("div", {"class" : "post-taglist grid gs4 gsy fd-column"}).find_all("a")],\
              len(soup.find('div', { "class" : re.compile("comments js-comments-container bt bc-black-2 mt12*")}).find_all("li"))
        ])
    except Exception as e:
        print(e, url)

np.array(temp_train_stats).dump(open('temp_train_stats.npy', 'wb'))

for url in tqdm(test["url"].values):
    
    response = requests.get(url)
    soup = bs(response.content, 'html.parser')
    body = soup.find('body')
    try:
        temp_test_stats.append([url, soup.find('div', { "class" : re.compile("js-voting-container")}).find_all("div")[0]["data-value"],\
              soup.find('div', { "class" : re.compile("js-favorite-count mt4")})["data-value"],\
              soup.find('div', {"class" : "grid--cell ws-nowrap mb8"})["title"],\
              soup.find("div", {"class": re.compile(r"subheader answers-subheader")}).h2["data-answercount"],\
              [i["href"].split("/")[-1] for i in soup.find("div", {"class" : "post-taglist grid gs4 gsy fd-column"}).find_all("a")],\
              len(soup.find('div', { "class" : re.compile("comments js-comments-container bt bc-black-2 mt12*")}).find_all("li"))
        ])
    except Exception as e:
        print(e, url)

np.array(temp_test_stats).dump(open('temp_test_stats.npy', 'wb'))