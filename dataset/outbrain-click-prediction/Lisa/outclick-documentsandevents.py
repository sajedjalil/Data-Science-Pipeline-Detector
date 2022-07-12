# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import copy
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

clicks_train = pd.read_csv("../input/clicks_train.csv")
promoted_content = pd.read_csv("../input/promoted_content.csv")
doc_cats = pd.read_csv("../input/documents_categories.csv")
doc_ents = pd.read_csv("../input/documents_entities.csv")
doc_meta = pd.read_csv("../input/documents_meta.csv")
doc_topics = pd.read_csv("../input/documents_topics.csv")
events = pd.read_csv("../input/events.csv")
page_views = pd.read_csv("../input/page_views_sample.csv")