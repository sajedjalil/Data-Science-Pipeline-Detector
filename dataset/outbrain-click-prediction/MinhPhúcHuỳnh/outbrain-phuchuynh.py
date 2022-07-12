# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Load data
clicks_train = pd.read_csv("../input/clicks_train.csv")
clicks_test = pd.read_csv("../input/clicks_test.csv")
doc_cat = pd.read_csv("../input/documents_categories.csv")
doc_ent = pd.read_csv("../input/documents_entities.csv")
meta = pd.read_csv("../input/documents_meta.csv")
topcis = pd.read_csv("../input/documents_topics.csv")
events = pd.read_csv("../input/events.csv")
pageview_sample = pd.read_csv("../input/page_views_sample.csv")
promo = pd.read_csv("../input/promoted_content.csv") 
sample_submiss = pd.read_csv("../input/sample_submission.csv") 