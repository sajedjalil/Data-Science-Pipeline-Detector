# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directoy

#clicks_test = pd.read_csv("../input/clicks_test.csv")
#mini_clicks_test = clicks_test.sample(10000


clicks_test = pd.read_csv("../input/clicks_test.csv")
mini_clicks_test = clicks_test.sample(10000, random_state = 0)
mini_clicks_test.to_csv("mini_clicks_test.csv")
'''
clicks_train = pd.read_csv("../input/clicks_train.csv")
mini_clicks_train = clicks_train.sample(40000, random_state = 0)
mini_clicks_train.to_csv("mini_clicks_train.csv")


promoted_content = pd.read_csv("../input/promoted_content.csv")
mini_promoted = promoted_content[promoted_content["ad_id"].isin(mini_clicks_train["ad_id"])]
mini_promoted.to_csv("mini_promoted.csv")


doc_cats = pd.read_csv("../input/documents_categories.csv")
mini_doc_cats = doc_cats[doc_cats["document_id"].isin(mini_promoted["document_id"])]
mini_doc_cats.to_csv("mini_doc_cats.csv")

doc_ents = pd.read_csv("../input/documents_entities.csv")
mini_doc_ents = doc_ents[doc_ents["document_id"].isin(mini_promoted["document_id"])]
mini_doc_ents.to_csv("mini_doc_ents.csv")

doc_meta = pd.read_csv("../input/documents_meta.csv")
mini_doc_meta = doc_meta[doc_meta["document_id"].isin(mini_promoted["document_id"])]
mini_doc_meta.to_csv("mini_doc_meta.csv")

doc_topics = pd.read_csv("../input/documents_topics.csv")
mini_doc_topics = doc_topics[doc_topics["document_id"].isin(mini_promoted["document_id"])]
mini_doc_topics.to_csv("mini_doc_topics.csv")

events = pd.read_csv("../input/events.csv")
mini_events = events[events["display_id"].isin(mini_clicks_train["display_id"])]
mini_events.to_csv("mini_events.csv")


#get an error on this, "../input/page_views.csv" does not exist idk
#page_views = pd.read_csv("../input/page_views.csv")
#mini_page_views = page_views[page_views["document_id"].isin(mini_promoted["document_id"])]
#mini_page_views.to_csv("mini_page_views.csv")

'''




















