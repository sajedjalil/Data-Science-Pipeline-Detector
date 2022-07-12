# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
docs = ['clicks_test', 'clicks_train', 'documents_categories', 'documents_entities', 'documents_meta', 'documents_topics', 'events', 'page_views_sample', 'promoted_content', 'sample_submission']
for doc in docs:
    content = pd.read_csv('../input/' + doc + '.csv')
    print(doc)
    print(content.shape)
    print(content.columns.tolist())
    print(content.head())
    print("")