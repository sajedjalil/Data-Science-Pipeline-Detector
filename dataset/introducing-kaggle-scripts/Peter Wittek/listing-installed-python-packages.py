# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pip
for package in sorted(pip.get_installed_distributions(), key=lambda package: package.project_name):
    print("{} ({})".format(package.project_name, package.version))