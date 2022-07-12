# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


def anh(pred, child_pref, gift_pref):

  _, c = np.where(gift_pref == pred[:, None])
  anch = (((10 - c) * 2).sum() - (1000000 - len(c))) / (1000000 * 10 * 2)

  m = np.arange(1000)[:, None].repeat(1000, axis=1)
  _, c = np.where(pred[child_pref] == m)
  ansh = (((1000 - c) * 2).sum() - (1000000 - c.size)) / (1000 * 1000 * 1000 * 2)

  # print('ANCH: ', anch)
  # print('ANSH: ', ansh)

  return anch + ansh

gift_pref = pd.read_csv('../input/child_wishlist.csv',header=None).drop(0, 1).values
child_pref = pd.read_csv('../input/gift_goodkids.csv',header=None).drop(0, 1).values

random_sub = pd.read_csv('../input/sample_submission_random.csv').values[:, 1]
print(anh(random_sub, child_pref, gift_pref))