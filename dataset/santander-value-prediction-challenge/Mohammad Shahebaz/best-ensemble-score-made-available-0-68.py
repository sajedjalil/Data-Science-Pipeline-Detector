# ENSEMBLING THE LEAKED SUBMISSION

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# THE BEST_KERNEL submission rounded to greater decimals mimicing the target values in train. 
BEST_69 = pd.read_csv("../input/rounded/baseline_submission_with_leaks.csv") #WITH LAG 39
ROUNED_MIN2 = pd.read_csv("../input/rounded/baseline_submission_with_leaks_ROUNDED_MINUS2.csv")

CORR = pd.DataFrame()
CORR['BEST_69'] = BEST_69.target
CORR['ROUNED_MIN2'] = ROUNED_MIN2.target
print(CORR.corr())
#               BEST_69  ROUNED_MIN2
# BEST_69      1.000000     0.955497
# ROUNED_MIN2  0.955497     1.000000



ROUNED_MIN2.target = (7 * CORR['BEST_69'] + 3 * CORR['ROUNED_MIN2'] )/10
ROUNED_MIN2.to_csv("SHAZ13_ENS_LEAKS.csv", index=None)