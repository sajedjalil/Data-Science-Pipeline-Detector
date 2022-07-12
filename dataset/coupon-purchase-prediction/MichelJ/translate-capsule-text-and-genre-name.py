__author__ = 'Toby Cheese'

"""
Reads Excel translation map for japanese strings and translates capsule text and genre name accordingly.
Please note that this still leaves area names untranslated, as we are not provided with a translation for those.
Needs python3, pandas and xlrd.
"""

import pandas as pd

# read the file and parse the first and only sheet (need python xlrd module)
f = pd.ExcelFile('../input/documentation/CAPSULE_TEXT_Translation.xlsx')
all = f.parse(parse_cols=[2,3,6,7], skiprows=4, header=1)

# data comes in two columns, produce a single lookup table from that
first_col = all[['CAPSULE_TEXT', 'English Translation']]
second_col = all[['CAPSULE_TEXT.1','English Translation.1']].dropna()
second_col.columns = ['CAPSULE_TEXT', 'English Translation']
all = first_col.append(second_col).drop_duplicates('CAPSULE_TEXT')
translation_map = {k:v for (k,v) in zip(all['CAPSULE_TEXT'], all['English Translation'])}

# write new files with substituted names
for f in ['../input/coupon_list_train.csv', '../input/coupon_list_test.csv']:
    infile = pd.read_csv(f)
    infile['CAPSULE_TEXT'] = infile['CAPSULE_TEXT'].map(translation_map)
    infile['GENRE_NAME'] = infile['GENRE_NAME'].map(translation_map)
    infile.to_csv(f.replace(".csv", "_translated.csv").replace("../input/", ""), index=False)