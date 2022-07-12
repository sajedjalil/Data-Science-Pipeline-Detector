__author__ = 'ntg'

"""
Translate all fields, using Google translate for not provided.
Extended from Toby Cheese's script.
Reads Excel translation map for japanese strings and translates capsule text and genre name accordingly.
This uses Google translate for area names
Needs python3, pandas, xlrd,  and goslate. (https://pypi.python.org/pypi/goslate)
need to uncomment lines 12,31,63,71
"""
#Need to uncomment line to make script work:
#import goslate
import numpy as np
import pandas as pd
pd.set_option('display.width', 1500)


# read the file and parse the first and only sheet (need python xlrd module)
f = pd.ExcelFile('../input/documentation/CAPSULE_TEXT_Translation.xlsx')
all = f.parse(parse_cols=[2,3,6,7], skiprows=4, header=1)

# data comes in two columns, produce a single lookup table from that
first_col = all[['CAPSULE_TEXT', 'English Translation']]
second_col = all[['CAPSULE_TEXT.1','English Translation.1']].dropna()
second_col.columns = ['CAPSULE_TEXT', 'English Translation']
all = first_col.append(second_col).drop_duplicates('CAPSULE_TEXT')
translation_map = {k:v for (k,v) in zip(all['CAPSULE_TEXT'], all['English Translation'])}


#Need to uncomment line to make script work:
#gs = goslate.Goslate()

# for k,v in translation_map.items():
#     gv =gs.translate(k,'en',source_language='ja')
#     if not (v == gv):
#         print ('Test translate: {} should be [{}] but auto-translated to [{}]'.format(k,v,gv))

allterms = np.array(list(translation_map.keys()))
# allterms = np.empty(0,dtype=np.object)

todo = {
     '../input/coupon_area_test.csv'         :['SMALL_AREA_NAME','PREF_NAME'],
     '../input/coupon_area_train.csv'        :['SMALL_AREA_NAME','PREF_NAME'],
     '../input/coupon_detail_train.csv'      :['SMALL_AREA_NAME'],
     '../input/coupon_list_test.csv'         :['CAPSULE_TEXT','GENRE_NAME','ken_name','large_area_name','small_area_name'],
     '../input/coupon_list_train.csv'        :['CAPSULE_TEXT','GENRE_NAME','ken_name','large_area_name','small_area_name'],
     # '../input/coupon_visit_train.csv'       :[],
     '../input/user_list.csv'                :['PREF_NAME'],
}

# find additional terms
for f,cols in todo.items():
    print ('Reading ', f)
    infile = pd.read_csv(f)
    print ('Enriching dictionary')
    for c in cols:
        toadd = infile[c].unique()
        if pd.isnull(toadd[0]) : toadd = toadd[1:]
        allterms = np.union1d(allterms, toadd)

auto_translation_map = { k:translation_map.get(k,"need to uncomment lines 12,31,63,71") for k in allterms}
#Need to uncomment line to make script work:
#auto_translation_map = { k:translation_map.get(k,gs.translate(k,'en','ja')) for k in allterms}

# write new files with substituted names
for f,cols in todo.items():
    print ('Mapping translation for: ', f)
    infile = pd.read_csv(f)
    for c in cols:
        infile[c] = infile[c].map(auto_translation_map)
    # infile.to_csv(f.replace(".csv", "_translated.csv"), index=False)

print ('Done.')


