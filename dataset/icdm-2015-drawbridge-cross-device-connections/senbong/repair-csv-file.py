# Note: Ben Hamner has provided the repaired csv with download in 
# https://www.kaggle.com/benhamner/icdm-2015-drawbridge-cross-device-connections/fixing-bad-csv-files-with-download
#
# Quotes are added to some items for the purporse of convert the string into list.
#
# Example:
#   id_all_ip_df.ix[:,2].apply(eval)

# import re
# from io import StringIO

# import pandas as pd

# # repair bad csv file & return StringIO
# def repair_bad_csv(filename, row_num=-1):
#     output = StringIO()
#     with open(filename) as fhandle:
#         for i,line in enumerate(fhandle):
#             # replace the regex pattern due to inconsistent values in
#             # id_all_property.csv where id=id_4346517
#             line = re.sub("((ip|category_|property_)[0-9]+|www\.www\.auctionzip\.ca)", "'\\1'", line)
#             output.write(line.replace("{", "\"[").replace("}", "]\""))
            
#             if i == (row_num - 1):
#                 break
#     output.seek(0)
#     return output

# # read first 10 lines
# id_all_ip_df = pd.read_csv(repair_bad_csv("../input/id_all_ip.csv", row_num=10))
# id_all_property_df = pd.read_csv(repair_bad_csv("../input/id_all_property.csv", row_num=10))
# property_category_df = pd.read_csv(repair_bad_csv("../input/property_category.csv", row_num=10))

# # display output
# print(id_all_ip_df.head())
# print(id_all_property_df.head())
# print(property_category_df.head())

import os
import zipfile

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

zipf = zipfile.ZipFile('all_data.zip', 'w', zipfile.ZIP_DEFLATED)
zipdir('../input/', zipf)
zipf.close()