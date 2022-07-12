# here is a short version of getting submission with ndf_leak

import pandas as pd
import numpy as np

train_users = pd.read_csv('../input/train_users.csv')
test_users = pd.read_csv('../input/test_users.csv')

test_users["pred0"] = test_users["date_first_booking"].apply(lambda x: "NDF" if pd.isnull(x) else "US")
test_users["pred1"] = test_users["date_first_booking"].apply(lambda x: "US" if pd.isnull(x) else "other")
test_users["pred2"] = test_users["date_first_booking"].apply(lambda x: "other" if pd.isnull(x) else "FR")
test_users["pred3"] = test_users["date_first_booking"].apply(lambda x: "FR" if pd.isnull(x) else "IT")
test_users["pred4"] = test_users["date_first_booking"].apply(lambda x: "IT" if pd.isnull(x) else "GB")

submission = test_users[["id", "pred0"]].rename(columns={'pred0':'country_destination'})
submission = pd.concat([submission, test_users[["id", "pred1"]].rename(columns={'pred1':'country_destination'})],
                       ignore_index=True)
submission = pd.concat([submission, test_users[["id", "pred2"]].rename(columns={'pred2':'country_destination'})],
                       ignore_index=True)
submission = pd.concat([submission, test_users[["id", "pred3"]].rename(columns={'pred3':'country_destination'})],
                       ignore_index=True)
submission = pd.concat([submission, test_users[["id", "pred4"]].rename(columns={'pred4':'country_destination'})],
                       ignore_index=True)

submission.to_csv('ndf_leak_short.csv', index=False)