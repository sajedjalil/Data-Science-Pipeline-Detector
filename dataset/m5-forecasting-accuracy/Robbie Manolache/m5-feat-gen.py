# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
# Feature Generating Script \--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\-
# -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-

import numpy as np
import pandas as pd
from m5_helper import load_data, create_id_col, add_grp_id, \
                      create_prc_grp_id, gen_price_features, \
                      gen_agg_prices, gen_agg_sales

# Load the data -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
train_df, cal_df, prc_df, sub_df = load_data()
prc_df = create_id_col(prc_df)

# Set meta data -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
value_vars = [c for c in train_df.columns if c.startswith('d_')]
id_vars = list(np.setdiff1d(train_df.columns.tolist(), value_vars))
grp_vars = [v for v in id_vars if v != 'id']
singles = [v[:-3] for v in grp_vars]
pairs = [['dept', 'state'], ['cat', 'state'], ['item', 'state'], 
         ['dept', 'store'], ['cat','store']]

# Create price groups -\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
prc_grps = prc_df.groupby('id')['sell_price'].mean().reset_index()
prc_grps = add_grp_id(prc_grps, ['dept'])
prc_grps = create_prc_grp_id(prc_grps, nmin=10, ncuts=2)

# Create features -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
prc_ft_df = gen_price_features(prc_df)
grp_list = [[s] for s in singles] + pairs + ["prc_grp"]
grp_prices = gen_agg_prices(grp_list, prc_ft_df, prc_grps)
grp_sales = gen_agg_sales(train_df, grp_vars, value_vars, pairs, prc_grps)

# Save output -/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/--/|.|\--\|.|/-
train_df[['id']+value_vars].to_csv("item_sales.csv", index=False)
grp_sales.to_csv("grp_sales.csv", index=False)
prc_ft_df.to_csv("item_prices.csv", index=False)
prc_grps.to_csv("price_grps.csv", index=False)
grp_prices.to_csv("grp_prices.csv", index=False)


