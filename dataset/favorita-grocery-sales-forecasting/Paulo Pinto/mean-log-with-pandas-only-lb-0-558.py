import pandas as pd

df_train = pd.read_csv('../input/train.csv', usecols=[1, 2, 3, 4], skiprows=range(1, 124035460),
    converters={'unit_sales': lambda u: pd.np.log1p(float(u)) if float(u) > 0 else 0}
)

# Fill gaps in dates
u_dates = df_train.date.unique()
u_stores = df_train.store_nbr.unique()
u_items = df_train.item_nbr.unique()
df_train.set_index(["date", "store_nbr", "item_nbr"], inplace=True)
df_train = df_train.reindex(
    pd.MultiIndex.from_product(
        (u_dates, u_stores, u_items),
        names=["date", "store_nbr", "item_nbr"]
    )
)

# Fill NAs
df_train.loc[:, "unit_sales"].fillna(0, inplace=True)

# Calculate means
df_train = df_train.groupby(
        ['item_nbr','store_nbr']
        )['unit_sales'].mean().to_frame('unit_sales').apply(pd.np.expm1)

# Create submission
pd.read_csv(
    "../input/test.csv", usecols=[0, 2, 3]
).set_index(
    ['item_nbr', 'store_nbr']
).join(
    df_train, how='left'
).fillna(0).to_csv(
    'meanlog.csv.gz', float_format='%.3f', index=None, compression="gzip"
)