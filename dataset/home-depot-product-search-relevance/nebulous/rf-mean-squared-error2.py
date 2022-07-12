import pandas as pd

df_attr = pd.read_csv('../input/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
df_brand["product_uid"] = df_brand["product_uid"].astype(int)
df_brand.to_csv('brand_info.csv',index=False)
brand_counts = df_brand.groupby('brand').size()
brand_counts.to_csv('brand_counts.csv',index=False)