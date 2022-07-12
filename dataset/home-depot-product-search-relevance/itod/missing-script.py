import numpy as np
import pandas as pd
import csv

df_attr = pd.read_csv('../input/attributes.csv')

# delete line bellow
df_attr = df_attr.iloc[:1000]

def extract_unique_words(group):
	group.fillna("", inplace=True)
	return {"col_helper": " ".join(list(set(" ".join(group).split(" "))))}

df_brand =  df_attr[df_attr["name"].str.contains("MFG Brand Name", na=False)]\
	[["product_uid", "value"]].rename(columns={"value": "brand"})
df_helper =  df_attr[df_attr["name"].str.contains("Type|type|Application", na=False)]\
	[["product_uid", "value"]]
df_type = df_helper["value"].groupby(df_helper["product_uid"])\
	.apply(extract_unique_words).unstack().rename(columns={"col_helper": "type"})
df_type['product_uid'] = df_type.index
df_helper =  df_attr[df_attr["name"].str.contains("Color|color", na=False)]\
	[["product_uid", "value"]]
df_color = df_helper["value"].groupby(df_helper["product_uid"])\
	.apply(extract_unique_words).unstack().rename(columns={"col_helper": "color"})
df_color['product_uid'] = df_color.index
df_helper =  df_attr[df_attr["name"].str.contains("Material|material", na=False)]\
	[["product_uid", "value"]]
df_material = df_helper["value"].groupby(df_helper["product_uid"])\
	.apply(extract_unique_words).unstack().rename(columns={"col_helper": "material"})
df_material['product_uid'] = df_material.index
df_helper =  df_attr[df_attr["name"].str.contains("Style|style|Shape", na=False)]\
	[["product_uid", "value"]]
df_style = df_helper["value"].groupby(df_helper["product_uid"])\
	.apply(extract_unique_words).unstack().rename(columns={"col_helper": "style"})
df_style['product_uid'] = df_style.index
df_helper =  df_attr[df_attr["name"].str.contains("Bullet", na=False)]\
	[["product_uid", "value"]]
df_bullets = df_helper["value"].groupby(df_helper["product_uid"])\
	.apply(extract_unique_words).unstack().rename(columns={"col_helper": "bullets"})
df_bullets['product_uid'] = df_bullets.index

df_all = pd.merge(df_brand, df_type, how="left", on="product_uid", sort=False)
df_all = pd.merge(df_all, df_color, how="left", on="product_uid", sort=False)
df_all = pd.merge(df_all, df_material, how="left", on="product_uid", sort=False)
df_all = pd.merge(df_all, df_style, how="left", on="product_uid", sort=False)
df_all = pd.merge(df_all, df_bullets, how="left", on="product_uid", sort=False)
df_all = df_all.fillna("")

df_all['product_uid'] = df_all['product_uid'].map(lambda x:x.astype(int))
pd.DataFrame(df_all).to_csv('attributes_in_useful_form.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)
