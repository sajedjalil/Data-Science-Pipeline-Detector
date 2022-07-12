"""
Automatic features engineering usign featuretools: https://docs.featuretools.com/
To install: pip install featuretools
"""

import pandas as pd
import featuretools as ft

# Can't keep all of them since memory is restricted.
COLS_TO_TRANSFORM = ['var_0', 'var_1', 'var_2']
# Here, I limit the depth since memory is restricted.
MAX_DEPTH = 2
train_df = pd.read_csv('../input/train.csv')

es = ft.EntitySet()
es = es.entity_from_dataframe(entity_id="features", 
                              dataframe=train_df[COLS_TO_TRANSFORM],
                              index="index")
                              
                              
# To see the available transform primitives: ft.primitives.list_primitives()
# Trying some random ones here...

TRANSFORM_PRIMITIVES = ['add_numeric', 'multiply_numeric', 'subtract_numeric', 'divide_by_feature']

# TODO: Use Dask for || execution and speedup.
augmented_train_df, _ = ft.dfs(entityset=es, target_entity="features", 
                               trans_primitives=TRANSFORM_PRIMITIVES, max_depth=MAX_DEPTH)
                            
print(augmented_train_df)
print(augmented_train_df.sample(1).T)
