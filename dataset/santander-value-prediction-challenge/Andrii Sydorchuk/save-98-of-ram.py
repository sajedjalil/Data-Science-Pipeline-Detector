import numpy as np
import pandas as pd

# This trick will enable those participants with less than 16GB of RAM to effectively
# iterate on Santander competition. The code below shows how to save around 2GB of RAM
# for each copy of test data in the ipython notebook.

test_df = pd.read_csv('../input/test.csv', index_col='ID')
test_size_mb = test_df.memory_usage().sum() / 1024 / 1024
print("Test memory size: %.2f MB" % test_size_mb)
# Test memory size: 1879.24 MB

test_df_sparse = test_df.replace(0, np.nan).to_sparse()
test_sparse_size_mb = test_df_sparse.memory_usage().sum() / 1024 / 1024
print("Test sparse memory size: %.2f MB" % test_sparse_size_mb)
# Test sparse memory size: 26.78 MB
