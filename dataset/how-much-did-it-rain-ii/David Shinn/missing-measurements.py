import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('../input/train.csv')
df.info()

# Count how many measures are not null for each record
df['count_non_null_measures'] = df.loc[:, 'Ref':'Kdp_5x5_90th'].count(1)

# Plot histogram 
plt.figure()
df['count_non_null_measures'].hist(bins=20)
plt.title('Histogram of available non-null measures per record')
plt.xlabel('Count of Record Level Non-null Measures')
plt.show()
plt.savefig("1_Records_Available_Measures.png")

# Capture most populated records for each Id
grouped = df.groupby('Id')[['count_non_null_measures', 'Expected']].max()

# Plot histogram
plt.figure()
grouped['count_non_null_measures'].hist(bins=20)
plt.title('Histogram of most available non-null measures per Hour Id')
plt.xlabel('Count of Highest Record Level Non-null Measures on each Id')
plt.show()
plt.savefig("2_Ids_Available_Measures.png")

print('\nDistributions of Expected for Ids that have no measures vs. at least one')
print(grouped.groupby(grouped.count_non_null_measures > 0)['Expected']
        .describe(percentiles=[i/10. for i in range(11)]).unstack(0))

print('\nCounts of highest available measures across Ids')
print(grouped.count_non_null_measures.value_counts().sort_index())

df = pd.read_csv('../input/test.csv')
df.info()
df['count_non_null_measures'] = df.loc[:, 'Ref':'Kdp_5x5_90th'].count(1)
grouped = df.groupby('Id')[['count_non_null_measures']].max()
plt.figure()
grouped['count_non_null_measures'].hist(bins=20)
plt.title('Test Histogram of most available non-null measures per Hour Id')
plt.xlabel('Count of Highest Record Level Non-null Measures on each Id')
plt.show()
plt.savefig("3_Test_Ids_Available_Measures.png")