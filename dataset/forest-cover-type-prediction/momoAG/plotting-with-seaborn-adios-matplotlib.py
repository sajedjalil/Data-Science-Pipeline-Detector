import pandas as pd
import seaborn as sns
from pylab import savefig

train = pd.read_csv("../input/train.csv")
useful_feats_to_plot = ['Elevation', 'Horizontal_Distance_To_Hydrology',  'Horizontal_Distance_To_Roadways',
                           'Cover_Type']
points_to_show  = 1000 #Downsample for plotting purposes

df_to_plot = train.ix[:points_to_show, useful_feats_to_plot]
sns.pairplot(df_to_plot, hue="Cover_Type", diag_kind="kde")
savefig("my_plots.png")
