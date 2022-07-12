import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks 
import peakutils
from pandas.plotting import scatter_matrix
from itertools import compress
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

print(os.listdir("../input"))

# Loading data

X_train = pd.read_csv("../input/X_train.csv")
X_test = pd.read_csv("../input/X_test.csv")
y_train = pd.read_csv("../input/y_train.csv")
sample_submission = pd.read_csv("../input/sample_submission.csv")

# Proportion of each surface in training data
sns.countplot(y= "surface",data=y_train,order=y_train['surface'].value_counts().index)
plt.show()

# Scatter matrix
sns.set(rc={"figure.facecolor":"gray","axes.facecolor":"gray",
    "xtick.color":"black","ytick.color":"black","axes.grid":False,"axes.labelcolor":"black"})

scatter_matrix(X_train[X_train["series_id"]==2][X_train.columns[3:]],figsize=(26,16),color="red")
plt.show()

# Train and test data numerical column histogram
plt.style.use('default')
plt.figure(figsize=(15,10))
for i,col in enumerate(X_train.columns[3:]):
    plt.subplot(3,4,i+1)
    plt.title(col,fontsize=5)
    plt.hist(X_train[col],color="blue",label="train",bins=100)
    plt.hist(X_test[col],color="green",label="test",bins=100)
    plt.tick_params(labelsize=5)
plt.savefig("histogram.png")
plt.show()

# Some variables are approximately normal?
plt.figure(figsize=(15,10))
for i,col in enumerate(X_train.columns[7:]):
    mean = round((X_train[col]).mean(),2)
    stdv = round((X_train[col]).std(),2)
    plt.subplot(2,3,i+1)
    sns.distplot(X_train[col], hist = True, kde = True,label=col,bins=100,kde_kws = {'linewidth': 1})
    sns.distplot(np.random.normal(mean,stdv,len(X_train)), 
        hist = False, kde = True,label="N({},{})".format(mean,stdv),kde_kws = {'linewidth': 1})
    plt.xlabel("")
    plt.legend(fontsize=8)
plt.savefig("Normal_approximation.png",dpi=96)
plt.show()


# Any outliers in train data? -- Boxplot
plt.figure(figsize=(15,10))
for i,col in enumerate(X_train.columns[7:]):
    plt.subplot(2,3,i+1)
    sns.boxplot(x=X_train[col]) #,label = col
    plt.xlabel(col)
plt.savefig("X_train_box.png",dpi=96)
plt.show()
# Any outliers in test data?  -- Boxplot
plt.figure(figsize=(15,10))
for i,col in enumerate(X_test.columns[7:]):
    plt.subplot(2,3,i+1)
    sns.boxplot(x=X_test[col]) #,label = col
    plt.xlabel(col)
plt.savefig("X_test_box.png",dpi=96)
plt.show()

# Densityplot 
sns.set(rc={"figure.facecolor":"white","axes.facecolor":"white",
    "xtick.color":"black","ytick.color":"black","axes.grid":False,"axes.labelcolor":"black"})
def density_plot_surfacewise(df,col):
    for label in y_train["surface"].unique():
        sns.kdeplot(df[df.surface == label][col],label=label,bw=1.0) #,bins=100
    plt.xlabel(col)
    plt.legend(fontsize=7)

plt.figure(figsize = (26,12))
# sns.set_style('darkgrid')
for i,col in enumerate(X_train.columns[3:]):
    plt.subplot(2,5,i+1)
    density_plot_surfacewise(X_train.merge(y_train,on="series_id",how="inner"),col=col)
plt.savefig("histogram.png")
plt.show()

#-------------------------------------------------------------------------------------

np.random.seed(100)
k = [0 for x in range(3)]
# k=[12,13,15]
for i in range(3):
    k[i] = np.random.randint(0,len(X_train['series_id'].unique()))

sns.set(rc={"figure.facecolor":"white","axes.facecolor":"white",
    "xtick.color":"black","ytick.color":"black","axes.grid":False,"axes.labelcolor":"black"})

fig = plt.figure(figsize=(26, 16))
plt.title("Three random series",color="purple",y=1,fontsize=20)
plt.xticks(None);plt.yticks(None)
outer = gridspec.GridSpec(3, 4, wspace=0.2, hspace=0.2)

for i,col in enumerate(X_train.columns[3:]):
    inner = gridspec.GridSpecFromSubplotSpec(1, 1,
                    subplot_spec=outer[i], wspace=0.1, hspace=0.1)

    ax1 = plt.Subplot(fig,inner[0])
    t1 = ax1.plot(X_train[X_train["series_id"]== k[0]][col].reset_index(drop=True),"c--",alpha = 0.9,
        label = y_train[y_train['series_id'] == k[0]]["surface"].values[0])
    ax1.hlines(y=np.mean(X_train[X_train["series_id"]== k[0]][col]),xmin = 0, xmax = 127,colors="c")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title(col,color="black")
    leg = ax1.legend(loc="upper center",facecolor=None,fontsize=9,frameon=False)
    for text in leg.get_texts():
        plt.setp(text, color = 'black')
    fig.add_subplot(ax1)

    ax2 = ax1.twinx()
    t2 = ax2.plot(X_train[X_train["series_id"]== k[1]][col].reset_index(drop=True),"r-.",alpha = 0.8,
        label = y_train[y_train['series_id'] == k[1]]["surface"].values[0])
    ax2.hlines(y=np.mean(X_train[X_train["series_id"]== k[1]][col]),xmin = 0, xmax = 127,colors="r")
    ax2.set_xticks([])
    ax2.set_yticks([])
    leg = ax2.legend(loc="lower right",facecolor=None,fontsize=9,frameon=False)
    for text in leg.get_texts():
        plt.setp(text, color = 'black')
    
    ax3 = ax2.twinx()
    t3 = ax3.plot(X_train[X_train["series_id"]== k[2]][col].reset_index(drop=True),"y-",alpha = 0.99,
        label = y_train[y_train['series_id'] == k[2]]["surface"].values[0])
    ax3.hlines(y=np.mean(X_train[X_train["series_id"]== k[2]][col]),xmin = 0, xmax = 127,colors="y")
    ax3.set_xticks([])
    ax3.set_yticks([])
    leg = ax3.legend(loc="lower left",facecolor=None,fontsize=9,frameon=False)
    for text in leg.get_texts():
        plt.setp(text, color = 'black')
plt.savefig("plot1.png")
fig.show()


#------------------------------------------------------------------------------------------
# Correlation in data columns
corr_train = round(X_train.iloc[:,3:].corr(),2)
corr_test = round(X_test.iloc[:,3:].corr(),2)

mask_train = np.zeros_like(corr_train, dtype=np.bool)
mask_train[np.triu_indices_from(mask_train)] = True
mask_test = np.zeros_like(corr_test, dtype=np.bool)
mask_test[np.triu_indices_from(mask_test)] = True
cmap = sns.diverging_palette(10,220, as_cmap=True)

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.title("Training data correlation status",color="purple")
sns.heatmap(corr_train, mask=mask_train, cmap=cmap, vmax=1, center=0,vmin=-1,annot=True,
            square=True, linewidths=.2, cbar_kws={"shrink": .3})

plt.subplot(1,2,2)
plt.title("Test data correlation status",color="purple")
sns.heatmap(corr_test, mask=mask_test, cmap=cmap, vmax=1, center=0,vmin=-1,annot=True,
            square=True, linewidths=.2, cbar=False)
plt.savefig("corrplot1.png")
plt.show()

# correlation between pairs of columns for particular series
k = np.random.randint(0,X_train["series_id"].unique().max(),1)[0]

corr_train = round(X_train[X_train["series_id"]==k][X_train.columns[3:]].corr(),2)
corr_test = round(X_test[X_test["series_id"]==k][X_test.columns[3:]].corr(),2)

mask_train = np.zeros_like(corr_train, dtype=np.bool)
mask_train[np.triu_indices_from(mask_train)] = True
mask_test = np.zeros_like(corr_test, dtype=np.bool)
mask_test[np.triu_indices_from(mask_test)] = True
cmap = sns.diverging_palette(10,220, as_cmap=True)

plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.title("Training data correlation status \n series : " + str(k) ,color="purple")
sns.heatmap(corr_train, mask=mask_train, cmap=cmap, vmax=1, center=0,vmin=-1,annot=True,
            square=True, linewidths=.2, cbar_kws={"shrink": .3})

plt.subplot(1,2,2)
plt.title("Test data correlation status \n series : " + str(k),color="purple")
sns.heatmap(corr_test, mask=mask_test, cmap=cmap, vmax=1, center=0,vmin=-1,annot=True,
            square=True, linewidths=.2, cbar=False)
plt.savefig("corrplot2.png")
plt.show()

#------------------------------------------------------------------------------------------

## [Normalization](https://stackoverflow.com/questions/11667783/quaternion-and-normalization)
def normalise(data):
    data['mod_quat'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2 + data['orientation_W']**2)**0.5
    data['norm_X'] = data['orientation_X'] / data['mod_quat']
    data['norm_Y'] = data['orientation_Y'] / data['mod_quat']
    data['norm_Z'] = data['orientation_Z'] / data['mod_quat']
    data['norm_W'] = data['orientation_W'] / data['mod_quat']
    return(data)
X_train.columns
X_train = normalise(X_train)
X_test = normalise(X_test)
print(X_test.shape)
#----------------------------------------------------------
## [quaternions to euler angles](https://quaternions.online/)
def quaternion_to_euler(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    X = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    Z = math.atan2(t3, t4)

    return(X, Y, Z)

def quaternion_to_euler_convert (data):
    
    x, y, z, w = data['norm_X'].tolist(), data['norm_Y'].tolist(), data['norm_Z'].tolist(), data['norm_W'].tolist()
    nx, ny, nz = [], [], []
    for i in range(len(x)):
        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])
        nx.append(xx)
        ny.append(yy)
        nz.append(zz)
    
    data['euler_x'] = nx
    data['euler_y'] = ny
    data['euler_z'] = nz
    return(data)

X_train = quaternion_to_euler_convert(X_train)
X_test = quaternion_to_euler_convert(X_test)
print(X_train.shape)
print(X_test.shape)
#----------------------------------------------------------
## Creating new columns
def feat_eng(data):
    
    df = pd.DataFrame()
    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5
    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5
    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5
    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']
    
    def mean_change_of_abs_change(x):
        return np.mean(np.diff(np.abs(np.diff(x))))
    
    for col in data.columns:
        if col in ['row_id','series_id','measurement_number']:
            continue
        df[col + '_mean'] = data.groupby(['series_id'])[col].mean()
        df[col + '_median'] = data.groupby(['series_id'])[col].median()
        df[col + '_max'] = data.groupby(['series_id'])[col].max()
        df[col + '_min'] = data.groupby(['series_id'])[col].min()
        df[col + '_std'] = data.groupby(['series_id'])[col].std()
        df[col + '_range'] = df[col + '_max'] - df[col + '_min']
        df[col + '_maxtoMin'] = df[col + '_max'] / df[col + '_min']
        df[col + '_mean_abs_chg'] = data.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))
        df[col + '_mean_change_of_abs_change'] = data.groupby('series_id')[col].apply(mean_change_of_abs_change)
        df[col + '_abs_max'] = data.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))
        df[col + '_abs_min'] = data.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))
        df[col + '_abs_avg'] = (df[col + '_abs_min'] + df[col + '_abs_max'])/2
    return(df)

X_train1 = feat_eng(X_train)
X_test1 = feat_eng(X_test)
print(X_train1.shape)
print(X_test1.shape)


#------------------------------------------------------------------------------------------
# Peaks of each column
xb = X_train[X_train["series_id"]==33]["linear_acceleration_X"].reset_index(drop=True)
indices = peakutils.indexes(xb, thres=0.02/max(xb), min_dist=10)
plt.plot(xb)
plt.scatter(x=indices,y=xb.loc[indices])
plt.hlines(y=xb.mean(),colors="r",xmin=0,xmax=128)
plt.savefig("peaks1.png")
plt.show()

def find_peaks(data):
    mydf = pd.DataFrame(columns = list(data.columns[3:]+'_peaks'))
    for i in (data["series_id"].drop_duplicates()).tolist():
        for col in data.columns[3:]:
            xb = data[data["series_id"]==i][col].reset_index(drop=True)
            indices = peakutils.indexes(xb, thres=0.02/max(xb), min_dist=10)
            mydf.loc[i,col+'_peaks'] = (xb.loc[indices]).tolist()
            # mydf.loc[i,col+"_peaks_mean"] = xb.loc[indices].mean()
    return(mydf)

peak_data_train = find_peaks(X_train)
peak_data_train["series_id"] = (X_train["series_id"].drop_duplicates()).tolist()
peak_data_train = peak_data_train[["series_id"]+peak_data_train.columns.tolist()[:-1]]
peak_data_train.head()

peak_data_test = find_peaks(X_test)
peak_data_test["series_id"] = (X_test["series_id"].drop_duplicates()).tolist()
peak_data_test = peak_data_test[["series_id"]+peak_data_test.columns.tolist()[:-1]]
#---------------------------------------------------------------
# number of peak values in each column
peak_data_train_count = peak_data_train.iloc[:,1:].applymap(len)
peak_data_train_count.columns = ['{0}_count'.format(x) for x in peak_data_train_count.columns.tolist()]
# peak_data_train_count.head()
X_train1 = pd.merge(X_train1,peak_data_train_count,left_index=True,right_index=True)

peak_data_test_count = peak_data_test.iloc[:,1:].applymap(len)
peak_data_test_count.columns = ['{0}_count'.format(x) for x in peak_data_test_count.columns.tolist()]
X_test1 = pd.merge(X_test1,peak_data_test_count,left_index=True,right_index=True)

# Percentage of series showing peaks for column
pd.DataFrame(peak_data_train_count.apply(lambda x : str(round(((peak_data_train.shape[0]- len(x[x==0]))/peak_data_train.shape[0])*100,3))+"%",axis=0),columns=["series with peak train"]).merge(
    pd.DataFrame(peak_data_test_count.apply(lambda x : str(round(((peak_data_test.shape[0]- len(x[x==0]))/peak_data_test.shape[0])*100,3))+"%",axis=0),columns=["series with peak test"]),left_index=True,right_index=True)

# Minimum of peak values
def min1(x):
    return(min(x,default=np.NaN))

peak_data_train_min = peak_data_train.iloc[:,1:].applymap(min1)
peak_data_train_min.columns = ['{0}_min'.format(x) for x in peak_data_train_min.columns.tolist()]
# peak_data_train_min.head()
X_train1 = pd.merge(X_train1,peak_data_train_min,left_index=True,right_index=True)

peak_data_test_min = peak_data_test.iloc[:,1:].applymap(min1)
peak_data_test_min.columns = ['{0}_min'.format(x) for x in peak_data_test_min.columns.tolist()]
X_test1 = pd.merge(X_test1,peak_data_test_min,left_index=True,right_index=True)

# Maximum of peak values
def max1(x):
    return(max(x,default=np.NaN))

peak_data_train_max = peak_data_train.iloc[:,1:].applymap(max1)
peak_data_train_max.columns = ['{0}_max'.format(x) for x in peak_data_train_max.columns.tolist()]
X_train1 = pd.merge(X_train1,peak_data_train_max,left_index=True,right_index=True)

peak_data_test_max = peak_data_test.iloc[:,1:].applymap(max1)
peak_data_test_max.columns = ['{0}_max'.format(x) for x in peak_data_test_max.columns.tolist()]
X_test1 = pd.merge(X_test1,peak_data_test_max,left_index=True,right_index=True)


df1 = X_train1[list(compress(X_train1.columns.tolist(), ["_mean" in s for s in X_train1.columns.tolist()]))].merge(peak_data_train,how='outer',left_index=True,right_index=True)
for col in [x[:-6] for x in peak_data_train.columns[1:]]:
    for i in X_train["series_id"].unique():
        df1.loc[i,col+"_peaks_above_mean"] = len(df1.loc[i,col+"_peaks"] > df1.loc[i,col+"_mean"])
# len(df1.loc[0,"orientation_YZ"+"_peaks"] > df1.loc[0,"orientation_YZ"])
df1.drop(columns=list(compress(df1.columns.tolist(), ["above_mean" not in s for s in df1.columns.tolist()])),inplace=True)
X_train1 = pd.merge(X_train1,df1,left_index=True,right_index=True)

df2 = X_test1[list(compress(X_test1.columns.tolist(), ["_mean" in s for s in X_test1.columns.tolist()]))].merge(peak_data_test,how='outer',left_index=True,right_index=True)
for col in [x[:-6] for x in peak_data_test.columns[1:]]:
    for i in X_test["series_id"].unique():
        df2.loc[i,col+"_peaks_above_mean"] = len(df2.loc[i,col+"_peaks"] > df2.loc[i,col+"_mean"])
df2.drop(columns=list(compress(df2.columns.tolist(), ["above_mean" not in s for s in df2.columns.tolist()])),inplace=True)
X_test1 = pd.merge(X_test1,df2,left_index=True,right_index=True)

# Mean of peak values
peak_data_train_mean = peak_data_train.iloc[:,1:].applymap(np.mean)
peak_data_train_mean.columns = ['{0}_mean'.format(x) for x in peak_data_train_mean.columns.tolist()]
# peak_data_train_mean.head()
X_train1 = pd.merge(X_train1,peak_data_train_mean,left_index=True,right_index=True)

peak_data_test_mean = peak_data_test.iloc[:,1:].applymap(np.mean)
peak_data_test_mean.columns = ['{0}_mean'.format(x) for x in peak_data_test_mean.columns.tolist()]
X_test1 = pd.merge(X_test1,peak_data_test_mean,left_index=True,right_index=True)


#---------------------------------------------------------------

print(X_train1.shape)
print(X_test1.shape)
print(y_train.shape)

#------------------------------------------------------------------------------------------------------------------

## Model

### Missing values
X_train1.isna().sum().sort_values(ascending=False)
X_test1.isna().sum().sort_values(ascending=False)

X_train_imputed = X_train1.fillna(X_train1.mean())
X_test_imputed = X_test1.fillna(X_test1.mean())

y_train_9 = y_train["surface"]
y_train_9.index = list(y_train_9.index)


## SGD Classifier
sgd_clf = SGDClassifier(random_state=99)
skfolds = StratifiedKFold(n_splits=3, random_state=22)

for train_index, test_index in skfolds.split(X_train_imputed, y_train_9):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train_imputed.loc[train_index]
    y_train_folds = (y_train_9.loc[train_index])
    X_test_fold = X_train_imputed.loc[test_index]
    y_test_fold = (y_train_9.loc[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    # print(rnd_clf.oob_score_) #
    print(n_correct / len(y_pred))
    print(confusion_matrix(y_test_fold,y_pred))

#------------------------------------------------------------------------------------------------------------------

# Random forest classifier
rnd_clf = RandomForestClassifier(random_state=22,n_estimators=200,n_jobs=-1,
    max_depth=15,warm_start = True,oob_score=True)
rnd_clf.fit(X_train_imputed, y_train_9)
y_pred = rnd_clf.predict(X_train_imputed)
n_correct = sum(y_pred == y_train_9)
print("oob score ",rnd_clf.oob_score_)
print(n_correct / len(y_pred))
print(confusion_matrix(y_train_9,y_pred))
print(pd.DataFrame(np.c_[X_train_imputed.columns,rnd_clf.feature_importances_],columns=["column","importance"]).sort_values("importance",ascending=False))


#------------------------------------------------------------------------------------------------------------------