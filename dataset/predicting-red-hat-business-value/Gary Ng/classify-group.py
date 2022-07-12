import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
from itertools import product
from scipy import interpolate


def interpolateFun0(x):
    """Original script author's function rewritten in Python.
    The author interpolates between two known values by averaging them. We
    can think of this as 0th order interpolation. """

    ## TODO: This function could use some optimization. The R version is much faster...
    x = x.reset_index(drop=True)
    g = x['outcome'] ## g should be a list or a pandas Series.
    if g.shape[0] < 3: ## If we have at most two rows.
        x['filled'] = g ## Will be replaced by a mean.
        return x
    missing_index = g.isnull()
    borders = np.append([g.index[0]], g[~missing_index].index, axis=0)
    borders = np.append(borders, [g.index[-1]+1], axis=0)
    forward_border = borders[1:]
    backward_border = borders[:-1]
    forward_border_g = g[forward_border]
    backward_border_g = g[backward_border]
    ## Interpolate borders.
    ## TODO: Why does the script author use the value 0.1?
    border_fill = 0.1
    forward_border_g[forward_border_g.index[-1]] = abs(forward_border_g[forward_border_g.index[-2]]-border_fill)
    backward_border_g[backward_border_g.index[0]] = abs(forward_border_g[forward_border_g.index[0]]-border_fill)
    times = forward_border-backward_border
    forward_x_fill = np.repeat(forward_border_g, times).reset_index(drop=True)
    backward_x_fill = np.repeat(backward_border_g, times).reset_index(drop=True)
    vec = (forward_x_fill+backward_x_fill)/2
    g[missing_index] = vec[missing_index] ## Impute missing values only.
    x['filled'] = g
    return x


def interpolateFun1(x):
    g = x['outcome']
    missing_index = g.isnull()
    border_fill = 0.8
    if g.index[0] in missing_index:
        g[g.index[0]] = border_fill
    if g.index[-1] in missing_index:
        g[g.index[-1]] = border_fill
    known_index = ~g.isnull()
    try:
        f = interpolate.interp1d(g[known_index].index,g[known_index],kind='linear')
        x['filled'] = [f(x) for x in g.index]
        x['filled'] = np.interp(g.index,g[known_index].index,g[known_index])
    except KeyError as e:
        x['filled'] = x['outcome']
    return x
    
    
## load and transform people data
ppl = pd.read_csv('../input/people.csv')

## convert bool to integer
ppl_logi = ppl.select_dtypes(include=['bool']).columns
ppl[ppl_logi] = ppl[ppl_logi].astype(int)
del ppl_logi

## transform date
ppl['date'] = pd.to_datetime(ppl['date'])

## load activites
## combine train and test data
activs = pd.read_csv('../input/act_train.csv')
TestActivs = pd.read_csv('../input/act_test.csv')
TestActivs['outcome'] = np.nan ## add the missing column to the test set
activs = pd.concat((activs,TestActivs),axis=0) ## append the train and test set
del TestActivs

## extract only required variables
activs = activs[['people_id','activity_id','date','outcome']]

## merge people into activity
d1 = pd.merge(activs,ppl,on='people_id',how='right')

## these are the indices of the rows from the test set
testset = ppl[ppl['people_id'].isin(d1[d1['outcome'].isnull()]['people_id'])].index
d1['activdate'] = pd.to_datetime(d1['date_x'])
del activs

## create group1 / day grid
min_activdate = d1['activdate'].min()
max_activdate = d1['activdate'].max()
print('min activdate : {}'.format(min_activdate))
print('max activdate : {}'.format(max_activdate))

## http://www.wklken.me/posts/2015/03/03/python-base-datetime.html
## make a list of all days from min to max
alldays = [max_activdate - datetime.timedelta(x) for x in range((max_activdate - min_activdate).days + 1)][::-1]

## http://wklken.me/posts/2013/08/20/python-extra-itertools.html#itertoolsproductiterables-repeat
## take the value of group_1 from the row of d1 which do not belong to the test set
grid_left = set(d1[~d1['people_id'].isin(ppl.iloc[testset]['people_id'])]['group_1'])
allCompanyAndDays = pd.DataFrame.from_records(product(grid_left,alldays))
allCompanyAndDays.columns = ['group_1','date_p']
allCompanyAndDays.sort_values(['group_1','date_p'],inplace=True)
print(allCompanyAndDays.head())

meanbydate = d1[~d1['people_id'].isin(ppl.iloc[testset]['people_id'])].groupby(['group_1','activdate'])['outcome'].agg('mean')
## convert series to dataframe
meanbydate = meanbydate.to_frame().reset_index()

## add them to full data
allCompanyAndDays = pd.merge(allCompanyAndDays,meanbydate,left_on=['group_1','date_p'],right_on=['group_1','activdate'],how='left')
allCompanyAndDays.drop('activdate',axis=1,inplace=True)
allCompanyAndDays.sort_values(['group_1','date_p'],inplace=True)
print(allCompanyAndDays.head())


#allCompanyAndDays = allCompanyAndDays.groupby('group_1').apply(interpolateFun0)
allCompanyAndDays = allCompanyAndDays.groupby('group_1').apply(interpolateFun1)
d1 = pd.merge(d1,allCompanyAndDays,left_on=['group_1','activdate'],right_on=['group_1','date_p'],how='left')

testset = d1[d1['people_id'].isin(ppl.iloc[testset]['people_id'])][['activity_id','filled']]
testset.columns = [testset.columns[0],'outcome']
testset['outcome'] = testset['outcome'].fillna(testset['outcome'].mean())
testset.to_csv('submission_ver2.csv',index=False)






