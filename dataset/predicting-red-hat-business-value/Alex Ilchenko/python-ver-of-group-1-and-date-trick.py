# --- original script author's comments.
## --- Alex Ilchenko's comments.  

import pandas as pd
import numpy as np
import datetime
from itertools import product
from scipy import interpolate ## For other interpolation functions.

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
    """First-order interpolation between known values. """
    g = x['outcome']
    missing_index = g.isnull()
    border_fill = 0.1 ## TODO: Shouldn't this be some kind of a mean value for the group?
    #border_fill = g.mean() ## Uncomment to try a different border fill.
    if g.index[0] in missing_index:
        g[g.index[0]] = border_fill
    if g.index[-1] in missing_index:
        g[g.index[-1]] = border_fill
    known_index = ~g.isnull()
    try:
        f = interpolate.interp1d(g[known_index].index, g[known_index], kind='linear')
        x['filled'] = [f(x) for x in g.index]
        x['filled'] = np.interp(g.index, g[known_index].index, g[known_index])
    except ValueError:
        x['filled'] = x['outcome']
    return x

if __name__ == '__main__':
    # Load and transform people data. ----------------------------------------------
    ppl = pd.read_csv('../input/people.csv')

    # Convert booleans to integers.
    p_logi = ppl.select_dtypes(include=['bool']).columns
    ppl[p_logi] = ppl[p_logi].astype('int')
    del p_logi

    # Transform date.
    ppl['date'] = pd.to_datetime(ppl['date'])

    # Load activities.--------------------------------------------------------------
    # Read and combine.
    activs = pd.read_csv('../input/act_train.csv')
    TestActivs = pd.read_csv('../input/act_test.csv')
    TestActivs['outcome'] = np.nan ## Add the missing column to the test set.
    activs = pd.concat([activs, TestActivs], axis=0) ## Append train and test sets.
    del TestActivs

    # Extract only required variables.
    activs = activs[['people_id', 'outcome', 'activity_id', 'date']] ## Let's look at these columns only.

    # Merge people data into activities.
    ## This keeps all the rows from activities.
    ## TODO: We are not using rows from ppl who have no activities...
    d1 = pd.merge(activs, ppl, on='people_id', how='right')

    ## These are the indices of the rows from the test set.
    testset = ppl[ppl['people_id'].isin(d1[d1['outcome'].isnull()]['people_id'])].index

    d1['activdate'] = pd.to_datetime(d1['date_x'])

    del activs

    # Prepare grid for prediction. -------------------------------------------------

    # Create all group_1/day grid.
    minactivdate = min(d1['activdate'])
    maxactivdate = max(d1['activdate'])

    ## Make a list of all days from min to max.
    alldays = [maxactivdate - datetime.timedelta(days=x) for x in range(0, (maxactivdate - minactivdate).days+1)][::-1]

    ## Take the values of group_1 from the rows of d1 which do not belong to the test set.
    grid_left = set(d1[~d1['people_id'].isin(ppl.iloc[testset]['people_id'])]['group_1'])
    ## Take cartesian product between the above variable and the list of all days.
    ## I think in the original script author thinks of the values in group_1 as companies.
    allCompaniesAndDays = pd.DataFrame.from_records(product(grid_left, alldays))

    # Nicer names.
    allCompaniesAndDays.columns = ['group_1', 'date_p']

    # Sort it.
    allCompaniesAndDays.sort_values(['group_1', 'date_p'], inplace=True)

    ## This is what allCompaniesAndDays looks like so far.
    """
    >>> allCompaniesAndDays.sample(n=10)
                  group_1     date_p
    10318543  group 14386 2023-08-09
    3470112    group 8767 2022-08-25
    5542924   group 30061 2023-01-11
    2328370   group 39750 2022-09-10
    7764760    group 1175 2022-12-12
    4788523    group 3788 2023-07-25
    5545711   group 12085 2022-10-13
    859359    group 28900 2023-07-21
    11188454  group 21110 2023-02-14
    9277889   group 26980 2023-08-07
    """

    # What are values on days where we have data?
    ## For a combination of group_1 and activdate, calculate the mean of the outcome variable.
    meanbycomdate = d1[~d1['people_id'].isin(ppl.iloc[testset]['people_id'])].groupby(['group_1', 'activdate'])['outcome'].agg('mean')
    ## Convert the calculation into a proper DataFrame.
    meanbycomdate = meanbycomdate.to_frame().reset_index()

    # Add them to full data grid.
    allCompaniesAndDays = pd.merge(allCompaniesAndDays, meanbycomdate, left_on=['group_1', 'date_p'], right_on=['group_1', 'activdate'], how='left')
    allCompaniesAndDays.drop('activdate', axis=1, inplace=True)
    allCompaniesAndDays.sort_values(['group_1', 'date_p'], inplace=True)

    ## This is what allCompaniesAndDays looks like so far.
    """
    >>> allCompaniesAndDays.sample(n=10)
                  group_1     date_p  outcome
    9536947   group 45684 2022-10-28      NaN
    11989016   group 8966 2022-12-10      NaN
    11113251   group 6012 2023-02-24      NaN
    9945551    group 4751 2023-01-06      1.0
    2273368   group 18350 2022-11-21      NaN
    12276013   group 9956 2023-04-08      NaN
    371765    group 11362 2023-02-23      NaN
    10065054  group 48049 2022-09-30      NaN
    5525397   group 29428 2023-06-06      NaN
    4911409   group 27233 2023-07-22      NaN
    """

    ## Add a column 'filled' which gives the imputed values for missing values in column 'outcome'.
    # groups = [df for _, df in list(allCompaniesAndDays.groupby('group_1'))]
    # dfs = [interpolateFun1(group) for group in groups]
    # allCompaniesAndDays = pd.concat(dfs)
    # allCompaniesAndDays.reset_index(drop=True, inplace=True)

    allCompaniesAndDays = allCompaniesAndDays.groupby('group_1').apply(interpolateFun0)

    d1 = pd.merge(d1, allCompaniesAndDays,
                  left_on=['group_1', 'activdate'], right_on=['group_1', 'date_p'], how='left')

    testsetdt = d1[d1['people_id'].isin(ppl.iloc[testset]['people_id'])][['activity_id', 'filled']]
    ## There are no NAs.
    testsetdt.columns = [testsetdt.columns[0], 'outcome']
    testsetdt['outcome'] = testsetdt['outcome'].fillna(testsetdt['outcome'].mean())
    testsetdt.to_csv('Submission.csv', index=False)
