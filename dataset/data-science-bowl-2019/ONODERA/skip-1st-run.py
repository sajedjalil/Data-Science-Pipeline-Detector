import pandas as pd

sub = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')
if len(sub)==1000:
    sub.to_csv('submission.csv', index=False)
    exit(0)

# =============================================================================
# write your code from here
# =============================================================================

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')
test = test[test.installation_id.isin(sub.installation_id)]

test.sort_values(['installation_id', 'timestamp'], inplace=True)
test = test[['installation_id', 'title']].drop_duplicates('installation_id', keep='last')
test.reset_index(drop=True, inplace=True)

di = {'Bird Measurer (Assessment)': 0,
 'Cart Balancer (Assessment)': 3,
 'Cauldron Filler (Assessment)': 3,
 'Chest Sorter (Assessment)': 0,
 'Mushroom Sorter (Assessment)': 3}

test['accuracy_group'] = test.title.map(di)
test[['installation_id', 'accuracy_group']].to_csv('submission.csv', index=False)



