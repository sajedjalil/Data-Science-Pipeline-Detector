'''
    Approximate calculation of EF1. Based on algorythm from "Dembczy´nski, K., Waegeman, W., Cheng, W., H¨ullermeier, E.: An exact algorithm
    for F-measure maximization. In: Neural Information Processing Systems (2011)".
    
    by @kruegger
    version 2. Include 'None' as product with probability p_none
    
'''

'''
Usage:

=> df_group is dataframe with following structure:

    user_id	product_id	order_id	pred	true
0	    5	3376	    2196797	0.330466	0.0
1	    5	5999	    2196797	0.330330	0.0
2	    5	6808	    2196797	0.319509	0.0
3	    5	8518	    2196797	0.388290	0.0
4	    5	11777	    2196797	0.535934	0.0

pred - predictions from your model
true - ground truth

<= dataframe:

    ef1	        products
0	0.542255	11777 26604 24535 43693 40706 8518 21413 13988...


Scenario:

df - dataset with all predictions

df.head()
    user_id	product_id	order_id	pred	true
0	    5	3376	    2196797	0.330466	0.0
1	    5	5999	    2196797	0.330330	0.0
2	    5	6808	    2196797	0.319509	0.0
3	    5	8518	    2196797	0.388290	0.0
4	    5	11777	    2196797	0.535934	0.0
...

dfg = df.groupby(['order_id'])
df_ef1 = dfg.apply(lambda x: calc_approx_ef1(x))

df_ef1['products'].reset_index().to_csv(r'sub.csv', header=['order_id','products'], index=False)


''' 
def calc_approx_ef1(df_group):

    df = df_group.copy()
    order_id = np.int(df.iloc[0]['order_id'])
    
    df = df.sort_values('pred', ascending=False)[['product_id', 'pred', 'true']]
    products, preds = (zip(*df.sort_values('pred', ascending=False)[['product_id', 'pred']].values))
    _true = list(map(int, df['true'].values))
    pred_none = np.cumprod([1-x for x in preds])[-1]

    # add 'None' as product with p_none
    # ************************************************************
    products = list(products)[::-1]
    preds = list(preds)[::-1]
    ii = bisect.bisect(preds, pred_none)
    bisect.insort(preds, pred_none)
    products.insert(ii, 65535)
    products = products[::-1]
    preds = preds[::-1]
    # ************************************************************
    
    pi_sum = np.sum(preds)
    _len = len(products)
    mask = np.tril(np.ones((_len, _len)))
    hi_sum = mask.sum(1)

    phi_sum = 2*np.dot(mask, preds)
    ef1 = phi_sum / (pi_sum+hi_sum)

    ef1_max = np.max(ef1)

    prod_max = ' '.join(map(str, map(int, filter(bool, mask[np.argmax(ef1)]*products))))
    prod_list = prod_max.replace('65535', 'None') # if ef1_max > pred_none else 'None'

    return pd.DataFrame({'products':[prod_list], 'ef1':[ef1_max]})
