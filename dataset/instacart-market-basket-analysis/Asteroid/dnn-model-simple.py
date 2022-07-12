''' Simple neural network model'''
import pandas as pd
import tensorflow as tf
import tempfile
import numpy as np

# Suppress warnings and logging info
tf.logging.set_verbosity(tf.logging.ERROR) 
root = 'ROOT TO INPUT CSV DATA FILES'

def input_fn(df):
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],
                                           values=df[k].values,
                                           dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = {**continuous_cols, **categorical_cols}
    label = tf.constant(df[LABEL_COLUMN].values)
    return feature_cols, label

def train_input_fn():
    return input_fn(DF_TRAIN)

def eval_input_fn():
    return input_fn(DF_EVAL)

df_complete = pd.read_csv(root + 'complete.csv', dtype = str)
df_prior = df_complete.loc[df_complete['eval_set'] == 'prior']
df_train = pd.read_csv(root + 'complete_train.csv', dtype = str)
df_test  = pd.read_csv(root + 'complete_test.csv', dtype = str)
df_test.drop(['product_id','add_to_cart_order','reordered','product_name','aisle_id',
              'department_id','aisle','department'], axis = 1, inplace = True)
products = (df_complete['product_id'].drop_duplicates()).astype(str).tolist()

# products sorted based on frequency
s = df_complete['product_id'].value_counts()
products = s.index.tolist()
#####################################

users_trainset   = (df_train['user_id'].drop_duplicates()).astype(str).tolist()
users_testset    = (df_test['user_id'].drop_duplicates()).astype(str).tolist()
n = len(products) # total number of available products

DF_RESUlTS_IF_PRODUCT_EXISTS = pd.DataFrame()
DF_RESUlTS_IF_PRODUCT_EXISTS['order_id'] = df_test['order_id']

LABEL_COLUMN = 'label'

# Categorical and Continous base columns.
CATEGORICAL_COLUMNS = ['order_dow', 'order_hour_of_day', 'user_id',
                       'same_prd_ever_purchased']
CONTINUOUS_COLUMNS  = ['same_prd_prior_counts', 'same_prd_popularity','days_since_prior_order',
                       'order_number']


START, END = 0, n  # START is an index but END refers to actual position in list!!!
counter = START # tracks currect index; starting from START
total = 0
inf = 10e3 # days_since_prior_order is set to a big number for the first order
skipped = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for product in products[START : END]:
        print('Product ID:', product)
        try:
            counter += 1
            ''' TRAIN DATA PREPARATION: '''
            
            train_sample_size_frac = 0.1 * 9       # fraction of ones in 'label'
            df_ones = df_train.loc[df_train['product_id'] == product].set_index('order_id')
            df_ones['label'] = 1
            df_ones_sample = df_ones.sample(frac = train_sample_size_frac)
            df_zeros = df_train.loc[df_train['product_id'] != product].set_index('order_id')
            df_zeros['label'] = 0
            n_ones_sample = df_ones_sample.shape[0]
            n_zeros_sample = n_ones_sample * 4    # number of zeros in 'label'
            df_zeros_sample = df_zeros.sample(n = n_zeros_sample)
            # main training dataframe for product_id = product:
            DF_TRAIN = df_ones_sample.append(df_zeros_sample)
            DF_TRAIN['label'] = (DF_TRAIN['label']).astype(int)
            
            print('\nTrain set size =', DF_TRAIN.shape[0])
            users = (DF_TRAIN['user_id'].drop_duplicates()).astype(str).tolist()
            # get prior acitivies of selected users in DF_TRAIN about product_id=product
            
            ####################################
            sub_prior_all = df_prior[df_prior.user_id.isin(users)].fillna(inf)
            sub_prior_YES = sub_prior_all.loc[sub_prior_all.product_id == product]
            sub_prior_NO  = sub_prior_all[~sub_prior_all.isin(sub_prior_YES).all(1)]
            temp_grouped = sub_prior_YES.groupby('user_id')
            same_prd_prior_counts = temp_grouped.user_id.count().rename('same_prd_prior_counts') # var: continuous
            DF_TRAIN = DF_TRAIN.set_index('user_id', drop = False).join(same_prd_prior_counts)
            temp_grouped = sub_prior_all.groupby('user_id')
            total_prd_prior_counts = temp_grouped.user_id.count().rename('total_prd_prior_counts')
            same_prd_popularity = (same_prd_prior_counts/total_prd_prior_counts).rename('same_prd_popularity')
            DF_TRAIN = DF_TRAIN.set_index('user_id', drop = False).join(same_prd_popularity)
            DF_TRAIN['same_prd_ever_purchased'] = 0
            DF_TRAIN['same_prd_ever_purchased'] = DF_TRAIN['same_prd_prior_counts'].apply(lambda x: min(x,1))
            DF_TRAIN['same_prd_ever_purchased'] = (DF_TRAIN['same_prd_ever_purchased']).astype(str)
            DF_TRAIN['days_since_prior_order'] = (DF_TRAIN['days_since_prior_order']).astype(float)
            DF_TRAIN['days_since_prior_order'] = DF_TRAIN['days_since_prior_order'].fillna(inf)
            DF_TRAIN['order_number'] = (DF_TRAIN['order_number']).astype(int)
            
            
            DF_TRAIN.fillna(0, inplace = True)
            if DF_TRAIN.shape[0] is 0: continue
            ####################################
            
            ''' EVALUATION DATA PREPARATION:
                        
                
            because the test dataset doesnt have 'label', to evaluate the model we took a sample from df_ones+df_zeros
            '''
            eval_sample_size_frac = 0.5 * train_sample_size_frac * 1.8
            df_temp1 = df_ones[~df_ones.isin(df_ones_sample).all(1)]
            # df_temp0 = df_zeros[~df_zeros.isin(df_zeros_sample).all(1)]
            df_temp0 = df_zeros.loc[df_zeros.index.difference(df_zeros_sample.index)]
            df_temp1 = df_temp1.sample(frac = eval_sample_size_frac)
            df_temp0 = df_temp0.sample(n = df_temp1.shape[0])
            DF_EVAL = df_temp1.append(df_temp0)
            DF_EVAL['label'] = (DF_EVAL['label']).astype(int)
            
            print('Eval set size =', DF_EVAL.shape[0])
            users = (DF_EVAL['user_id'].drop_duplicates()).astype(str).tolist()
            # get prior acitivies of selected users in DF_EVAL about product_id=product
            
            ##############################
            sub_prior_all = df_prior[df_prior.user_id.isin(users)].fillna(inf)
            sub_prior_YES = sub_prior_all.loc[sub_prior_all.product_id == product]
            sub_prior_NO  = sub_prior_all[~sub_prior_all.isin(sub_prior_YES).all(1)]
            temp_grouped = sub_prior_YES.groupby('user_id') # var: continuous
            same_prd_prior_counts = temp_grouped.user_id.count().rename('same_prd_prior_counts') # var: continuous
            DF_EVAL = DF_EVAL.set_index('user_id', drop = False).join(same_prd_prior_counts)
            temp_grouped = sub_prior_all.groupby('user_id')
            total_prd_prior_counts = temp_grouped.user_id.count().rename('total_prd_prior_counts')
            same_prd_popularity = (same_prd_prior_counts/total_prd_prior_counts).rename('same_prd_popularity')
            DF_EVAL = DF_EVAL.set_index('user_id', drop = False).join(same_prd_popularity)
            DF_EVAL['same_prd_ever_purchased'] = 0
            DF_EVAL['same_prd_ever_purchased'] = DF_EVAL['same_prd_prior_counts'].apply(lambda x: min(x,1))
            DF_EVAL['same_prd_ever_purchased'] = (DF_EVAL['same_prd_ever_purchased']).astype(str)
            DF_EVAL['days_since_prior_order'] = (DF_EVAL['days_since_prior_order']).astype(float)
            DF_EVAL['days_since_prior_order'] = DF_EVAL['days_since_prior_order'].fillna(inf)
            DF_EVAL['order_number'] = (DF_EVAL['order_number']).astype(int)
             
            
            DF_EVAL.fillna(0, inplace = True)
            if DF_EVAL.shape[0] is 0: continue
            ##############################
       
            
            ''' TEST DATA PREPARATION '''
            
    
           
            
            DF_TEST = df_test
            DF_TEST['label'] = -1
            DF_TEST['label'] = (DF_TEST['label']).astype(int)
            print('Test set size =', DF_TEST.shape[0])
            users = (DF_TEST['user_id'].drop_duplicates()).astype(str).tolist()
            # get prior acitivies of selected users in DF_TEST about product_id=product
            
            ###############################
            sub_prior_all = df_prior[df_prior.user_id.isin(users)].fillna(inf)
            sub_prior_YES = sub_prior_all.loc[sub_prior_all.product_id == product]
            sub_prior_NO  = sub_prior_all[~sub_prior_all.isin(sub_prior_YES).all(1)]
            temp_grouped = sub_prior_YES.groupby('user_id')
            same_prd_prior_counts = temp_grouped.user_id.count().rename('same_prd_prior_counts') # var: continuous
            DF_TEST = DF_TEST.set_index('user_id', drop = False).join(same_prd_prior_counts)
            temp_grouped = sub_prior_all.groupby('user_id')
            total_prd_prior_counts = temp_grouped.user_id.count().rename('total_prd_prior_counts')
            same_prd_popularity = (same_prd_prior_counts/total_prd_prior_counts).rename('same_prd_popularity') # var: continuous
            DF_TEST = DF_TEST.set_index('user_id', drop = False).join(same_prd_popularity)
            DF_TEST['same_prd_ever_purchased'] = 0
            DF_TEST['same_prd_ever_purchased'] = DF_TEST['same_prd_prior_counts'].apply(lambda x: min(x,1))
            DF_TEST['same_prd_ever_purchased'] = (DF_TEST['same_prd_ever_purchased']).astype(str)
            DF_TEST['days_since_prior_order'] = (DF_TEST['days_since_prior_order']).astype(float)
            DF_TEST['days_since_prior_order'] = DF_TEST['days_since_prior_order'].fillna(inf)
            DF_TEST['order_number'] = (DF_TEST['order_number']).astype(int)
            
            
            DF_TEST.fillna(0, inplace = True)
            #######################
             
    
            
            user_id = tf.contrib.layers.sparse_column_with_hash_bucket('user_id', 
                                                                         hash_bucket_size = 500000)
            order_dow = tf.contrib.layers.sparse_column_with_hash_bucket('order_dow', 
                                                                         hash_bucket_size = 10)
            order_hour_of_day = tf.contrib.layers.sparse_column_with_hash_bucket('order_hour_of_day', 
                                                                         hash_bucket_size = 30)
            same_prd_prior_counts = tf.contrib.layers.real_valued_column('same_prd_prior_counts')
            same_prd_popularity = tf.contrib.layers.real_valued_column('same_prd_popularity')
            days_since_prior_order = tf.contrib.layers.real_valued_column('days_since_prior_order')
            days_since_prior_order_buckets = tf.contrib.layers.bucketized_column(days_since_prior_order, boundaries=[5, 10, 20, 40])
            same_prd_ever_purchased = tf.contrib.layers.sparse_column_with_hash_bucket('same_prd_ever_purchased', 
                                                                         hash_bucket_size = 2)
            order_number = tf.contrib.layers.real_valued_column('order_number')    
                
            # DNN model
            FEATURE_COLUMNS = [tf.contrib.layers.embedding_column(user_id, dimension = 8),
                               tf.contrib.layers.embedding_column(order_dow, dimension = 8),
                               tf.contrib.layers.embedding_column(order_hour_of_day, dimension = 8),
                               same_prd_prior_counts,
                               same_prd_popularity,
                               tf.contrib.layers.embedding_column(same_prd_ever_purchased, dimension = 8),
                               order_number]                
                
            model_dir = tempfile.mkdtemp()
                
            HU = [15]
            
            NN = tf.contrib.learn.DNNClassifier(feature_columns = FEATURE_COLUMNS,
                                                   hidden_units = HU,
                                                   model_dir = model_dir,
                                                   dropout = None)
            
            
            
            FIT = NN.fit(input_fn = train_input_fn, steps = 5000)
            results = FIT.evaluate(input_fn = eval_input_fn, steps = 1)
            print('\n****** PRODUCT', counter, 'of', END-START, 'ANALYZED ******\n')
            for key in sorted(results):
                print("%s: %s" % (key, results[key]))  
            ## Prediction!!! 
            #predicted = FIT.predict(input_fn = lambda: input_fn(DF_TEST))
            #predicted_list = [key for key in predicted]
            proba = FIT.predict_proba(input_fn = lambda: input_fn(DF_TEST))
            proba_list = [key[1] for key in proba]
            threshold = 0.95
            proba_list_binary = []
            title = str(int(float(product)))
            for key in proba_list:
                if key >= threshold:
                    proba_list_binary.append(1)
                else:
                    proba_list_binary.append(0)
                    
            DF_RESUlTS_IF_PRODUCT_EXISTS[title] = proba_list_binary
            total = total + np.sum(proba_list_binary)
            print('\nTotal products predicted  ----> ', total,'\n')
            if counter % 100 == 0:
                fname = root+'DF_RESULTS_PRODUCTS_' + str(START+1) + '-' + str(counter) + '.csv'
                DF_RESUlTS_IF_PRODUCT_EXISTS.to_csv(fname , encoding = 'UTF-8', index = False)
    
            print('product', counter, 'of', len(products[START : END]), 'done!\n')
        except Exception:
            skipped.append(product)
            print('\nError: Unable to continue! Skipped products are saved in list skipped[]\n')
            fname = root+'DF_RESULTS_PRODUCTS_' + str(START+1) + '-' + str(counter) + '.csv'
            DF_RESUlTS_IF_PRODUCT_EXISTS.to_csv(fname , encoding = 'UTF-8', index = False)
            print('Computations carried out from product',START+1,'(at index',START,')to product',
                  counter,'(at index',counter-1,')')
            print('Output in the above range was saved in', fname)
            continue
    
    fname = root+'DF_RESULTS_PRODUCTS_' + str(START+1) + '-' + str(counter) + '.csv'
    DF_RESUlTS_IF_PRODUCT_EXISTS.to_csv(fname , encoding = 'UTF-8', index = False)
    print('Computations carried out from product',START+1,'(at index',START,')to product',counter,'(at index',counter-1,')')
    print('Output in the above range was saved in', fname)

    
