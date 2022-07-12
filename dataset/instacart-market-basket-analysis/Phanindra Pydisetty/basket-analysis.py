import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from progress.bar import Bar
import matplotlib.pyplot as plt

# tools for model assesment
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV

class BasketAnalysis:
  def __init__(self, folderPath, threshold):
    self.DATA_FOLDER = folderPath
    self.THRESHOLD = threshold
    self.load_data()
    self.extract_relations() 
    return

  def load_data(self):
    # Load data into memory
    print('Loading products...')
    products = pd.read_csv(self.DATA_FOLDER + 'products.csv')

    print('Loading order products relation (prior) from order_products__prior...')
    priors = pd.read_csv(self.DATA_FOLDER + 'order_products__prior.csv')

    ### Get product order insights ###
    print('Calculate re-order rate for each order')
    product_order_rel = pd.DataFrame()

    product_order_rel['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
    print('Get reorder ratio for each product irrespective of the user')
    product_order_rel['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)

    product_order_rel['reorder_rate'] = (product_order_rel.reorders / product_order_rel.orders).astype(np.float32)
    print('Merge product_order_rel with products')
    products = products.join(product_order_rel, on='product_id')
    products.set_index(
      'product_id', 
      inplace=True, 
      drop=False,
      # append=False
    )

    print('Loading orders...')
    orders = pd.read_csv(self.DATA_FOLDER + 'orders.csv')

    ### Get order and prod information into priors dataframe
    print('Join orders table with priors to get user and order information to one place')
    orders.set_index(
      'order_id', 
      drop=False, 
      inplace=True,
      # append=False
    )
    priors = priors.join(
      # how='inner',
      orders, 
      rsuffix='_right', 
      on='order_id'
    )
    
    # # Remove duplicate join column
    priors.drop('order_id_right', inplace=True, axis=1)

    ### Get insights from user order history
    # 1. Total orders and total items so far
    # 2. Average gap between orders (in days)
    # 3. List of products they ordered
    ###
    print('Getting insights from user history from orders table...')
    user_history = pd.DataFrame()
    user_history['avg_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean()
    user_history['total_orders'] = orders.groupby('user_id').size()

    print('Getting insights from priors table based on user id...')
    users = pd.DataFrame()
    users['total_items'] = priors.groupby('user_id').size()
    # Get all unique products order by the user
    users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
    # Get count of the unique products
    users['total_unique_items'] = (users.all_products.map(len))

    users = users.join(user_history)
    print('Getting avg basket size of each user...')
    users['avg_cart_size'] = (users.total_items / users.total_orders)

    self.orders = orders
    self.priors = priors
    self.users = users
    self.products = products
    return
  
  def extract_relations(self):
    ###
    # Get user and product relation. 
    # 1. Number of times user bought the same product
    # 2. Sum product position in the cart as it provides valuable insight 
    #   - People tend to order important one's before adding others. 
    # 3. Total orders placed for by a user for each product
    #   - Helps in knowing the importance of the product to the user
    # 4. Last order they placed 
    ###
    print('Get user and product features')
    # up_rel -> relation between each user and each product that he ordered in past
    up_rel = self.priors.copy()
    # Build product_user column as unique identifier
    # Max product Id we have is less than 49000
    up_rel['user_product'] = up_rel.product_id + up_rel.user_id * 100000
    up_rel = up_rel.sort_values('order_number')
    up_rel = up_rel.groupby('user_product', sort=False) \
      .agg({
        'add_to_cart_order': 'sum',
        'order_id': ['size', 'last'] 
      })

    up_rel.columns = ['total_orders', 'last_order_id', 'cart_pos_sum']
    # Add col names and datatypes to new dataframe
    up_rel.astype({
      'total_orders': np.int16,
      'cart_pos_sum': np.int16,
      'last_order_id': np.int32
    }, inplace=True)

    self.up_rel = up_rel
    return

  def extract_features(self, orders, get_labels=False):
    ### Get user products that they might reorder ###
    print('Computing useful feature from given data...')
    order_list = []
    product_list = []
    labels = []
    size = len(orders)
    HOURS = 24 # day
    print('Processing %s records' % size)
    bar = Bar('Processing', max=size)

    for i, row in enumerate(orders.itertuples()):
      bar.next()
      user_id = row.user_id
      order_id = row.order_id
      user_prods = self.users['all_products'][user_id]
      product_list += user_prods
      order_list += [order_id] * len(user_prods)

      if get_labels:
        labels += [(order_id, prod) in self.train_index for prod in user_prods]

    # Trigger finish once labeling is done
    bar.finish()

    final_features = pd.DataFrame({
      'order_id': order_list, 
      'product_id': product_list
    }, dtype=np.int32)

    labels = np.array(labels, dtype=np.int8)

    print('Addding features related to product')
    final_features['aisle_id'] = final_features.product_id.map(self.products.aisle_id)
    final_features['department_id'] = final_features.product_id.map(self.products.department_id)
    final_features['product_orders'] = final_features.product_id.map(self.products.orders)
    final_features['product_reorders'] = final_features.product_id.map(self.products.reorders)
    final_features['product_reorder_rate'] = final_features.product_id.map(self.products.reorder_rate)

    print('Addding features related to user')
    final_features['user_id'] = final_features.order_id.map(self.orders.user_id)
    final_features['total_orders'] = final_features.user_id.map(self.users.total_orders)
    final_features['total_items'] = final_features.user_id.map(self.users.total_items)
    final_features['total_unique_items'] = final_features.user_id.map(self.users.total_unique_items)
    final_features['avg_days_between_orders'] = final_features.user_id.map(self.users.avg_days_between_orders)
    final_features['avg_cart_size'] = final_features.user_id.map(self.users.avg_cart_size)

    print('Addding features related to order')
    final_features['order_hour_of_day'] = final_features.order_id.map(self.orders.order_hour_of_day)
    final_features['days_since_prior_order'] = final_features.order_id.map(
        self.orders.days_since_prior_order)
    final_features['days_since_prior_order_ratio'] = (final_features.days_since_prior_order / final_features.avg_days_between_orders)

    print('User and product related features')
    # up -> short form for user product relation
    final_features['up_identifier'] = final_features.product_id + final_features.user_id * 100000

    final_features.drop(['user_id'], axis=1, inplace=True)

    final_features['up_last_order_id'] = final_features.up_identifier.map(self.up_rel.last_order_id)
    final_features['product_wise_total_orders'] = final_features.up_identifier.map(self.up_rel.total_orders)
    final_features['up_orders_ratio'] = (
        final_features.product_wise_total_orders / final_features.total_orders
    )
    final_features['up_avg_pos_in_cart'] = (
        final_features.up_identifier.map(self.up_rel.cart_pos_sum) / final_features.product_wise_total_orders)
    final_features['up_reorder_rate'] = (
        final_features.product_wise_total_orders / final_features.total_orders)
    final_features['up_orders_since_last'] = final_features.total_orders - final_features.up_last_order_id.map(self.orders.order_number)
    final_features['up_del_hour_vs_last'] = abs(
      final_features.order_hour_of_day - final_features.up_last_order_id.map(self.orders.order_hour_of_day)
    ).map(lambda t: min(t, HOURS - t))
    
    print('Drop unwanted columns')
    final_features.drop(['up_last_order_id', 'up_identifier'], axis=1, inplace=True)
    return final_features, labels

  def prepare_test_data(self):
    print('Extract test orders')
    test_set = self.orders[self.orders.eval_set == 'test']
    # Build feature set on test data
    data, _ = self.extract_features(test_set)  
    self.test_data = data
    self.test_set = test_set
    return

  def prepare_train_data(self):
    print('Extract orders to train')
    train_orders = self.orders[self.orders.eval_set == 'train']

    print('Loading order products relation (train) from order_products__train...')
    train_data = pd.read_csv(self.DATA_FOLDER + 'order_products__train.csv')

    train_data.set_index(['order_id', 'product_id'], inplace=True, drop=False)

    # Build set from train data for (order,product) tuples to speed up building labels
    self.train_index = set(train_data.index)

    del train_data

    # Create validation set for find best iteration
    train_bin, validation_bin = train_test_split(train_orders, test_size=0.25)

    train_set, tLabels = self.extract_features(train_bin, get_labels=True)
    validation_set, vLabels = self.extract_features(validation_bin, get_labels=True)

    self.features = [
      'total_items',
      'total_unique_items',
      'days_since_prior_order',
      'total_orders',
      'avg_days_between_orders',
      'order_hour_of_day',
      'avg_cart_size',
      'days_since_prior_order_ratio',
      'aisle_id',
      'department_id',
      'product_reorders',
      'product_orders',
      'product_reorder_rate',
      'product_wise_total_orders',
      'up_reorder_rate',
      'up_orders_ratio',
      'up_avg_pos_in_cart',
      'up_del_hour_vs_last',
      'up_orders_since_last',
    ]

    self.train_set = train_set
    self.tLabels = tLabels
    self.validation_set = validation_set
    self.vLabels = vLabels
    return

  def train_lgb(self):
    print('Build dataset for LightGBM (Model fitting)')
    lgb_train_dataset = lgb.Dataset(
      self.train_set[self.features],
      label=self.tLabels,
      free_raw_data=False,
      # https://lightgbm.readthedocs.io/en/latest/Quick-Start.html#categorical-feature-support
      categorical_feature=['aisle_id', 'department_id']
    )

    ROUNDS = 100
    params = {
      'task': 'train',
      'num_leaves': 90,
      'max_depth': 10,
      'max_bin': 512,
      'nthread': 5, 
      'bagging_freq': 5,
      'boosting_type': 'gbdt',
      'bagging_fraction': 0.95,
      'feature_fraction': 0.9,
      'metric': ['auc', 'binary_logloss'],
      'objective': 'binary'
    }

    print('Train LightGBM with %s data points' % len(self.train_set.index))
    self.model = lgb.train(
      params, 
      lgb_train_dataset,
      num_boost_round=ROUNDS,
      # valid_sets=[lgb_train_dataset, lgb_valid_dataset],
      verbose_eval = 4,
      # early_stopping_rounds = 50
    )

    return
  
  def train_xgboost(self):
    print('Build dataset for XGBoost')
    params = {
      "objective": "binary:logistic",
      'eval_metric': 'logloss',
      "eta": 0.05,
      "subsample": 0.7,
      "min_child_weight": 10,
      "colsample_bytree": 0.7,
      "max_depth": 5,
      "silent": 1,
      "seed": 0,
    }
    print ('params for XGBoost', params)

    ROUNDS = 50
    param_list = list(params.items())
    xgtrain = xgb.DMatrix(self.train_set, label=self.tLabels)
    print('Training using XGBoost...')
    self.model_xgb = xgb.train(
      param_list, 
      xgtrain, 
      ROUNDS
    )
    return

  def predict_lgb(self, data):   
    print('Running LightGBM prediction on provided data using created model')
    predictons = self.model.predict(
      data[self.features],
      # num_iteration=self.model.best_iteration
    )
    temp_df = data.copy()
    temp_df['prediction'] = predictons
    return temp_df, predictons

  def predict_xgboost(self, data):
    print('Running XGBoost prediction on provided data using created model')
    xgtest = xgb.DMatrix(data)
    predictions = self.model_xgb.predict(xgtest)

    temp_df = data.copy()
    temp_df['prediction'] = predictions
    # self.roc_curve(predictions, self.vLabels, 'Validation set')
    return temp_df, predictions

  def formatOutput(self, result, test_data):
    print('Building output in required format')
    output = dict()
    bar = Bar('Processing', max=(len(result.index) / 1000))
    for i, order in enumerate(result.itertuples()):
      
      if (i % 1000 == 0):
        bar.next()
      
      # form a string when we have multiple products for an order id
      if order.prediction > self.THRESHOLD:
        if order.order_id in output:
          output[order.order_id] += ' ' + str(order.product_id)
        else:
          output[order.order_id] = str(order.product_id)
    
    bar.finish()
    # Add missing orders in output with None indicator
    for order in test_data.order_id:
      if order not in output:
        output[order] = 'None'
    return output

  def plot_roc(self, tpred, fpred, label=''):
    print('Plot ROC curve')
    plt.plot(fpred, tpred, label=label)
    plt.legend()
    plt.ylabel('True positive rate (tpr)')
    plt.xlabel('False positive rate (fpr)')
    plt.show()

  def roc_curve(self, preds, labels, title):
    print('Plot ROC curve')
    # acc_score = accuracy_score(labels, np.round(preds))
    fpr, tpr, thresholdsTrain = roc_curve(labels, preds)
    area_under_curve =  auc(fpr, tpr)
 
    self.plot_roc(tpr, fpr, label=title)
    print(area_under_curve)
    return 

  def generate_stats(self, model, preds, labels):
    print('Generating stats...')
    # trainAcc = accuracy_score(labels, np.round(preds))
    fpr, tpr, _ = roc_curve(labels, preds)
    area_under_curve =  auc(fpr, tpr)
    self.plot_roc(tpr, fpr, label='Validation set')
    print('Stats')
    print(area_under_curve, tpr, fpr)
    # self.roc_curve(preds, labels, 'Validation set')
    return

  def parameter_tuning_lgb(self):
    params = {
      'num_leaves': [10, 15, 30, 90],
      'max_depth': [5, 10],
      'bagging_freq': [5],
      'boosting_type': ['gbdt'],
      'bagging_fraction': [0.9, 0.95],
      'feature_fraction': [0.8, 0.9],
      'metric': ['auc', 'binary_logloss'],
      'objective': ['binary']
    }

    # Initialize with default keys
    model = lgb.LGBMClassifier(
      nthread= 5, 
      max_depth= -1,
      max_bin= 128, 
      boosting_type= 'gbdt', 
      subsample= 1, 
      objective='binary', 
      subsample_for_bin= 500,
      subsample_freq= 1, 
      min_split_gain = 0.5, 
      min_child_weight = 1, 
      min_child_samples = 5, 
    )
    
    g_cv = GridSearchCV(model, params, cv=5)
    g_cv.fit(self.train_set, self.tLabels)
    
    print('GridSearchCV..')
    print(g_cv.best_params_, g_cv.best_score_)

    # Store fine tuned params and use it to make predictions
    self.lgb_best_params = g_cv.best_params_
    return

  def feature_stats(self):
    # Plot importance
    lgb.plot_importance(self.model, importance_type="split", title="split")
    plt.show()

    lgb.plot_importance(self.model, importance_type="gain", title='gain')
    plt.show()

    # Importance values are also available in:
    print(self.model.feature_importance("split"))
    print(self.model.feature_importance("gain"))
    return

  # Runs all steps sequentially and save to file
  def run(self, algo='lgb', tune=False, saveToFile=False):
    self.prepare_train_data()
    self.prepare_test_data()
    result = None
    
    if tune:
      self.parameter_tuning_lgb()

    if (algo == 'lgb'):
      self.train_lgb()
      # result, preds = self.predict_lgb(self.test_data)
      result, preds = self.predict_lgb(self.validation_set)
      self.feature_stats()
      self.generate_stats(self.model, preds, self.vLabels)
    else:
      self.train_xgboost()
      # result, preds = self.predict_xgboost(self.test_data)
      result, preds = self.predict_xgboost(self.validation_set)
      self.generate_stats(self.model_xgb, preds, self.vLabels)
    
    # Save to file for submitting results to kaggle
    if saveToFile:
      predictions = self.formatOutput(result, self.test_data)
      self.saveToFile(predictions, 'result.csv')
    
    return 

  # Train and test both lightGBM and XGBoost and plot the roc curves
  def compare(self): 
    self.prepare_train_data()
    self.prepare_test_data()

    plt.ylabel('True positive rate (tpr)')
    plt.xlabel('False positive rate (fpr)')
    
    self.train_lgb()
    result, preds = self.predict_lgb(self.validation_set)

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(self.vLabels, preds)
    area_under_curve =  auc(fpr, tpr)
    plt.plot(fpr, tpr, label='LightGBM AUC ' + str(area_under_curve))
    plt.legend()
    
    self.train_xgboost()
    result, preds = self.predict_xgboost(self.validation_set)
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(self.vLabels, preds)
    area_under_curve =  auc(fpr, tpr)
    plt.plot(fpr, tpr, label='XGBoost AUC ' + str(area_under_curve))    
    plt.legend()

    plt.show()
    return 

  def saveToFile(self, dic, filepath):
    print('Saving to file: %s' % filepath)
    # Prepare final result by converting dictionary to dataframe and save to a file
    result = pd.DataFrame.from_dict(dic, orient='index')
    result.reset_index(inplace=True)
    result.columns = ['order_id', 'products']
    result.to_csv(filepath, index=False)


defaultInputFolder = '../input/'
defaultThreshold = 0.2

analysis = BasketAnalysis(defaultInputFolder, defaultThreshold)
analysis.run('lgb')
# Uncomment below line to compare xgboost and lightgbm performance
# analysis.compare()