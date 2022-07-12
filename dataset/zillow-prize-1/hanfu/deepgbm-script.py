import argparse, os, logging, random, time, gc, collections, pdb
import numpy as np
import pandas as pd
import category_encoders as ce
from tqdm import tqdm
from pathlib import Path

import math
import time
import scipy.sparse
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable



random.seed(14)
np.random.seed(14)
torch.cuda.manual_seed_all(14)

data_dir = Path('../input/zillow-prize-1')
num_cols = ['bathroomcnt','bedroomcnt','calculatedbathnbr','threequarterbathnbr','finishedfloor1squarefeet','calculatedfinishedsquarefeet','finishedsquarefeet6','finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15','finishedsquarefeet50','fireplacecnt','fullbathcnt','garagecarcnt','garagetotalsqft','latitude','longitude','lotsizesquarefeet','numberofstories','poolcnt','poolsizesum','roomcnt','unitcnt','yardbuildingsqft17','taxvaluedollarcnt','structuretaxvaluedollarcnt','landtaxvaluedollarcnt','taxamount','taxdelinquencyyear','yearbuilt']
cat_cols = ['architecturalstyletypeid', 'yearbuilt_cate', 'buildingqualitytypeid', 'propertyzoningdesc', 'regionidneighborhood', 'yardbuildingsqft26', 'fireplaceflag', 'propertycountylandusecode', 'hashottuborspa', 'basementsqft', 'fips', 'buildingclasstypeid', 'pooltypeid2', 'pooltypeid10', 'regionidcounty', 'heatingorsystemtypeid', 'rawcensustractandblock', 'censustractandblock', 'taxdelinquencyflag', 'airconditioningtypeid', 'pooltypeid7', 'regionidcity', 'regionidzip', 'decktypeid', 'typeconstructiontypeid', 'propertylandusetypeid', 'storytypeid']
label_col = ['logerror']

dtype_dict = dict()
for col in num_cols:
    dtype_dict[col] = 'float'
for col in cat_cols:
    dtype_dict[col] = 'str'

def prepare_rawdata():
    #return train, test
    print('read raw data')
    tr_2016 = pd.read_csv(data_dir / 'train_2016_v2.csv')
    pr_2016 = pd.read_csv(data_dir / 'properties_2016.csv', dtype=dtype_dict)
    tr_2017 = pd.read_csv(data_dir / 'train_2017.csv')
    pr_2017 = pd.read_csv(data_dir / 'properties_2017.csv', dtype=dtype_dict)
    te_2016  = pd.read_csv(data_dir / 'sample_submission.csv').rename(columns=dict(ParcelId='parcelid'))
    tr_merge_2016 = tr_2016.merge(pr_2016, how='left', on='parcelid')
    tr_merge_2017 = tr_2017.merge(pr_2017, how='left', on='parcelid')

    assert((te_2016['parcelid'].isin(pr_2016['parcelid'])).all())
    te_all = te_2016.merge(pr_2016, how='left', on='parcelid')

    assert((tr_merge_2016.columns == tr_merge_2017.columns).all())
    tr_all = tr_merge_2017.append(tr_merge_2016, ignore_index=True)

    # massage
    tr_all["yearbuilt_cate"] = tr_all["yearbuilt"].astype('str')
    te_all["yearbuilt_cate"] = te_all["yearbuilt"].astype('str')

    tr_all = tr_all[num_cols + cat_cols + label_col]
    te_all  = te_all[num_cols + cat_cols]
    del [[tr_2016,pr_2016,tr_2017,pr_2017,tr_merge_2016,tr_merge_2017, te_2016]]

#     print('change dtypes')
#     dtype_dict = dict()
#     for col in num_cols:
#         dtype_dict[col] = 'float'
#     for col in cat_cols:
#         dtype_dict[col] = 'str'
#     tr_final = tr_all.astype(dtype_dict, skipna=True)
#     te_final = te_all.astype(dtype_dict, skipna=True)

    # check if all cols in the model are in current dataframe
    # assert(all(col in train_all.columns.tolist() for col in all_col))
    # assert(all(col in test_all.columns.tolist()+['logerror'] for col in all_col))
    # def finddiff(list1, list2):
    #     for col in list2:
    #         if col not in list1:
    #             print(col)
    # finddiff(train_all.columns.tolist(), all_col)
    print('save train and test data to csv')
    tr_all.to_csv('train.csv', index=False)
    te_all.to_csv('test.csv', index=False)
    return tr_all, te_all

train, test = prepare_rawdata()

# check train vs. test distribution difference
# test_final.isna().sum()/test_final.shape[0] -(train_final.isna().sum()/train_final.shape[0])[:58]

# train = pd.read_csv('train.csv')
# test = pd.read_csv('test.csv')

print('save train labels')
# save label col
y_tr = train[label_col[0]].values.astype(np.float32)
# drop label col
train.drop(columns = label_col, inplace=True)


print('cat cols fill na with <unknown>, and assign <rare>')
unkn = '<UNK>'
rare = '<RARE>'
cat_thresh_rate = 0.9
cat_threshold = 10
# save_value_filter = dict()

for col in cat_cols:
    vc = train[col].value_counts()
    n_rows = vc.shape[0]
    save_value_list = list(vc.iloc[:int(n_rows*cat_thresh_rate)][vc>cat_threshold].index) + [unkn]
#     rm_values = set(vc.index) - set(save_value_filter[col])
    # first fill na
    # then fill rare based on train data stats
    for df in (train, test):
        df[col] = df[col].fillna(unkn)
        df[col] = df[col].map(lambda x: x if x in save_value_list else rare)

del vc
gc.collect()

#num data is edit in-place, so backup here
train_original = train.copy()
test_original = test.copy()
# making categorical dataset
train_cat = train.copy()
test_cat = test.copy()


print('cat df: num col fillna and bucketize')
n_bins = 32
save_num_bins = dict()
for col in num_cols:
    # bucketize cols to 32 bins at most
    #qcut is based on quantiles, each bin has same number of samples
    qlist = pd.qcut(train_cat[col], n_bins, labels=False, retbins=True, duplicates='drop')
    # use result directly in train data
    train_cat[col] = qlist[0].fillna(-1).astype('int')
    # use bucket ticks in test data
    test_cat[col] = pd.cut(test_cat[col], qlist[1], labels=False, include_lowest=True).fillna(-1).astype('int')

print('cat df: both cat and num cols ordinal encoding')
# then ordinal encode all cols
ord_encoder_cat = ce.ordinal.OrdinalEncoder(cols=cat_cols+num_cols)
train_cat = ord_encoder_cat.fit_transform(train_cat)
test_cat = ord_encoder_cat.transform(test_cat)



print('cat df: df-wise cumulative ordinal encoding')
catx_size = [] # a list
catx_tr = train_cat[cat_cols+num_cols].values.astype(np.float32)
catx_te = test_cat[cat_cols+num_cols].values.astype(np.float32)
for col in cat_cols+num_cols:
    catx_size.append(train_cat[col].max()) #ce.oe start with 1
print(sum(catx_size))

sum_feats = 0
for idx in range(len(catx_size)):
    catx_tr[:,idx] += sum_feats
    catx_te[:,idx] += sum_feats
    sum_feats += catx_size[idx]



np.save('catx_tr.npy',catx_tr) 
np.save('catx_te.npy',catx_te)
np.save('catx_size.npy',catx_size)
print('cat df saved')
del train_cat, test_cat
gc.collect()


# making numerical dataset
# set and save train set encoding values   
save_num_emb = dict()
save_cate_avgs = dict()
target_num_cols = list()
n_row = train.shape[0]
cat_thresh_rate = 0.9
cat_threshold = 10

print('num df: num cols fill na with mean')
# note train and test here is the original ones
for col in num_cols:
    train[col] = train[col].fillna(train[col].mean())
    save_num_emb[col] = dict(sum=train[col].sum(), rown=train[col].shape[0])


print('num df: cat cols ordinal encoding')
ord_encoder_num = ce.ordinal.OrdinalEncoder(cols=cat_cols)
train = ord_encoder_num.fit_transform(train)

print('num df: cat cols target encoding')
for col in cat_cols:
    feats = train[col].values
    labels = y_tr
    feat_encoding = dict(mean=[], count=[])
    #feat_temp_result = collections.defaultdict(lambda : [0, 0])
    save_cate_avgs[col] = collections.defaultdict(lambda : [0, 0])
    for i,row in enumerate(feats):
        # smoothing optional
        if row in save_cate_avgs[col]:
            # feat_temp_result[cur_feat][0] = 0.9*feat_temp_result[cur_feat][0] + 0.1*self.save_cate_avgs[item][cur_feat][0]/self.save_cate_avgs[item][cur_feat][1]
            # feat_temp_result[cur_feat][1] = 0.9*feat_temp_result[cur_feat][1] + 0.1*self.save_cate_avgs[item][cur_feat][1]/idx
            feat_encoding['mean'].append(save_cate_avgs[col][row][0]/save_cate_avgs[col][row][1])
            feat_encoding['count'].append(save_cate_avgs[col][row][1]/i)
        else:
            feat_encoding['mean'].append(0)
            feat_encoding['count'].append(0)
        save_cate_avgs[col][row][0] += labels[i]
        save_cate_avgs[col][row][1] += 1
    train[col+'_t_mean'] = feat_encoding['mean']
    train[col+'_t_count'] = feat_encoding['count']
    target_num_cols.append(col+'_t_mean')
    target_num_cols.append(col+'_t_count')

print('num df: num col to ndarray')
tr_encoded = None
for col in num_cols+target_num_cols:
    feats = train[col].values
    if tr_encoded is None:
        tr_encoded = feats.reshape((-1,1))
    else:
        tr_encoded = np.concatenate([tr_encoded,feats.reshape((-1,1))],axis=1)
    del feats
    gc.collect()
    
print('num df: cat col manual binary encode')
def unpackbits(x,num_bits):
    xshape = list(x.shape)
    x = x.reshape([-1,1])
    to_and = 2**np.arange(num_bits).reshape([1,num_bits])
    return (x & to_and).astype(bool).astype(int).reshape(xshape + [num_bits])

max_len = dict()
for col in cat_cols:
    feats = train[col].values
    feat_len = train[col].max()
    bit_len = len(bin(feat_len)) - 2
    max_len[col] = bit_len
    res = unpackbits(feats, bit_len).reshape((n_row,-1))
    tr_encoded = np.concatenate([tr_encoded,res],axis=1)
    del feats
    gc.collect()

print('num data: ndarray and standardize')
numx_tr_pre = np.array(tr_encoded).astype(np.float32)
numx_tr = (numx_tr_pre - np.mean(numx_tr_pre,axis=0))/(np.std(numx_tr_pre,axis=0) + 1e-5)
numy_tr = np.array(train[label_col[0]].values).reshape((-1,1)).astype(np.float32)

np.save('numx_tr.npy',numx_tr)
np.save('numy_tr.npy',numy_tr)

print('num df saved')

# deepgbm = deepfm + tree2nn = nn + fm + tree2nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# not sure why type_prefix
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    type_prefix = torch.cuda
else:
    type_prefix = torch
    
class DeepFM(nn.Module):
    '''
    fields: normal features in FM context, get splited to multiple features
    features: encoded variables from each fields; ex. if one-hot-encoded, n_feature = n_unique
    x: (10000, 10), that is fields
    y: (10000, 1)
    features: 200, each field is encoded to 20 features on average
    k: 5, the embedded dimension
    '''
    def __init__(self,x_sizes, k=5, nn_size=[32,32]):
        super(DeepFM,self).__init__()
        self.x_sizes = x_sizes
        self.n_fields = len(x_sizes)
        self.n_features = sum(x_sizes)
        self.k = k
        self.std = math.sqrt(1.0 / self.n_fields) # for normalize
        self.nn_size = nn_size
        
        # h = b + wx + v1v2x1x2
        self.bias = nn.Parameter(torch.randn(1))
        self.fm_1st_emb = nn.Embedding(self.n_features+1, 1) #(200,1), used as wx; plus one for features is 1-base indexd
        self.fm_1st_emb.weight.data.normal_(0,self.std) # init weights as normal(0, 1)
        self.fm_1st_do = nn.Dropout(0.5)
        
        self.fm_2nd_emb = nn.Embedding(self.n_features+1, k) #(200, k), embedded value is used as vx, v1x1v2x2 = (1,k) @ (k,1)
        self.fm_2nd_emb.weight.data.normal_(0,self.std)
        self.fm_2nd_do = nn.Dropout(0.5)
        
        self.nn_0_do = nn.Dropout(0.5)
        self.nn_1_fc = nn.Linear(self.n_fields * k, self.nn_size[0]) # (50,32) for first fc layer, all fields are flattened as a neuron
        self.nn_1_bn = nn.BatchNorm1d(self.nn_size[0])
        self.nn_1_do = nn.Dropout(0.5)
        self.nn_2_fc = nn.Linear(self.nn_size[0],self.nn_size[1])
        self.nn_2_bn = nn.BatchNorm1d(self.nn_size[1])
        self.nn_2_do = nn.Dropout(0.5)
        
        self.criterion = nn.MSELoss()
        
        
    def forward(self, x):
        bs = x.size(0)
        xint = x.long().view(bs*self.n_fields) # (100000,), also make input from float tensor to long tensor (int)
        relu = torch.nn.ReLU()
        
        fm_1st = self.fm_1st_emb(xint).view(bs, self.n_fields, -1) # (10000, 10)
        fm_1st = self.fm_1st_do(fm_1st)
        fm_1st_sum = torch.sum(fm_1st, 1).squeeze() # (10000,1) by row-wise sum
        
        fm_2nd = self.fm_2nd_emb(xint).view(bs,self.n_fields, -1) # (10000, 10, 5) from (100000,5)
        fm_2nd_sum = torch.sum(fm_2nd, 1) # (10000, 5) by row-wise sum and squeeze off column dimension
        fm_2nd_sum_sq = fm_2nd_sum * fm_2nd_sum # (10000, 5) by element-wise calculation
        fm_2nd_sq = fm_2nd * fm_2nd # (10000, 10, 5) by element-wise calculation
        fm_2nd_sq_sum = torch.sum(fm_2nd_sq, 1) # (10000, 5) by row-wise sum and squeeze off column dimension
        fm_2nd_out = 0.5 * (fm_2nd_sum_sq - fm_2nd_sq_sum) # (10000, 5)
        fm_2nd_out = self.fm_2nd_do(fm_2nd_out)
        fm_2nd_sum = torch.sum(fm_2nd_out, 1).squeeze() # (10000, 1)
        
        nn_in = fm_2nd.reshape(bs, -1) # (10000, 50), each field's single embed is a node
        nn = self.nn_0_do(nn_in)
        nn = self.nn_1_fc(nn)
        nn = self.nn_1_bn(nn)
        nn = relu(nn)
        nn = self.nn_1_do(nn)
        nn = self.nn_2_fc(nn)
        nn = self.nn_2_bn(nn)
        nn = relu(nn)
        nn = self.nn_2_do(nn)
        nn_sum = torch.sum(nn, 1).squeeze()
        
        deepfm_sum = fm_1st_sum + fm_2nd_sum + nn_sum + self.bias
        return deepfm_sum # no activation because this is regression task
    
    def true_loss(self, h, y):
        return self.criterion(h.view(-1), y.view(-1))
    
print('make deepfm model')
if torch.cuda.is_available():
    model_deepfm = DeepFM(catx_size).cuda()
else:
    model_deepfm = DeepFM(catx_size)

print('make deepfm optimizer')
opt_deepfm = torch.optim.Adam(model_deepfm.parameters())


print('make trees')

n_row = numx_tr.shape[0]
n_features = numx_tr.shape[1]
n_tree = 100
n_leaf = 64
n_slice = 10


tree_params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'num_class': 1,
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': n_leaf,
    'min_data': 40,
    'boost_from_average': True,
    'num_threads': 6,
    'feature_fraction': 0.8,
    'bagging_freq': 3,
    'bagging_fraction': 0.9,
    'learning_rate': 1e-3,
    'num_boost_round': n_tree,
    # 'early_stopping_rounds': 20,
}

print('train lgb trees')
tree_data = lgb.Dataset(numx_tr, numy_tr.reshape(-1), params=tree_params)
tree_model = lgb.train(tree_params, tree_data)
tree_model_dump = tree_model.dump_model()
# predict = tree_models.predict(numx_tr, raw_score=True).astype(np.float32)
pred_leaf = tree_model.predict(numx_tr, pred_leaf=True) # (n_row, n_tree) .reshape(n_row, -1)

print('tree 2 ndarray')
leaf_array = np.zeros([n_tree, n_leaf], dtype=np.float32)
for tree in range(n_tree):
    n_leaves = np.max(pred_leaf[:,tree]) + 1
    for leaf in range(n_leaves):
        leaf_array[tree][leaf] = tree_model.get_leaf_output(tree,leaf)

        
        
print('make tree part: t2nn model')


print('make deepgbm')

print('train cat model')

def TrainWithLog(train_x, train_y, model, opt,
                 epoch=5, batch_size=64,key=""):
#     if isinstance(test_x, scipy.sparse.csr_matrix):
#         test_x = test_x.todense()
    train_len = train_x.shape[0]
    global_iter = 0 # total iter number across epochs
    trn_batch_size = batch_size
    train_num_batch = math.ceil(train_len / trn_batch_size)
#     total_iterations = epoch * train_num_batch
#     start_time = time.time()
#     total_time = 0.0
#     min_loss = float("Inf")
#     # min_error = float("Inf")
#     max_auc = 0.0
    
    log_freq = 100
    
    for epoch in range(epoch):
        shuffled_indices = np.random.permutation(np.arange(train_x.shape[0]))
        Loss_trn_epoch = 0.0
        Loss_trn_log = 0.0
        log_st = 0
        for local_iter in range(train_num_batch): #iter in current epoch, previous calculated
            trn_st = local_iter * trn_batch_size  #start index
            trn_ed = min(train_len, trn_st + trn_batch_size) #end index
            batch_trn_x = train_x[shuffled_indices[trn_st:trn_ed]] #shuffled[start:end]
#             if isinstance(batch_trn_x, scipy.sparse.csr_matrix):
#                 batch_trn_x = batch_trn_x.todense()
            inputs = torch.from_numpy(batch_trn_x.astype(np.float32)).to(device) # x
            targets = torch.from_numpy(train_y[shuffled_indices[trn_st:trn_ed]]).to(device) # y
            model.train() #tell the model is in training mode, not predicting mode; so dropout is on
#             if train_x_opt is not None:
#                 inputs_opt = torch.from_numpy(train_x_opt[shuffled_indices[trn_st:trn_ed]].astype(np.float32)).to(device)
#                 outputs = model(inputs, inputs_opt)
#             else:
            outputs = model(inputs)
            opt.zero_grad()
#             if isinstance(outputs, tuple) and train_y_opt is not None:
#                 # targets_inner = torch.from_numpy(s_train_y_opt[trn_st:trn_ed,:]).to(device)
#                 targets_inner = torch.from_numpy(train_y_opt[shuffled_indices[trn_st:trn_ed],:]).to(device)
#                 loss_ratio = args.loss_init * max(0.3,args.loss_dr ** (epoch // args.loss_de))#max(0.5, args.loss_dr ** (epoch // args.loss_de))
#                 if len(outputs) == 3:
#                     loss_val = model.joint_loss(outputs[0], targets, outputs[1], targets_inner, loss_ratio, outputs[2])
#                 else:
#                     loss_val = model.joint_loss(outputs[0], targets, outputs[1], targets_inner, loss_ratio)
#                 loss_val.backward()
#                 loss_val = model.true_loss(outputs[0], targets)
#             elif isinstance(outputs, tuple):
#                 loss_val = model.true_loss(outputs[0], targets)
#                 loss_val.backward()
#             else:
            loss_val = model.true_loss(outputs, targets)
            loss_val.backward()
            opt.step()
            loss_val = loss_val.item()
            
            global_iter += 1
            Loss_trn_epoch += (trn_ed - trn_st) * loss_val
            Loss_trn_log += (trn_ed - trn_st) * loss_val
            
            if global_iter % log_freq == 0:
                print(key+"Epoch-{:0>3d} {:>5d} Batches, Step {:>6d}, Training Loss: {:>9.6f} (AllAvg {:>9.6f})"
                            .format(epoch, local_iter + 1, global_iter, Loss_trn_log/(trn_ed-log_st), Loss_trn_epoch/trn_ed))
                # trn_summ = tf.Summary()
                # trn_summ.value.add(tag=args.data+ "/Train/Loss", simple_value = Loss_trn_log/(trn_ed-log_st))
                # trn_writer.add_summary(trn_summ, global_iter)
                log_st = trn_ed
                Loss_trn_log = 0.0
                
                #for testing
#             if global_iter % test_freq == 0 or local_iter == train_num_batch - 1:
#                 if args.model == 'deepgbm' or args.model == 'd1':
#                     try:
#                         print('Alpha: '+str(model.alpha))
#                         print('Beta: '+str(model.beta))
#                     except:
#                         pass
#                 # tst_summ = tf.Summary()
#                 torch.cuda.empty_cache()
#                 test_loss, pred_y = EvalTestset(test_x, test_y, model, args.test_batch_size, test_x_opt)
#                 current_used_time = time.time() - start_time
#                 start_time = time.time()
#                 total_time += current_used_time
#                 remaining_time = (total_iterations - (global_iter) ) * (total_time / (global_iter))
#                 if args.task == 'binary':
#                     metrics = eval_metrics(args.task, test_y, pred_y)
#                     _, test_auc = metrics
#                     # min_error = min(min_error, test_error)
#                     max_auc = max(max_auc, test_auc)
#                     # tst_summ.value.add(tag=args.data+"/Test/Eval/Error", simple_value = test_error)
#                     # tst_summ.value.add(tag=args.data+"/Test/Eval/AUC", simple_value = test_auc)
#                     # tst_summ.value.add(tag=args.data+"/Test/Eval/Min_Error", simple_value = min_error)
#                     # tst_summ.value.add(tag=args.data+"/Test/Eval/Max_AUC", simple_value = max_auc)
#                     print(key+"Evaluate Result:\nEpoch-{:0>3d} {:>5d} Batches, Step {:>6d}, Testing Loss: {:>9.6f}, Testing AUC: {:8.6f}, Used Time: {:>5.1f}m, Remaining Time: {:5.1f}m"
#                             .format(epoch, local_iter + 1, global_iter, test_loss, test_auc, total_time/60.0, remaining_time/60.0))
#                 else:
#                 print(key+"Evaluate Result:\nEpoch-{:0>3d} {:>5d} Batches, Step {:>6d}, Testing Loss: {:>9.6f}, Used Time: {:>5.1f}m, Remaining Time: {:5.1f}m"
#                         .format(epoch, local_iter + 1, global_iter, test_loss, total_time/60.0, remaining_time/60.0))
#                 min_loss = min(min_loss, test_loss)
#                 # tst_summ.value.add(tag=args.data+"/Test/Loss", simple_value = test_loss)
#                 # tst_summ.value.add(tag=args.data+"/Test/Min_Loss", simple_value = min_loss)
#                 print("-------------------------------------------------------------------------------")
#                 # tst_writer.add_summary(tst_summ, global_iter)
#                 # tst_writer.flush()
#         print("Best Metric: %s"%(str(max_auc) if args.task=='binary' else str(min_loss)))
#         print("####################################################################################")
#     print("Final Best Metric: %s"%(str(max_auc) if args.task=='binary' else str(min_loss)))
#     return min_loss        
TrainWithLog(catx_tr, y_tr, model_deepfm, opt_deepfm)

torch.save(model_deepfm, 'model_deepfm')
torch.save({'state_dict':model_deepfm.state_dict()}, 'model_deepfm.pt.tar')
torch.save(opt_deepfm, 'opt_deepfm')

'''
model = DeepFM()
model_checkpoint = torch.load('model_deepfm.pth.tar')
model.load_state_dict(model_checkpoint['state_dict'])
'''
n_rows = catx_te.shape[0]
bs_test = 5000
n_batches = math.ceil(n_rows/bs_test)
sum_loss = 0.0
y_preds = []
model_deepfm.eval()
with torch.no_grad():
    for batch in range(n_batches):
        tst_st = batch * bs_test
        tst_ed = min(n_rows, tst_st + bs_test)
        inputs = torch.from_numpy(catx_te[tst_st:tst_ed].astype(np.float32)).to(device)
        outputs = model_deepfm(inputs)
        y_preds.append(outputs)
        
caty_te = np.concatenate(y_preds, 0)

print('making submission.csv')
submission = pd.read_csv(data_dir / 'sample_submission.csv')
for col in submission.columns[1:]:
    submission[col] = caty_te
submission.to_csv('submission.csv', index=False)
