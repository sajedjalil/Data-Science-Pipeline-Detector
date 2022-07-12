import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# read train and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# remove constant columns
remove = []
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = train.columns
for i in range(len(c)-1):
    v = train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,train[c[j]].values):
            remove.append(c[j])

train.drop(remove, axis=1, inplace=True)
test.drop(remove, axis=1, inplace=True)
np.random.seed(10)
train = train.reindex(np.random.permutation(train.index),).reset_index(drop=True)
saved_features = np.array([['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var40_comer_ult1', 'imp_op_var40_comer_ult3', 'imp_op_var40_efect_ult1', 'imp_op_var40_efect_ult3', 'imp_op_var40_ult1', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'imp_sal_var16_ult1', 'ind_var1_0', 'ind_var1', 'ind_var5_0', 'ind_var5', 'ind_var6_0', 'ind_var6', 'ind_var8_0', 'ind_var8', 'ind_var12_0', 'ind_var12', 'ind_var13_0', 'ind_var13_corto_0', 'ind_var13_corto', 'ind_var13_largo_0', 'ind_var13_largo', 'ind_var13_medio_0', 'ind_var13', 'ind_var14_0', 'ind_var14', 'ind_var17_0', 'ind_var17', 'ind_var18_0', 'ind_var19', 'ind_var20_0', 'ind_var20', 'ind_var24_0', 'ind_var24', 'ind_var25_cte', 'ind_var26_0', 'ind_var26_cte', 'ind_var25_0', 'ind_var30_0', 'ind_var30', 'ind_var31_0', 'ind_var31', 'ind_var32_cte', 'ind_var32_0', 'ind_var33_0', 'ind_var33', 'ind_var34_0', 'ind_var37_cte', 'ind_var37_0', 'ind_var39_0', 'ind_var40_0', 'ind_var40', 'ind_var41_0', 'ind_var44_0', 'ind_var44', 'num_var1_0', 'num_var1', 'num_var4', 'num_var5_0', 'num_var5', 'num_var6_0', 'num_var6', 'num_var8_0', 'num_var8', 'num_var12_0', 'num_var12', 'num_var13_0', 'num_var13_corto_0', 'num_var13_corto', 'num_var13_largo_0', 'num_var13_largo', 'num_var13_medio_0', 'num_var13', 'num_var14_0', 'num_var14', 'num_var17_0', 'num_var17', 'num_var18_0', 'num_var20_0', 'num_var20', 'num_var24_0', 'num_var24', 'num_var26_0', 'num_var25_0', 'num_op_var40_hace2', 'num_op_var40_hace3', 'num_op_var40_ult1', 'num_op_var40_ult3', 'num_op_var41_hace2', 'num_op_var41_hace3', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_hace3', 'num_op_var39_ult1', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var31_0', 'num_var31', 'num_var32_0', 'num_var33_0', 'num_var33', 'num_var34_0', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var39_0', 'num_var40_0', 'num_var40', 'num_var41_0', 'num_var42_0', 'num_var42', 'num_var44_0', 'num_var44', 'saldo_var1', 'saldo_var5', 'saldo_var6', 'saldo_var8', 'saldo_var12', 'saldo_var13_corto', 'saldo_var13_largo', 'saldo_var13_medio', 'saldo_var13', 'saldo_var14', 'saldo_var17', 'saldo_var18', 'saldo_var20', 'saldo_var24', 'saldo_var26', 'saldo_var25', 'saldo_var30', 'saldo_var31', 'saldo_var32', 'saldo_var33', 'saldo_var34', 'saldo_var37', 'saldo_var40', 'saldo_var42', 'saldo_var44', 'var36', 'delta_imp_amort_var18_1y3', 'delta_imp_amort_var34_1y3', 'delta_imp_aport_var13_1y3', 'delta_imp_aport_var17_1y3', 'delta_imp_aport_var33_1y3', 'delta_imp_compra_var44_1y3', 'delta_imp_reemb_var13_1y3', 'delta_imp_reemb_var17_1y3', 'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_in_1y3', 'delta_imp_trasp_var33_out_1y3', 'delta_imp_venta_var44_1y3', 'delta_num_aport_var13_1y3', 'delta_num_aport_var17_1y3', 'delta_num_aport_var33_1y3', 'delta_num_compra_var44_1y3', 'delta_num_venta_var44_1y3', 'imp_amort_var18_ult1', 'imp_amort_var34_ult1', 'imp_aport_var13_hace3', 'imp_aport_var13_ult1', 'imp_aport_var17_hace3', 'imp_aport_var17_ult1', 'imp_aport_var33_hace3', 'imp_aport_var33_ult1', 'imp_var7_emit_ult1', 'imp_var7_recib_ult1', 'imp_compra_var44_hace3', 'imp_compra_var44_ult1', 'imp_reemb_var13_ult1', 'imp_reemb_var17_hace3', 'imp_reemb_var17_ult1', 'imp_reemb_var33_ult1', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'imp_trasp_var17_in_hace3', 'imp_trasp_var17_in_ult1', 'imp_trasp_var17_out_ult1', 'imp_trasp_var33_in_hace3', 'imp_trasp_var33_in_ult1', 'imp_trasp_var33_out_ult1', 'imp_venta_var44_hace3', 'imp_venta_var44_ult1', 'ind_var7_emit_ult1', 'ind_var7_recib_ult1', 'ind_var10_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var9_ult1', 'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'var21', 'num_aport_var13_hace3', 'num_aport_var13_ult1', 'num_aport_var17_hace3', 'num_aport_var17_ult1', 'num_aport_var33_hace3', 'num_aport_var33_ult1', 'num_var7_emit_ult1', 'num_var7_recib_ult1', 'num_compra_var44_hace3', 'num_compra_var44_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var8_ult3', 'num_meses_var12_ult3', 'num_meses_var13_corto_ult3', 'num_meses_var13_largo_ult3', 'num_meses_var13_medio_ult3', 'num_meses_var17_ult3', 'num_meses_var29_ult3', 'num_meses_var33_ult3', 'num_meses_var39_vig_ult3', 'num_meses_var44_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var40_comer_ult1', 'num_op_var40_comer_ult3', 'num_op_var40_efect_ult1', 'num_op_var40_efect_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_reemb_var13_ult1', 'num_reemb_var17_hace3', 'num_reemb_var17_ult1', 'num_reemb_var33_ult1', 'num_sal_var16_ult1', 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_trasp_var11_ult1', 'num_trasp_var17_in_hace3', 'num_trasp_var17_in_ult1', 'num_trasp_var17_out_ult1', 'num_trasp_var33_in_hace3', 'num_trasp_var33_in_ult1', 'num_trasp_var33_out_ult1', 'num_venta_var44_hace3', 'num_venta_var44_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_hace2', 'saldo_medio_var12_hace3', 'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2', 'saldo_medio_var13_corto_hace3', 'saldo_medio_var13_corto_ult1', 'saldo_medio_var13_corto_ult3', 'saldo_medio_var13_largo_hace2', 'saldo_medio_var13_largo_hace3', 'saldo_medio_var13_largo_ult1', 'saldo_medio_var13_largo_ult3', 'saldo_medio_var13_medio_hace2', 'saldo_medio_var13_medio_ult3', 'saldo_medio_var17_hace2', 'saldo_medio_var17_hace3', 'saldo_medio_var17_ult1', 'saldo_medio_var17_ult3', 'saldo_medio_var29_hace2', 'saldo_medio_var29_hace3', 'saldo_medio_var29_ult1', 'saldo_medio_var29_ult3', 'saldo_medio_var33_hace2', 'saldo_medio_var33_hace3', 'saldo_medio_var33_ult1', 'saldo_medio_var33_ult3', 'saldo_medio_var44_hace2', 'saldo_medio_var44_hace3', 'saldo_medio_var44_ult1', 'saldo_medio_var44_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var40_efect_ult1', 'imp_op_var40_ult1', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'imp_sal_var16_ult1', 'ind_var5_0', 'ind_var5', 'ind_var8_0', 'ind_var12_0', 'ind_var13', 'ind_var30_0', 'ind_var30', 'ind_var39_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_op_var41_hace2', 'num_op_var41_hace3', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_hace3', 'num_op_var39_ult1', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'num_var42', 'saldo_var5', 'saldo_var8', 'saldo_var12', 'saldo_var26', 'saldo_var25', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2', 'saldo_medio_var13_corto_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var40_efect_ult1', 'imp_op_var40_ult1', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'imp_sal_var16_ult1', 'ind_var5_0', 'ind_var5', 'ind_var8_0', 'ind_var12_0', 'ind_var13', 'ind_var30', 'ind_var39_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_op_var41_hace2', 'num_op_var41_hace3', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_hace3', 'num_op_var39_ult1', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'num_var42', 'saldo_var5', 'saldo_var8', 'saldo_var12', 'saldo_var26', 'saldo_var25', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var43_emit_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2', 'saldo_medio_var13_corto_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var40_efect_ult1', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'imp_sal_var16_ult1', 'ind_var5_0', 'ind_var5', 'ind_var8_0', 'ind_var12_0', 'ind_var13', 'ind_var30', 'ind_var39_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_op_var41_hace2', 'num_op_var41_hace3', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_hace3', 'num_op_var39_ult1', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'num_var42', 'saldo_var5', 'saldo_var8', 'saldo_var12', 'saldo_var26', 'saldo_var25', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var43_emit_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2', 'saldo_medio_var13_corto_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var40_efect_ult1', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'imp_sal_var16_ult1', 'ind_var5_0', 'ind_var5', 'ind_var8_0', 'ind_var12_0', 'ind_var13', 'ind_var30', 'ind_var39_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_op_var41_hace2', 'num_op_var41_hace3', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_hace3', 'num_op_var39_ult1', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'num_var42', 'saldo_var5', 'saldo_var8', 'saldo_var12', 'saldo_var26', 'saldo_var25', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2', 'saldo_medio_var13_corto_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var40_efect_ult1', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'imp_sal_var16_ult1', 'ind_var5_0', 'ind_var5', 'ind_var8_0', 'ind_var12_0', 'ind_var30', 'ind_var39_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_op_var41_hace2', 'num_op_var41_hace3', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_hace3', 'num_op_var39_ult1', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'num_var42', 'saldo_var5', 'saldo_var8', 'saldo_var12', 'saldo_var26', 'saldo_var25', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2', 'saldo_medio_var13_corto_ult3', 'var38']])
features = saved_features[4][:]

#saved_features = np.array([['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var40_comer_ult1', 'imp_op_var40_comer_ult3', 'imp_op_var40_efect_ult1', 'imp_op_var40_efect_ult3', 'imp_op_var40_ult1', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'imp_sal_var16_ult1', 'ind_var1_0', 'ind_var1', 'ind_var5_0', 'ind_var5', 'ind_var6_0', 'ind_var6', 'ind_var8_0', 'ind_var8', 'ind_var12_0', 'ind_var12', 'ind_var13_0', 'ind_var13_corto_0', 'ind_var13_corto', 'ind_var13_largo_0', 'ind_var13_largo', 'ind_var13_medio_0', 'ind_var13', 'ind_var14_0', 'ind_var14', 'ind_var17_0', 'ind_var17', 'ind_var18_0', 'ind_var19', 'ind_var20_0', 'ind_var20', 'ind_var24_0', 'ind_var24', 'ind_var25_cte', 'ind_var26_0', 'ind_var26_cte', 'ind_var25_0', 'ind_var30_0', 'ind_var30', 'ind_var31_0', 'ind_var31', 'ind_var32_cte', 'ind_var32_0', 'ind_var33_0', 'ind_var33', 'ind_var34_0', 'ind_var37_cte', 'ind_var37_0', 'ind_var39_0', 'ind_var40_0', 'ind_var40', 'ind_var41_0', 'ind_var44_0', 'ind_var44', 'num_var1_0', 'num_var1', 'num_var4', 'num_var5_0', 'num_var5', 'num_var6_0', 'num_var6', 'num_var8_0', 'num_var8', 'num_var12_0', 'num_var12', 'num_var13_0', 'num_var13_corto_0', 'num_var13_corto', 'num_var13_largo_0', 'num_var13_largo', 'num_var13_medio_0', 'num_var13', 'num_var14_0', 'num_var14', 'num_var17_0', 'num_var17', 'num_var18_0', 'num_var20_0', 'num_var20', 'num_var24_0', 'num_var24', 'num_var26_0', 'num_var25_0', 'num_op_var40_hace2', 'num_op_var40_hace3', 'num_op_var40_ult1', 'num_op_var40_ult3', 'num_op_var41_hace2', 'num_op_var41_hace3', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_hace2', 'num_op_var39_hace3', 'num_op_var39_ult1', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var31_0', 'num_var31', 'num_var32_0', 'num_var33_0', 'num_var33', 'num_var34_0', 'num_var35', 'num_var37_med_ult2', 'num_var37_0', 'num_var39_0', 'num_var40_0', 'num_var40', 'num_var41_0', 'num_var42_0', 'num_var42', 'num_var44_0', 'num_var44', 'saldo_var1', 'saldo_var5', 'saldo_var6', 'saldo_var8', 'saldo_var12', 'saldo_var13_corto', 'saldo_var13_largo', 'saldo_var13_medio', 'saldo_var13', 'saldo_var14', 'saldo_var17', 'saldo_var18', 'saldo_var20', 'saldo_var24', 'saldo_var26', 'saldo_var25', 'saldo_var30', 'saldo_var31', 'saldo_var32', 'saldo_var33', 'saldo_var34', 'saldo_var37', 'saldo_var40', 'saldo_var42', 'saldo_var44', 'var36', 'delta_imp_amort_var18_1y3', 'delta_imp_amort_var34_1y3', 'delta_imp_aport_var13_1y3', 'delta_imp_aport_var17_1y3', 'delta_imp_aport_var33_1y3', 'delta_imp_compra_var44_1y3', 'delta_imp_reemb_var13_1y3', 'delta_imp_reemb_var17_1y3', 'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_in_1y3', 'delta_imp_trasp_var33_out_1y3', 'delta_imp_venta_var44_1y3', 'delta_num_aport_var13_1y3', 'delta_num_aport_var17_1y3', 'delta_num_aport_var33_1y3', 'delta_num_compra_var44_1y3', 'delta_num_venta_var44_1y3', 'imp_amort_var18_ult1', 'imp_amort_var34_ult1', 'imp_aport_var13_hace3', 'imp_aport_var13_ult1', 'imp_aport_var17_hace3', 'imp_aport_var17_ult1', 'imp_aport_var33_hace3', 'imp_aport_var33_ult1', 'imp_var7_emit_ult1', 'imp_var7_recib_ult1', 'imp_compra_var44_hace3', 'imp_compra_var44_ult1', 'imp_reemb_var13_ult1', 'imp_reemb_var17_hace3', 'imp_reemb_var17_ult1', 'imp_reemb_var33_ult1', 'imp_var43_emit_ult1', 'imp_trans_var37_ult1', 'imp_trasp_var17_in_hace3', 'imp_trasp_var17_in_ult1', 'imp_trasp_var17_out_ult1', 'imp_trasp_var33_in_hace3', 'imp_trasp_var33_in_ult1', 'imp_trasp_var33_out_ult1', 'imp_venta_var44_hace3', 'imp_venta_var44_ult1', 'ind_var7_emit_ult1', 'ind_var7_recib_ult1', 'ind_var10_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'ind_var9_ult1', 'ind_var43_emit_ult1', 'ind_var43_recib_ult1', 'var21', 'num_aport_var13_hace3', 'num_aport_var13_ult1', 'num_aport_var17_hace3', 'num_aport_var17_ult1', 'num_aport_var33_hace3', 'num_aport_var33_ult1', 'num_var7_emit_ult1', 'num_var7_recib_ult1', 'num_compra_var44_hace3', 'num_compra_var44_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var8_ult3', 'num_meses_var12_ult3', 'num_meses_var13_corto_ult3', 'num_meses_var13_largo_ult3', 'num_meses_var13_medio_ult3', 'num_meses_var17_ult3', 'num_meses_var29_ult3', 'num_meses_var33_ult3', 'num_meses_var39_vig_ult3', 'num_meses_var44_ult3', 'num_op_var39_comer_ult1', 'num_op_var39_comer_ult3', 'num_op_var40_comer_ult1', 'num_op_var40_comer_ult3', 'num_op_var40_efect_ult1', 'num_op_var40_efect_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_reemb_var13_ult1', 'num_reemb_var17_hace3', 'num_reemb_var17_ult1', 'num_reemb_var33_ult1', 'num_sal_var16_ult1', 'num_var43_emit_ult1', 'num_var43_recib_ult1', 'num_trasp_var11_ult1', 'num_trasp_var17_in_hace3', 'num_trasp_var17_in_ult1', 'num_trasp_var17_out_ult1', 'num_trasp_var33_in_hace3', 'num_trasp_var33_in_ult1', 'num_trasp_var33_out_ult1', 'num_venta_var44_hace3', 'num_venta_var44_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_hace3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'saldo_medio_var12_hace2', 'saldo_medio_var12_hace3', 'saldo_medio_var12_ult1', 'saldo_medio_var12_ult3', 'saldo_medio_var13_corto_hace2', 'saldo_medio_var13_corto_hace3', 'saldo_medio_var13_corto_ult1', 'saldo_medio_var13_corto_ult3', 'saldo_medio_var13_largo_hace2', 'saldo_medio_var13_largo_hace3', 'saldo_medio_var13_largo_ult1', 'saldo_medio_var13_largo_ult3', 'saldo_medio_var13_medio_hace2', 'saldo_medio_var13_medio_ult3', 'saldo_medio_var17_hace2', 'saldo_medio_var17_hace3', 'saldo_medio_var17_ult1', 'saldo_medio_var17_ult3', 'saldo_medio_var29_hace2', 'saldo_medio_var29_hace3', 'saldo_medio_var29_ult1', 'saldo_medio_var29_ult3', 'saldo_medio_var33_hace2', 'saldo_medio_var33_hace3', 'saldo_medio_var33_ult1', 'saldo_medio_var33_ult3', 'saldo_medio_var44_hace2', 'saldo_medio_var44_hace3', 'saldo_medio_var44_ult1', 'saldo_medio_var44_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'imp_sal_var16_ult1', 'ind_var1_0', 'ind_var5_0', 'ind_var5', 'ind_var8_0', 'ind_var8', 'ind_var12_0', 'ind_var13_0', 'ind_var13', 'ind_var30_0', 'ind_var30', 'ind_var31_0', 'ind_var31', 'ind_var39_0', 'ind_var40_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_var20_0', 'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var31_0', 'num_var31', 'num_var35', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'saldo_var5', 'saldo_var8', 'saldo_var30', 'saldo_var31', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var40_comer_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_sal_var16_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var1_0', 'ind_var5_0', 'ind_var5', 'ind_var8_0', 'ind_var12_0', 'ind_var13_0', 'ind_var30_0', 'ind_var30', 'ind_var39_0', 'ind_var40_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_var20_0', 'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'saldo_var5', 'saldo_var8', 'saldo_var30', 'saldo_var31', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var1_0', 'ind_var5_0', 'ind_var8_0', 'ind_var12_0', 'ind_var13_0', 'ind_var30_0', 'ind_var30', 'ind_var39_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'saldo_var5', 'saldo_var8', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'ind_var10cte_ult1', 'ind_var9_cte_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_op_var39_efect_ult3', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_hace2', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var1_0', 'ind_var5_0', 'ind_var8_0', 'ind_var12_0', 'ind_var13_0', 'ind_var30_0', 'ind_var30', 'ind_var39_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'saldo_var5', 'saldo_var8', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'ind_var9_cte_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var1_0', 'ind_var5_0', 'ind_var8_0', 'ind_var12_0', 'ind_var30_0', 'ind_var30', 'ind_var39_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'saldo_var5', 'saldo_var8', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'ind_var9_cte_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_op_var39_efect_ult1', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'var38'], ['var3', 'var15', 'imp_ent_var16_ult1', 'imp_op_var39_comer_ult1', 'imp_op_var39_comer_ult3', 'imp_op_var41_comer_ult1', 'imp_op_var41_comer_ult3', 'imp_op_var41_efect_ult1', 'imp_op_var41_efect_ult3', 'imp_op_var41_ult1', 'imp_op_var39_efect_ult1', 'imp_op_var39_efect_ult3', 'imp_op_var39_ult1', 'ind_var1_0', 'ind_var5_0', 'ind_var8_0', 'ind_var12_0', 'ind_var30_0', 'ind_var30', 'ind_var39_0', 'ind_var41_0', 'num_var4', 'num_var5_0', 'num_var5', 'num_var8_0', 'num_op_var41_hace2', 'num_op_var41_ult1', 'num_op_var41_ult3', 'num_op_var39_ult3', 'num_var30_0', 'num_var30', 'num_var35', 'num_var39_0', 'num_var41_0', 'num_var42_0', 'saldo_var5', 'saldo_var8', 'saldo_var30', 'saldo_var37', 'saldo_var42', 'var36', 'imp_var43_emit_ult1', 'ind_var9_cte_ult1', 'num_ent_var16_ult1', 'num_var22_hace2', 'num_var22_hace3', 'num_var22_ult1', 'num_var22_ult3', 'num_med_var22_ult3', 'num_med_var45_ult3', 'num_meses_var5_ult3', 'num_meses_var39_vig_ult3', 'num_op_var41_comer_ult1', 'num_op_var41_comer_ult3', 'num_op_var41_efect_ult1', 'num_op_var41_efect_ult3', 'num_var45_hace2', 'num_var45_hace3', 'num_var45_ult1', 'num_var45_ult3', 'saldo_medio_var5_hace2', 'saldo_medio_var5_hace3', 'saldo_medio_var5_ult1', 'saldo_medio_var5_ult3', 'saldo_medio_var8_ult1', 'saldo_medio_var8_ult3', 'var38']])
#features = saved_features[3][:]
ind_features = list(np.array(features)[(train[features].describe().T['max']==1).values])
unhappy = train.loc[train['TARGET']==1, features + ['TARGET']]
happy = train.loc[train['TARGET']==0, features+['TARGET']]

unhappy_index = 2500
happy_index = unhappy_index*6
models = []
prob_train = []
target_train = []
prob_all_train = []
prob_cv_test = []
target_cv_test = []
prob_test = []

def smote(data, enlarge=4, ng=2, seed=10):
    clf = NearestNeighbors(n_neighbors=ng, algorithm='ball_tree')
    norm_data = normalize(data)
    clf.fit(norm_data)
    distance, index = clf.kneighbors(norm_data)
    
    np.random.seed(seed)
    
    records = []
    for i, row in enumerate(data):
        vector = []
        for v in index[i,1:]:
            vector.append(data[v,:] - row)
        vector = np.array(vector)
        for e in range(enlarge):
            step = np.random.rand(len(vector))
            move = np.dot(step, vector)
            new_record = row + move
            records.append(new_record)
    return np.array(records)

def smote_data(data, features, ind, enlarge=5, ng=2):
    more_data = smote(data[features].values, enlarge, ng)
    more_data_with_target = np.ones((more_data.shape[0],more_data.shape[1]+1))
    more_data_with_target[:,:-1] = more_data
    new_data = pd.DataFrame(more_data_with_target, columns=features + ['TARGET'])
    new_data[ind] = new_data[ind].applymap(lambda x: 0 if x<=(ng-1)/2 else 1)
    new_data = pd.concat([data, new_data], ignore_index=True)
    return new_data.reindex(np.random.permutation(new_data.index)).reset_index(drop=True)
    

for i in range(30):
    unhappy = unhappy.reindex(np.random.permutation(unhappy.index)).reset_index(drop=True)
    happy = happy.reindex(np.random.permutation(happy.index)).reset_index(drop=True)
    train_data = pd.concat([
            # smote_data(unhappy[0:unhappy_index], features, ind_features, 5, 50),
            unhappy[0: unhappy_index],
            unhappy[0:unhappy_index].sample(happy_index-unhappy_index, replace=True),
            happy[0:happy_index]
        ], ignore_index=True)
    valid_data = pd.concat([unhappy[unhappy_index:], happy[happy_index:]])
    clf = xgb.XGBClassifier(
        objective='binary:logistic', n_estimators=500, learning_rate=0.04, 
        max_depth=5, nthread=4, subsample=0.7, colsample_bytree=0.5, 
        reg_lambda=6, reg_alpha=5, seed=10, silent=True
    )
    clf.fit(
        train_data[features].values, train_data['TARGET'], eval_metric="auc",
        early_stopping_rounds=30, verbose=False,
        eval_set=[(valid_data[features].values, valid_data['TARGET'])]
    )
    prob_train.append(clf.predict_proba(train_data[features])[:,1])
    target_train.append(train_data['TARGET'].values)
    prob_all_train.append(clf.predict_proba(train[features])[:,1])
    prob_cv_test.append(clf.predict_proba(valid_data[features])[:,1])
    target_cv_test.append(valid_data['TARGET'].values)
    prob_test.append(clf.predict_proba(test[features])[:,1])
    
def get_auc(y_array, x_array):
    auc = []
    for y, x in zip(y_array, x_array):
        auc.append(metrics.roc_auc_score(y,x))
    print("AUC Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" 
         % (np.mean(auc),np.std(auc),np.min(auc),np.max(auc)))
print('*'*80)
print('AUC Training Data')
get_auc(target_train, prob_train)
print('*'*80)
print('AUC All Training Data')
get_auc([train['TARGET'].values]*len(prob_all_train), prob_all_train)
print('*'*80)
print('AUC Validation Data')
get_auc(target_cv_test, prob_cv_test)


test['TARGET'] = np.mean(prob_test, axis=0)

nv = test['num_var33']+test['saldo_medio_var33_ult3']+test['saldo_medio_var44_hace2']+test['saldo_medio_var44_hace3']+test['saldo_medio_var33_ult1']+test['saldo_medio_var44_ult1']
test.loc[nv > 0, 'TARGET'] = 0
test.loc[test['var15'] < 23, 'TARGET'] = 0
test.loc[test['saldo_medio_var5_hace2'] > 160000, 'TARGET'] = 0
test.loc[test['saldo_var33'] > 0, 'TARGET'] = 0
test.loc[test['var38'] > 3988596, 'TARGET'] = 0
#test.loc[test['var21'] > 7500, 'TARGET'] = 0
#test.loc[test['num_var30'] > 9, 'TARGET'] = 0
#test.loc[test['num_var13_0'] > 6, 'TARGET'] = 0
#test.loc[test['num_var33_0'] > 0, 'TARGET'] = 0
#test.loc[test['imp_ent_var16_ult1'] > 51003, 'TARGET'] = 0
#test.loc[test['imp_op_var39_comer_ult3'] > 13184, 'TARGET'] = 0
#test.loc[test['saldo_medio_var5_ult3'] > 108251, 'TARGET'] = 0
#test.loc[test['num_var37_0'] > 45, 'TARGET'] = 0
#test.loc[test['saldo_var5'] > 137615, 'TARGET'] = 0
#test.loc[test['saldo_var8'] > 60099, 'TARGET'] = 0
#test.loc[(test['var15']+test['num_var45_hace3']+test['num_var45_ult3']+test['var36']) <= 24, 'TARGET'] = 0
#test.loc[test['saldo_var14'] > 19053.78, 'TARGET'] = 0
#test.loc[test['saldo_var17'] > 288188.97, 'TARGET'] = 0
#test.loc[test['saldo_var26'] > 10381.29, 'TARGET'] = 0
#test.loc[test['num_var13_largo_0'] > 3, 'TARGET'] = 0
#test.loc[test['imp_op_var40_comer_ult1'] > 3639.87, 'TARGET'] = 0



train['TARGET2'] = np.mean(prob_all_train, axis=0)

nv = train['num_var33']+train['saldo_medio_var33_ult3']+train['saldo_medio_var44_hace2']+train['saldo_medio_var44_hace3']+train['saldo_medio_var33_ult1']+train['saldo_medio_var44_ult1']
train.loc[nv > 0, 'TARGET2'] = 0
train.loc[train['var15'] < 23, 'TARGET2'] = 0
train.loc[train['saldo_medio_var5_hace2'] > 160000, 'TARGET2'] = 0
train.loc[train['saldo_var33'] > 0, 'TARGET2'] = 0
train.loc[train['var38'] > 3988596, 'TARGET2'] = 0
#train.loc[train['var21'] > 7500, 'TARGET2'] = 0
#train.loc[train['num_var30'] > 9, 'TARGET2'] = 0
#train.loc[train['num_var13_0'] > 6, 'TARGET2'] = 0
#train.loc[train['num_var33_0'] > 0, 'TARGET2'] = 0
#train.loc[train['imp_ent_var16_ult1'] > 51003, 'TARGET2'] = 0
#train.loc[train['imp_op_var39_comer_ult3'] > 13184, 'TARGET2'] = 0
#train.loc[train['saldo_medio_var5_ult3'] > 108251, 'TARGET2'] = 0
#train.loc[train['num_var37_0'] > 45, 'TARGET2'] = 0
#train.loc[train['saldo_var5'] > 137615, 'TARGET2'] = 0
#train.loc[train['saldo_var8'] > 60099, 'TARGET2'] = 0
#train.loc[(train['var15']+train['num_var45_hace3']+train['num_var45_ult3']+train['var36']) <= 24, 'TARGET2'] = 0
#train.loc[train['saldo_var14'] > 19053.78, 'TARGET2'] = 0
#train.loc[train['saldo_var17'] > 288188.97, 'TARGET2'] = 0
#train.loc[train['saldo_var26'] > 10381.29, 'TARGET2'] = 0
#train.loc[train['num_var13_largo_0'] > 3, 'TARGET2'] = 0
#train.loc[train['imp_op_var40_comer_ult1'] > 3639.87, 'TARGET2'] = 0


print('AUC Before')
print(metrics.roc_auc_score(train['TARGET'], np.mean(prob_all_train, axis=0)))
print('AUC After')
print(metrics.roc_auc_score(train['TARGET'], train['TARGET2']))

output = pd.DataFrame()
output['ID'] = test['ID'].copy()
output['TARGET'] = np.mean(prob_test, axis=0)
output.to_csv('submission.csv', index=False)

output['TARGET'] = test['TARGET'].copy()
output.to_csv('submission2.csv', index=False)
