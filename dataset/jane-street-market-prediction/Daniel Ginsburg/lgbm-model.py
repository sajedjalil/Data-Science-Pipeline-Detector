# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgbm
import pickle

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

feature_cols = ['weight', 'feature_51', 'feature_43', 'feature_41', 'feature_39', 'feature_50', 'feature_46', 'feature_47', 'feature_52', 'feature_33', 'feature_48', 'feature_42', 'feature_37', 'feature_5', 'feature_121', 'feature_69', 'feature_31', 'feature_11', 'feature_3', 'feature_70', 'feature_13', 'feature_45', 'feature_86', 'feature_21', 'feature_44', 'feature_55', 'feature_104', 'feature_105', 'feature_57', 'feature_1', 'feature_29', 'feature_23']

col_list = ['date', 'weight', 'resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp', 'feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_12', 'feature_13', 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24', 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_30', 'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35', 'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_40', 'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45', 'feature_46', 'feature_47', 'feature_48', 'feature_49', 'feature_50', 'feature_51', 'feature_52', 'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57', 'feature_58', 'feature_59', 'feature_60', 'feature_61', 'feature_62', 'feature_63', 'feature_64', 'feature_65', 'feature_66', 'feature_67', 'feature_68', 'feature_69', 'feature_70', 'feature_71', 'feature_72', 'feature_73', 'feature_74', 'feature_75', 'feature_76', 'feature_77', 'feature_78', 'feature_79', 'feature_80', 'feature_81', 'feature_82', 'feature_83', 'feature_84', 'feature_85', 'feature_86', 'feature_87', 'feature_88', 'feature_89', 'feature_90', 'feature_91', 'feature_92', 'feature_93', 'feature_94', 'feature_95', 'feature_96', 'feature_97', 'feature_98', 'feature_99', 'feature_100', 'feature_101', 'feature_102', 'feature_103', 'feature_104', 'feature_105', 'feature_106', 'feature_107', 'feature_108', 'feature_109', 'feature_110', 'feature_111', 'feature_112', 'feature_113', 'feature_114', 'feature_115', 'feature_116', 'feature_117', 'feature_118', 'feature_119', 'feature_120', 'feature_121', 'feature_122', 'feature_123', 'feature_124', 'feature_125', 'feature_126', 'feature_127', 'feature_128', 'feature_129', 'ts_id']
med_list = [252.0, 1.07421875, 4.5180320739746094e-05, 6.580352783203125e-05, 0.00012159347534179688, 0.00015676021575927734, 0.00010752677917480469, 1.0, 0.1534423828125, 0.1021728515625, -0.0240631103515625, -0.025634765625, -0.0105133056640625, -0.01091766357421875, 0.1328125, 0.04180908203125, 0.03240966796875, 0.0269317626953125, 0.07550048828125, 0.0301361083984375, 0.08697509765625, 0.050445556640625, 0.00128936767578125, -0.00751495361328125, -0.005218505859375, -0.0114898681640625, -0.0025730133056640625, -0.012359619140625, -0.00020682811737060547, -0.008575439453125, 0.0207672119140625, 0.027191162109375, 0.007537841796875, 0.01305389404296875, 0.0237579345703125, 0.0268402099609375, 0.0296478271484375, 0.0299072265625, -0.0035762786865234375, -0.0064239501953125, -9.173154830932617e-05, 0.001708984375, -0.1212158203125, 0.2398681640625, 0.185302734375, -0.019561767578125, -0.061920166015625, 0.07952880859375, 0.1708984375, 0.122802734375, 0.0877685546875, 0.0643310546875, 0.02032470703125, 0.039398193359375, 0.1614990234375, -0.013214111328125, 0.0272674560546875, -0.02191162109375, 0.033447265625, 0.033447265625, 0.003566741943359375, 0.0284576416015625, 0.0244903564453125, 0.01259613037109375, 0.01360321044921875, 0.02130126953125, 0.0126495361328125, 0.0091705322265625, 0.0283203125, 0.027008056640625, 0.163818359375, 0.038116455078125, 0.081298828125, 0.0214996337890625, 0.0015802383422851562, 0.00382232666015625, 0.024139404296875, 0.0030727386474609375, 0.01114654541015625, 0.00287628173828125, 0.00719451904296875, 0.0139923095703125, 0.0135955810546875, -0.037811279296875, 0.0018167495727539062, 0.00615692138671875, -0.01812744140625, 0.032379150390625, -0.085693359375, -0.07122802734375, -0.06756591796875, -0.052032470703125, -0.046875, -0.035919189453125, 0.005695343017578125, 0.00646209716796875, -0.00786590576171875, 0.02655029296875, -0.057037353515625, -0.057525634765625, -0.05615234375, -0.07562255859375, -0.0474853515625, 0.044403076171875, 0.0014553070068359375, 0.00677490234375, 0.0280303955078125, 0.039215087890625, -0.06451416015625, -0.1014404296875, -0.11346435546875, -0.08624267578125, -0.035308837890625, 0.1534423828125, -0.0509033203125, 0.1319580078125, -0.09625244140625, 0.139404296875, -0.06402587890625, 0.2012939453125, -0.061737060546875, 0.163330078125, -0.075927734375, 1186544.0]
med_dict = dict()
for i in range(len(med_list)):
    med_dict[col_list[i]] = med_list[i]   

# get model
f = open('../input/best-model-2/best_model - Copy.pkl', 'rb')
model = pickle.load(f)
f.close()

 
import janestreet
env = janestreet.make_env() # initialize the environment
iter_test = env.iter_test() # an iterator which loops over the test set


for (test_df, sample_prediction_df) in iter_test:
    X = test_df[feature_cols].fillna(med_dict).values
    action = (model.predict(X) > 0)*1
    sample_prediction_df.action = int(action)
    env.predict(sample_prediction_df)
    
    
