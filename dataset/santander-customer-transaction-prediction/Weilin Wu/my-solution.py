#!/usr/bin/env python
# coding: utf-8

import gc
import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.covariance import EllipticEnvelope

# load the data offered in this competetion
train_df = pd.read_csv("../input/santander-customer-transaction-prediction/train.csv")
test_df = pd.read_csv("../input/santander-customer-transaction-prediction/test.csv")

# seperate the target and the features
original_features = [col for col in train_df.columns if col not in ['ID_code', 'target']]
target = train_df['target']

# top_1k_important_features is a list that contains 1000 most important features.
# These features are found in the feature engineering process using LightGBM.
# The feature in the list can be an original feature, or the multiplication of 2 or 3
# original features.
top_1k_important_features = list(['var_78', 'var_164var_164', 'var_1', 'var_78var_78',
       'var_94var_94', 'var_94', 'var_82', 'var_164', 'var_184', 'var_53',
       'var_184var_184', 'var_82var_82', 'var_112var_112',
       'var_121var_57', 'var_1var_1', 'var_5var_5', 'var_53var_53',
       'var_49', 'var_18', 'var_116var_9', 'var_2', 'var_32var_32',
       'var_72var_92', 'var_58var_150', 'var_9', 'var_130var_130',
       'var_5', 'var_32', 'var_112', 'var_85_var_172_var_178',
       'var_20var_45', 'var_77_var_111_var_138', 'var_130', 'var_173',
       'var_20var_116', 'var_148var_120', 'var_37_var_184_var_184',
       'var_116var_45', 'var_51var_51', 'var_20var_169', 'var_20',
       'var_26var_101', 'var_180var_180', 'var_2var_2', 'var_141var_152',
       'var_40', 'var_63var_127', 'var_32_var_134_var_187', 'var_187',
       'var_199var_199', 'var_18var_70', 'var_132_var_197_var_198',
       'var_114var_102', 'var_180', 'var_95', 'var_99', 'var_63var_45',
       'var_137', 'var_196var_79', 'var_65var_155', 'var_22var_22',
       'var_70', 'var_148var_45', 'var_28_var_42_var_101',
       'var_149var_104', 'var_116var_156', 'var_76_var_142_var_169',
       'var_96_var_114_var_123', 'var_7_var_29_var_55', 'var_51',
       'var_196var_22', 'var_40var_40', 'var_18var_18',
       'var_85_var_154_var_192', 'var_76_var_80_var_101', 'var_85var_194',
       'var_24', 'var_11_var_95_var_147', 'var_68_var_104_var_149',
       'var_177var_63', 'var_156var_131', 'var_141var_9',
       'var_19_var_118_var_151', 'var_104var_152', 'var_49var_49',
       'var_196var_196', 'var_98_var_121_var_152',
       'var_85_var_150_var_197', 'var_151var_89', 'var_197var_150',
       'var_82var_48', 'var_136_var_190_var_190', 'var_91', 'var_134',
       'var_90', 'var_32var_180', 'var_179var_181', 'var_45var_102',
       'var_84var_189', 'var_111var_69', 'var_141var_127',
       'var_37_var_67_var_105', 'var_92var_44', 'var_53_var_79_var_144',
       'var_123var_133', 'var_114var_64', 'var_8', 'var_85var_34',
       'var_118var_151', 'var_177var_102', 'var_192var_120',
       'var_119var_138', 'var_81var_81', 'var_116var_50', 'var_151',
       'var_140var_5', 'var_192var_116', 'var_136var_190',
       'var_0_var_5_var_19', 'var_95var_157', 'var_128var_52',
       'var_162var_71', 'var_190', 'var_9var_121', 'var_149var_186',
       'var_81', 'var_38_var_52_var_71', 'var_196', 'var_33var_102',
       'var_190var_190', 'var_137_var_159_var_163', 'var_186var_156',
       'var_85var_169', 'var_174var_9', 'var_129var_107', 'var_22',
       'var_5var_69', 'var_169var_9', 'var_142var_120', 'var_24var_138',
       'var_159var_180', 'var_4_var_19_var_163', 'var_80',
       'var_165var_184', 'var_112_var_112_var_112', 'var_178',
       'var_97_var_99_var_159', 'var_68_var_88_var_148',
       'var_75_var_95_var_106', 'var_6var_106', 'var_56var_178', 'var_63',
       'var_137var_137', 'var_175var_62', 'var_67_var_177_var_177',
       'var_83_var_131_var_194', 'var_23_var_95_var_106', 'var_134var_89',
       'var_187var_32', 'var_119var_119', 'var_5var_65',
       'var_32_var_140_var_196', 'var_91var_91', 'var_140var_71',
       'var_67_var_133_var_133', 'var_70var_70', 'var_53var_51',
       'var_2var_21', 'var_144var_51', 'var_56var_149',
       'var_85_var_87_var_177', 'var_46_var_177_var_177', 'var_49var_18',
       'var_175var_82', 'var_162', 'var_125var_78',
       'var_135_var_140_var_199', 'var_5var_119', 'var_26var_0',
       'var_144', 'var_50var_107', 'var_159var_74', 'var_175var_123',
       'var_168var_195', 'var_55', 'var_36_var_123_var_166',
       'var_26var_26', 'var_77_var_104_var_146', 'var_95var_95', 'var_71',
       'var_149', 'var_36_var_154_var_165', 'var_127var_9', 'var_155',
       'var_32_var_83_var_85', 'var_31_var_104_var_149', 'var_34var_160',
       'var_2var_51', 'var_72var_75', 'var_52', 'var_60_var_88_var_188',
       'var_56var_50', 'var_198var_93', 'var_97_var_153_var_198',
       'var_118', 'var_193', 'var_47_var_119_var_119', 'var_58var_172',
       'var_188var_122', 'var_128var_128', 'var_46_var_195_var_199',
       'var_48_var_71_var_145', 'var_53var_164', 'var_149var_31',
       'var_127var_131', 'var_86var_73', 'var_165var_50', 'var_177',
       'var_195var_181', 'var_2var_48', 'var_147var_35', 'var_182var_77',
       'var_111var_164', 'var_136_var_145_var_193',
       'var_15_var_159_var_180', 'var_40var_52', 'var_11_var_63_var_153',
       'var_142var_58', 'var_133', 'var_135', 'var_100var_94',
       'var_186var_169', 'var_111', 'var_66_var_170_var_170',
       'var_171var_6', 'var_52var_180', 'var_36_var_165_var_174',
       'var_143var_21', 'var_177var_34', 'var_142', 'var_34var_9',
       'var_55_var_74_var_90', 'var_105var_5', 'var_56var_21', 'var_116',
       'var_46_var_186_var_192', 'var_136_var_148_var_183',
       'var_147var_157', 'var_80var_80', 'var_171var_157', 'var_110var_6',
       'var_173var_173', 'var_191var_95', 'var_128_var_147_var_147',
       'var_186var_93', 'var_16var_184', 'var_87_var_93_var_132',
       'var_22_var_78_var_155', 'var_24_var_31_var_77',
       'var_19_var_35_var_128', 'var_128', 'var_1var_180',
       'var_143var_154', 'var_145var_48', 'var_178var_83',
       'var_97_var_119_var_195', 'var_97_var_163_var_196',
       'var_171var_40', 'var_162var_6', 'var_119var_69', 'var_2var_170',
       'var_194var_197', 'var_98_var_164_var_164', 'var_13var_13',
       'var_92var_43', 'var_48_var_170_var_173', 'var_116var_150',
       'var_18_var_67_var_163', 'var_0_var_173_var_195', 'var_192var_193',
       'var_1_var_18_var_48', 'var_14_var_154_var_169', 'var_31var_54',
       'var_1_var_6_var_22', 'var_111var_52', 'var_125var_95',
       'var_94_var_133_var_155', 'var_169', 'var_99var_99',
       'var_128var_8', 'var_145', 'var_132var_156',
       'var_68_var_85_var_194', 'var_175var_151', 'var_140var_157',
       'var_144var_70', 'var_11_var_53_var_133', 'var_105var_3', 'var_33',
       'var_2var_133', 'var_92', 'var_169var_197', 'var_149var_93',
       'var_144var_52', 'var_95_var_120_var_142',
       'var_22_var_133_var_155', 'var_54_var_113_var_121',
       'var_91var_190', 'var_191var_180', 'var_99var_132',
       'var_177var_194', 'var_194var_114', 'var_169var_83',
       'var_165var_115', 'var_148var_43', 'var_177var_58', 'var_56var_9',
       'var_147', 'var_146var_142', 'var_118var_78', 'var_49var_89',
       'var_143var_139', 'var_187var_187', 'var_2var_89',
       'var_145var_163', 'var_154var_9', 'var_70var_94', 'var_94var_173',
       'var_166var_114', 'var_167var_173', 'var_4_var_18_var_49',
       'var_87var_148', 'var_137var_138', 'var_67var_51',
       'var_0_var_95_var_128', 'var_16', 'var_197var_81',
       'var_47_var_55_var_91', 'var_138_var_148_var_151',
       'var_11_var_18_var_125', 'var_35var_35', 'var_31_var_186_var_192',
       'var_93var_13', 'var_82var_128', 'var_59_var_139_var_160',
       'var_179var_106', 'var_156', 'var_148var_156',
       'var_37_var_48_var_126', 'var_26', 'var_162var_53',
       'var_86_var_115_var_122', 'var_75_var_95_var_95', 'var_185var_24',
       'var_187var_164', 'var_76_var_88_var_107',
       'var_31_var_131_var_156', 'var_74_var_145_var_167',
       'var_74_var_118_var_184', 'var_148var_92',
       'var_11_var_106_var_168', 'var_87_var_102_var_138',
       'var_20var_122', 'var_31_var_104_var_107',
       'var_112_var_133_var_145', 'var_133var_133',
       'var_11_var_90_var_147', 'var_97', 'var_175var_0', 'var_97var_62',
       'var_4_var_9_var_57', 'var_15_var_26_var_70',
       'var_23_var_64_var_104', 'var_34var_148',
       'var_108_var_109_var_115', 'var_4_var_5_var_94', 'var_113var_193',
       'var_191', 'var_3_var_140_var_176', 'var_144var_2',
       'var_123var_44', 'var_38_var_45_var_148', 'var_133var_70',
       'var_110var_0', 'var_52_var_177_var_177', 'var_92var_28',
       'var_146var_72', 'var_6var_170', 'var_48_var_135_var_140',
       'var_24var_48', 'var_162var_105', 'var_104var_81',
       'var_112var_145', 'var_123var_174', 'var_179', 'var_67var_8',
       'var_33var_172', 'var_5var_78', 'var_169var_44', 'var_199',
       'var_171var_176', 'var_177var_33', 'var_1var_6', 'var_2var_164',
       'var_123var_104', 'var_159var_95', 'var_49var_52', 'var_119var_51',
       'var_33var_13', 'var_149var_81', 'var_56var_172', 'var_177var_177',
       'var_10_var_157_var_168', 'var_69', 'var_76var_169',
       'var_149var_58', 'var_40var_78', 'var_26var_97', 'var_125var_151',
       'var_26var_6', 'var_198', 'var_90var_90', 'var_21var_198',
       'var_53var_90', 'var_66var_170', 'var_37_var_125_var_135',
       'var_36_var_64_var_172', 'var_114var_123', 'var_110var_189',
       'var_129', 'var_31_var_93_var_149', 'var_19_var_99_var_99',
       'var_33_var_34_var_75', 'var_28_var_33_var_109', 'var_145var_133',
       'var_18_var_94_var_94', 'var_44var_131', 'var_137var_199',
       'var_18_var_177_var_177', 'var_167var_130',
       'var_7_var_114_var_123', 'var_87_var_92_var_115',
       'var_112_var_112_var_134', 'var_110', 'var_87var_56',
       'var_96_var_186_var_194', 'var_93var_132', 'var_67var_119',
       'var_86_var_146_var_198', 'var_175var_106',
       'var_75_var_127_var_198', 'var_74_var_131_var_192',
       'var_77_var_110_var_195', 'var_74_var_135_var_145',
       'var_76_var_122_var_131', 'var_67_var_106_var_196',
       'var_145var_145', 'var_135var_90', 'var_167var_90', 'var_91var_55',
       'var_130var_5', 'var_25var_111', 'var_168var_48', 'var_169var_21',
       'var_95_var_137_var_140', 'var_76_var_122_var_132',
       'var_108_var_133_var_143', 'var_134_var_145_var_155',
       'var_34var_43', 'var_53var_5', 'var_113_var_139_var_169',
       'var_76var_58', 'var_113_var_150_var_198',
       'var_108_var_122_var_177', 'var_114', 'var_2var_135',
       'var_1_var_37_var_133', 'var_11_var_93_var_132',
       'var_12_var_13_var_33', 'var_162var_128', 'var_175var_181',
       'var_54_var_150_var_172', 'var_88var_107', 'var_143', 'var_168',
       'var_0_var_5_var_91', 'var_144var_53', 'var_62', 'var_178var_23',
       'var_190var_69', 'var_191var_164', 'var_162var_94',
       'var_7_var_8_var_173', 'var_130var_138', 'var_129var_81',
       'var_17var_94', 'var_149var_142', 'var_40var_62', 'var_127',
       'var_186var_104', 'var_54_var_54_var_83', 'var_175var_51',
       'var_110var_65', 'var_2var_35', 'var_95_var_104_var_139',
       'var_19_var_92_var_149', 'var_146var_45', 'var_105var_18',
       'var_52var_119', 'var_26var_195', 'var_167var_128',
       'var_146var_50', 'var_19_var_91_var_184', 'var_26_var_167_var_190',
       'var_193var_102', 'var_192var_75', 'var_193var_139',
       'var_11_var_62_var_147', 'var_32_var_71_var_180', 'var_6',
       'var_11_var_40_var_133', 'var_6var_173', 'var_14_var_63_var_107',
       'var_2var_78', 'var_8_var_24_var_49', 'var_66_var_130_var_191',
       'var_38_var_49_var_157', 'var_66_var_111_var_195', 'var_21var_58',
       'var_12_var_13_var_34', 'var_152', 'var_106var_180',
       'var_5var_190', 'var_99var_70', 'var_151var_7', 'var_108var_197',
       'var_99var_179', 'var_26var_130', 'var_177var_9', 'var_155var_155',
       'var_81var_44', 'var_168var_157', 'var_87var_80', 'var_162var_49',
       'var_40var_170', 'var_186var_172', 'var_151var_170',
       'var_53var_78', 'var_191var_191', 'var_45var_172', 'var_146var_34',
       'var_78var_199', 'var_86_var_107_var_174', 'var_26var_22',
       'var_33_var_34_var_86', 'var_197var_33', 'var_167var_111',
       'var_67_var_140_var_196', 'var_26var_110', 'var_6var_95',
       'var_31_var_78_var_99', 'var_110var_52', 'var_28_var_33_var_131',
       'var_163', 'var_1var_111', 'var_195var_111', 'var_167var_18',
       'var_179var_179', 'var_0var_24', 'var_82var_89', 'var_192var_31',
       'var_147var_94', 'var_4_var_24_var_157', 'var_12_var_21_var_92',
       'var_68_var_71_var_89', 'var_45', 'var_134var_138',
       'var_10_var_107_var_127', 'var_111var_51', 'var_138var_89',
       'var_94_var_118_var_196', 'var_95_var_127_var_141',
       'var_95_var_135_var_140', 'var_177var_101', 'var_85var_129',
       'var_31_var_172_var_192', 'var_28_var_44_var_81',
       'var_22_var_105_var_179', 'var_21var_50', 'var_53_var_106_var_157',
       'var_24var_74', 'var_32_var_71_var_91', 'var_149var_107',
       'var_98_var_153_var_183', 'var_21var_139', 'var_84var_62',
       'var_92var_139', 'var_32_var_71_var_162', 'var_177var_43',
       'var_22_var_97_var_128', 'var_122var_132', 'var_147var_199',
       'var_58var_43', 'var_127var_121', 'var_167var_157',
       'var_1_var_8_var_78', 'var_109var_9', 'var_3_var_130_var_191',
       'var_36_var_141_var_197', 'var_53_var_168_var_175',
       'var_147var_66', 'var_194var_88', 'var_54_var_150_var_197',
       'var_163var_106', 'var_22_var_90_var_184', 'var_110var_135',
       'var_31_var_150_var_197', 'var_83var_9', 'var_61_var_147_var_173',
       'var_122var_148', 'var_12_var_13_var_56', 'var_82var_133',
       'var_136_var_139_var_193', 'var_67_var_86_var_88', 'var_188var_13',
       'var_32_var_155_var_184', 'var_149var_149', 'var_119', 'var_60',
       'var_6_var_167_var_195', 'var_6var_180', 'var_170var_184',
       'var_175var_29', 'var_49_var_53_var_78', 'var_66_var_180_var_184',
       'var_163var_163', 'var_135var_6', 'var_11_var_90_var_105',
       'var_99var_167', 'var_66_var_138_var_191', 'var_78var_164',
       'var_6var_6', 'var_1_var_13_var_76', 'var_28_var_81_var_146',
       'var_87var_145', 'var_196var_78', 'var_188var_109',
       'var_168var_180', 'var_68_var_137_var_163', 'var_174var_132',
       'var_52var_155', 'var_91var_6', 'var_4_var_64_var_115',
       'var_0_var_13_var_80', 'var_33var_50', 'var_109var_13',
       'var_171var_184', 'var_94_var_118_var_147', 'var_134var_170',
       'var_34var_131', 'var_63var_12', 'var_86_var_109_var_182',
       'var_24var_51', 'var_121var_13', 'var_118var_180', 'var_160',
       'var_18_var_110_var_133', 'var_39', 'var_90var_97',
       'var_54_var_114_var_131', 'var_87_var_93_var_174', 'var_78var_97',
       'var_98_var_121_var_121', 'var_28var_102', 'var_69_var_78_var_190',
       'var_192var_13', 'var_28_var_43_var_107', 'var_21var_21',
       'var_149var_120', 'var_138_var_151_var_171', 'var_169var_198',
       'var_146var_21', 'var_26var_180', 'var_199var_157',
       'var_171var_111', 'var_180var_94', 'var_75_var_186_var_192',
       'var_28_var_34_var_109', 'var_166var_150', 'var_112var_26',
       'var_111var_163', 'var_60_var_93_var_154', 'var_95_var_95_var_184',
       'var_13', 'var_135var_74', 'var_61_var_118_var_125',
       'var_94_var_157_var_190', 'var_87_var_92_var_165', 'var_198var_13',
       'var_95var_51', 'var_53_var_131_var_156', 'var_5var_18',
       'var_54_var_122_var_188', 'var_189', 'var_97_var_148_var_188',
       'var_61_var_156_var_197', 'var_27_var_40_var_91', 'var_167var_51',
       'var_61_var_127_var_127', 'var_167var_0', 'var_69_var_86_var_97',
       'var_0var_128', 'var_67', 'var_72', 'var_67var_35',
       'var_110_var_130_var_191', 'var_0_var_53_var_167',
       'var_109_var_123_var_166', 'var_10_var_140_var_158',
       'var_11_var_68_var_114', 'var_4_var_87_var_142', 'var_87var_172',
       'var_179var_135', 'var_67_var_104_var_174',
       'var_61_var_85_var_127', 'var_33_var_56_var_131',
       'var_109_var_150_var_198', 'var_109', 'var_56var_13',
       'var_106var_133', 'var_12var_9', 'var_15_var_49_var_91',
       'var_24var_82', 'var_86_var_122_var_188', 'var_42var_132',
       'var_4var_155', 'var_26_var_179_var_184', 'var_178var_44',
       'var_193var_115', 'var_77_var_99_var_99', 'var_19_var_114_var_169',
       'var_186var_21', 'var_174var_92', 'var_40var_119', 'var_149var_21',
       'var_32_var_151_var_191', 'var_127var_13', 'var_11var_195',
       'var_198var_33', 'var_52_var_169_var_186', 'var_21var_104',
       'var_96_var_111_var_138', 'var_55_var_71_var_90', 'var_19var_8',
       'var_96_var_114_var_132', 'var_66_var_133_var_179',
       'var_95_var_130_var_191', 'var_132var_107', 'var_195var_155',
       'var_167var_162', 'var_66var_180', 'var_110var_164',
       'var_67_var_107_var_188', 'var_7_var_115_var_148', 'var_15',
       'var_164var_51', 'var_74_var_147_var_147', 'var_179var_163',
       'var_55_var_70_var_90', 'var_110var_110', 'var_137var_97',
       'var_1var_91', 'var_24var_151', 'var_135_var_177_var_177',
       'var_171var_164', 'var_49var_130', 'var_83_var_194_var_197',
       'var_38_var_55_var_67', 'var_134_var_157_var_163',
       'var_0_var_32_var_118', 'var_140', 'var_87_var_92_var_149',
       'var_76_var_90_var_97', 'var_42', 'var_44',
       'var_53_var_133_var_190', 'var_23_var_58_var_123', 'var_26var_179',
       'var_21', 'var_2var_5', 'var_4_var_9_var_21', 'var_148var_172',
       'var_169var_148', 'var_52_var_170_var_170', 'var_175var_22',
       'var_88', 'var_110_var_133_var_173', 'var_43var_132',
       'var_166var_143', 'var_125var_40', 'var_11_var_15_var_71',
       'var_193var_121', 'var_77var_33', 'var_22_var_106_var_168',
       'var_67_var_155_var_190', 'var_14_var_123_var_176',
       'var_31_var_123_var_131', 'var_34var_44', 'var_47_var_104_var_121',
       'var_2var_106', 'var_5var_95', 'var_60_var_93_var_115', 'var_138',
       'var_12var_107', 'var_36_var_80_var_108', 'var_15_var_99_var_190',
       'var_109var_150', 'var_128var_90', 'var_14_var_169_var_192',
       'var_136_var_193_var_194', 'var_26var_95', 'var_0var_5',
       'var_3_var_135_var_145', 'var_20var_127', 'var_1_var_35_var_67',
       'var_195var_94', 'var_11_var_18_var_49', 'var_139var_107',
       'var_10_var_171_var_173', 'var_137var_5', 'var_56var_33',
       'var_54_var_139_var_146', 'var_147var_91', 'var_87var_102',
       'var_118var_190', 'var_105var_164', 'var_141var_83',
       'var_116var_21', 'var_22var_173', 'var_0_var_89_var_190',
       'var_127var_107', 'var_169var_13', 'var_85_var_87_var_107',
       'var_98_var_122_var_188', 'var_53_var_128_var_173',
       'var_119var_164', 'var_70var_173', 'var_24var_94',
       'var_112var_118', 'var_155var_81', 'var_4_var_11_var_120',
       'var_53_var_91_var_147', 'var_178var_109',
       'var_59_var_121_var_132', 'var_4_var_6_var_89',
       'var_53_var_99_var_190', 'var_140var_196', 'var_38_var_50_var_75',
       'var_192var_188', 'var_36_var_139_var_197', 'var_4_var_18_var_70',
       'var_40var_22', 'var_87_var_99_var_145', 'var_9var_80',
       'var_114var_121', 'var_145var_78', 'var_24_var_26_var_53',
       'var_54_var_108_var_141', 'var_48_var_85_var_115',
       'var_31_var_177_var_177', 'var_95_var_118_var_130',
       'var_165var_58', 'var_33_var_34_var_81', 'var_31_var_81_var_165',
       'var_66_var_177_var_177', 'var_99var_78', 'var_76_var_146_var_174',
       'var_26var_184', 'var_112_var_191_var_199',
       'var_19_var_140_var_151', 'var_26_var_191_var_195',
       'var_177var_12', 'var_25var_69', 'var_0_var_32_var_180',
       'var_66var_95', 'var_40var_130', 'var_26var_155', 'var_147var_195',
       'var_19_var_80_var_148', 'var_112_var_138_var_196',
       'var_47_var_150_var_197', 'var_144var_71', 'var_3_var_105_var_191',
       'var_83_var_154_var_174', 'var_11_var_36_var_92', 'var_2var_6',
       'var_92var_121', 'var_23var_121', 'var_125',
       'var_85_var_104_var_127', 'var_134var_151',
       'var_66_var_105_var_180', 'var_65', 'var_175var_40',
       'var_15_var_154_var_197', 'var_91var_5', 'var_97_var_110_var_110',
       'var_15_var_40_var_118', 'var_28_var_67_var_110', 'var_92var_13',
       'var_165', 'var_54_var_93_var_139', 'var_26var_8', 'var_56',
       'var_25var_74', 'var_165var_142', 'var_127var_150',
       'var_32_var_95_var_199', 'var_18_var_85_var_177',
       'var_37_var_43_var_109', 'var_89var_89', 'var_0var_70',
       'var_49var_164', 'var_36_var_141_var_188', 'var_56var_154',
       'var_102var_107', 'var_34var_178', 'var_144var_130',
       'var_187var_163', 'var_68var_156', 'var_187var_191',
       'var_59_var_149_var_186', 'var_135var_95',
       'var_60_var_106_var_145', 'var_0var_119', 'var_7_var_71_var_162',
       'var_21var_81', 'var_11_var_87_var_88', 'var_0_var_127_var_131',
       'var_60_var_81_var_148', 'var_106var_74', 'var_133var_155',
       'var_106var_61', 'var_86_var_130_var_190', 'var_4_var_6_var_71',
       'var_141var_121', 'var_7_var_155_var_184', 'var_197var_43',
       'var_137var_11', 'var_199var_184', 'var_5var_94',
       'var_84_var_140_var_180', 'var_27_var_139_var_146',
       'var_99var_111', 'var_60_var_148_var_198',
       'var_26_var_162_var_179', 'var_99var_2', 'var_27_var_107_var_148',
       'var_0var_6', 'var_170var_61', 'var_192var_93',
       'var_23_var_56_var_139', 'var_25var_3', 'var_111var_196',
       'var_87var_169', 'var_137var_151', 'var_27_var_133_var_144',
       'var_139', 'var_97_var_99_var_184', 'var_86_var_139_var_165',
       'var_26var_62', 'var_164var_8', 'var_28_var_33_var_121',
       'var_112var_199', 'var_86_var_138_var_167', 'var_188var_146',
       'var_44var_121', 'var_170var_155', 'var_28_var_55_var_164',
       'var_31_var_123_var_149', 'var_127var_44', 'var_37_var_52_var_90',
       'var_26var_162', 'var_77', 'var_108_var_121_var_139', 'var_0',
       'var_147var_106', 'var_12_var_13_var_44', 'var_194var_86',
       'var_7_var_92_var_174', 'var_48_var_56_var_139',
       'var_15_var_35_var_145', 'var_113_var_123_var_131', 'var_187var_4',
       'var_34var_12', 'var_147var_147', 'var_181',
       'var_0_var_110_var_190', 'var_3_var_89_var_199', 'var_35var_184'])

def generate_fetures(train_set, test_set, scaler, feature_names):
    """Generate new features according to the original features and <feature_names>.
    
    <feature_names> is a list of new feature names. There are 3 different ways to
    generate new features according to the feature name:
        1. exactly the copy of the original feature, if var_X
        2. mutiplication of two original features, if var_Xvar_Y
        3. mutiplication of two original features, if var_X_var_Y_var_Z
    """
    set1 = set()
    set2 = set()
    set3 = set()
    for feature_name in feature_names:
        if(len(feature_name) >= 15):
            set3.add(feature_name)
        elif(len(feature_name) >= 10):
            set2.add(feature_name)
        else:
            set1.add(feature_name)
            
    new_train_set = None
    if(type(train_set) != type(None)):
        train_set = scaler.fit_transform(train_set)
        train_set = pd.DataFrame(train_set, columns=original_features)
        new_train_set = pd.DataFrame()
        for feature_name in set1:
            new_train_set[feature_name] = train_set[feature_name]
        for feature_name in set2:
            cols = feature_name.split('v')
            col1 = 'v' + cols[1]
            col2 = 'v' + cols[2]
            new_train_set[feature_name] = train_set[col1] * train_set[col2]
        for feature_name in set3:
            cols = feature_name.split('_')
            col1 = 'var_' + cols[1]
            col2 = 'var_' + cols[3]
            col3 = 'var_' + cols[5]
            new_train_set[feature_name] = train_set[col1] * train_set[col2] * train_set[col3]
    
    new_test_set = None
    if(type(test_set) != type(None)):
        test_set = scaler.transform(test_set)
        test_set = pd.DataFrame(test_set, columns=original_features)
        new_test_set = pd.DataFrame()
        for feature_name in set1:
            new_test_set[feature_name] = test_set[feature_name]
        for feature_name in set2:
            cols = feature_name.split('v')
            col1 = 'v' + cols[1]
            col2 = 'v' + cols[2]
            new_test_set[feature_name] = test_set[col1] * test_set[col2]
        for feature_name in set3:
            cols = feature_name.split('_')
            col1 = 'var_' + cols[1]
            col2 = 'var_' + cols[3]
            col3 = 'var_' + cols[5]
            new_test_set[feature_name] = test_set[col1] * test_set[col2] * test_set[col3]
    
    print('new features are successfully generated!~')
    return new_train_set, new_test_set

def remove_outliers(dataset, index, outlier_ratio_0, outlier_ratio_1):
    """Remove outliers by using EllipticEnvelope.
    
    <dataset>: original training set
    <index>: a list of row index of <dataset>
    <outlier_ratio_0>: the ratio of smaples with label 0 regarded as outlers
    <outlier_ratio_1>: the ratio of smaples with label 1 regarded as outlers
    """
    index_set = set(index)
    selection_list = list()
    for i in range(dataset.target.count()):
        selection_list.append(i in index_set)
    train_set_0 = (dataset[pd.Series(selection_list) & (dataset.target==0)])
    train_set_1 = (dataset[pd.Series(selection_list) & (dataset.target==1)])
    
    elliptic0 = EllipticEnvelope(contamination=outlier_ratio_0)
    elliptic1 = EllipticEnvelope(contamination=outlier_ratio_1)
    
    outlier_label_0 = elliptic0.fit_predict(train_set_0[original_features])
    print('outlier0 labels generated!')
    outlier_label_1 = elliptic1.fit_predict(train_set_1[original_features])
    print('outlier1 labels generated!')
    train_set_0 = train_set_0[list(pd.Series(outlier_label_0) == 1)]
    train_set_1 = train_set_1[list(pd.Series(outlier_label_1) == 1)]
    
    return pd.concat([train_set_0, train_set_1]).sample(frac=1.0)

# settings for LightBGM model
# These settings are found after many tries.
param = {
    'bagging_freq': 100,
    'bagging_fraction': 0.6,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.002,
    'learning_rate': 0.01,
    'max_depth': -1,
    'metric':'auc',
    'min_data_in_leaf': 100,
    'min_sum_hessian_in_leaf': 0.001,
    'num_leaves': 3,
    'num_threads': 4,
    'objective': 'binary',
    'verbosity': 1
}

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
feature_importance_df = pd.DataFrame()

# The model is trained based on cross validation
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    gc.collect()
    print("Fold {}".format(fold_))
    
    # prepare training data and validation data
    scaler = MinMaxScaler((1, 2))
    trainSet = remove_outliers(train_df, trn_idx, 0.04, 0.04)
    trainSetTarget = trainSet.target
    gc.collect()
    valSet = train_df.iloc[val_idx][original_features]
    trainSet, valSet = generate_fetures(trainSet[original_features], valSet, scaler, top_1k_important_features)
    new_features = [col for col in trainSet.columns if col not in ['ID_code', 'target']]
    gc.collect()
    trn_data = lgb.Dataset(trainSet, label=trainSetTarget)
    val_data = lgb.Dataset(valSet, label=target.iloc[val_idx])

    # train
    num_round = 40000
    clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=3000, early_stopping_rounds=1000)
    oof[val_idx] = clf.predict(valSet, num_iteration=clf.best_iteration)
    
    # gather the importance information of features
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = new_features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    # prediction
    trainSet, testSet = generate_fetures(None, test_df[original_features], scaler, top_1k_important_features)
    gc.collect()
    predictions += clf.predict(testSet, num_iteration=clf.best_iteration) / folds.n_splits

print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

# generation CSV file for submission
sub_df = pd.DataFrame({"ID_code":test_df["ID_code"].values})
sub_df["target"] = predictions
sub_df.to_csv("submission.csv", index=False)