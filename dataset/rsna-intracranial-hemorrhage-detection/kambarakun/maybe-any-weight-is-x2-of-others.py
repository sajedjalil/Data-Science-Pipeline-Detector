import math

import pandas as pd
import sympy


# Define logloss, using LB digits(XX.XXX)
logloss      = {}
logloss[0.5] = -math.log(0.5)           * 1000 * 1000 // 1000 / 1000 #  0.693
logloss[0]   = -math.log(1 - 10**(-15)) * 1000 * 1000 // 1000 / 1000 #  0.000
logloss[1]   = -math.log(    10**(-15)) * 1000 * 1000 // 1000 / 1000 # 34.538

# Define Variables, weights are 0-1(relative value), positives are 0-78545(N of stage 1 test images)
weights_epidural           = sympy.Symbol('weights_epidural')
weights_intraparenchymal   = sympy.Symbol('weights_intraparenchymal')
weights_intraventricular   = sympy.Symbol('weights_intraventricular')
weights_subarachnoid       = sympy.Symbol('weights_subarachnoid')
weights_subdural           = sympy.Symbol('weights_subdural')
weights_any                = sympy.Symbol('weights_any')
positives_epidural         = sympy.Symbol('positives_epidural')
positives_intraparenchymal = sympy.Symbol('positives_intraparenchymal')
positives_intraventricular = sympy.Symbol('positives_intraventricular')
positives_subarachnoid     = sympy.Symbol('positives_subarachnoid')
positives_subdural         = sympy.Symbol('positives_subdural')
positives_any              = sympy.Symbol('positives_any')

# Load sample_submission.csv, set labels are 0.5 in case overwrited
df_submission          = pd.read_csv('../input/rsna-intracranial-hemorrhage-detection/stage_1_sample_submission.csv')
df_submission['Label'] = 0.5

# Create submission_file, df_any_{0, 1} means *_any's label is {0, 1} else labels are 0.5
'''
df_epidural_0               = df_submission.copy()
df_epidural_1               = df_submission.copy()
df_epidural_0.iloc[0::6, 1] = 0
df_epidural_1.iloc[0::6, 1] = 1
df_epidural_0.to_csv('submission_epidural_0.csv', index=False)
df_epidural_1.to_csv('submission_epidural_1.csv', index=False)
df_intraparenchymal_0               = df_submission.copy()
df_intraparenchymal_1               = df_submission.copy()
df_intraparenchymal_0.iloc[1::6, 1] = 0
df_intraparenchymal_1.iloc[1::6, 1] = 1
df_intraparenchymal_0.to_csv('submission_intraparenchymal_0.csv', index=False)
df_intraparenchymal_1.to_csv('submission_intraparenchymal_1.csv', index=False)
df_intraventricular_0               = df_submission.copy()
df_intraventricular_1               = df_submission.copy()
df_intraventricular_0.iloc[2::6, 1] = 0
df_intraventricular_1.iloc[2::6, 1] = 1
df_intraventricular_0.to_csv('submission_intraventricular_0.csv', index=False)
df_intraventricular_1.to_csv('submission_intraventricular_1.csv', index=False)
df_subarachnoid_0               = df_submission.copy()
df_subarachnoid_1               = df_submission.copy()
df_subarachnoid_0.iloc[3::6, 1] = 0
df_subarachnoid_1.iloc[3::6, 1] = 1
df_subarachnoid_0.to_csv('submission_subarachnoid_0.csv', index=False)
df_subarachnoid_1.to_csv('submission_subarachnoid_1.csv', index=False)
df_subdural_0               = df_submission.copy()
df_subdural_1               = df_submission.copy()
df_subdural_0.iloc[4::6, 1] = 0
df_subdural_1.iloc[4::6, 1] = 1
df_subdural_0.to_csv('submission_subdural_0.csv', index=False)
df_subdural_1.to_csv('submission_subdural_1.csv', index=False)
df_any_0               = df_submission.copy()
df_any_1               = df_submission.copy()
df_any_0.iloc[5::6, 1] = 0
df_any_1.iloc[5::6, 1] = 1
df_any_0.to_csv('submission_any_0.csv', index=False)
df_any_1.to_csv('submission_any_1.csv', index=False)
'''

# Input LB socre: submission_epidural_{0, 1}.csv, 'lb_score_* = - XX.XXX + ...' is LB SCORE
lb_score_epidural_0 = - 0.618 + ((1 - weights_epidural) * logloss[0.5]) + (weights_epidural * (sympy.Symbol('positives_epidural') * logloss[1] + (78545 - sympy.Symbol('positives_epidural')) * logloss[0]) / 78545)
lb_score_epidural_1 = - 5.504 + ((1 - weights_epidural) * logloss[0.5]) + (weights_epidural * (sympy.Symbol('positives_epidural') * logloss[0] + (78545 - sympy.Symbol('positives_epidural')) * logloss[1]) / 78545)

# Input LB socre: submission_intraparenchymal_{0, 1}.csv, 'lb_score_* = - XX.XXX + ...' is LB SCORE
lb_score_intraparenchymal_0 = - 0.817 + ((1 - weights_intraparenchymal) * logloss[0.5]) + (weights_intraparenchymal * (sympy.Symbol('positives_intraparenchymal') * logloss[1] + (78545 - sympy.Symbol('positives_intraparenchymal')) * logloss[0]) / 78545)
# lb_score_intraparenchymal_1 = - X.XXX + ((1 - weights_intraparenchymal) * logloss[0.5]) + (weights_intraparenchymal * (sympy.Symbol('positives_intraparenchymal') * logloss[0] + (78545 - sympy.Symbol('positives_intraparenchymal')) * logloss[1]) / 78545)

# Input LB socre: submission_intraventricular_{0, 1}.csv, 'lb_score_* = - XX.XXX + ...' is LB SCORE
# lb_score_intraventricular_0 = - X.XXX + ((1 - weights_intraventricular) * logloss[0.5]) + (weights_intraventricular * (sympy.Symbol('positives_intraventricular') * logloss[1] + (78545 - sympy.Symbol('positives_intraventricular')) * logloss[0]) / 78545)
# lb_score_intraventricular_1 = - X.XXX + ((1 - weights_intraventricular) * logloss[0.5]) + (weights_intraventricular * (sympy.Symbol('positives_intraventricular') * logloss[0] + (78545 - sympy.Symbol('positives_intraventricular')) * logloss[1]) / 78545)

# Input LB socre: submission_subarachnoid_{0, 1}.csv, 'lb_score_* = - XX.XXX + ...' is LB SCORE
# lb_score_subarachnoid_0 = - X.XXX + ((1 - weights_subarachnoid) * logloss[0.5]) + (weights_subarachnoid * (sympy.Symbol('positives_subarachnoid') * logloss[1] + (78545 - sympy.Symbol('positives_subarachnoid')) * logloss[0]) / 78545)
# lb_score_subarachnoid_1 = - X.XXX + ((1 - weights_subarachnoid) * logloss[0.5]) + (weights_subarachnoid * (sympy.Symbol('positives_subarachnoid') * logloss[0] + (78545 - sympy.Symbol('positives_subarachnoid')) * logloss[1]) / 78545)

# Input LB socre: submission_subdural_{0, 1}.csv, 'lb_score_* = - XX.XXX + ...' is LB SCORE
# lb_score_subdural_0 = - X.XXX + ((1 - weights_subdural) * logloss[0.5]) + (weights_subdural * (sympy.Symbol('positives_subdural') * logloss[1] + (78545 - sympy.Symbol('positives_subdural')) * logloss[0]) / 78545)
# lb_score_subdural_1 = - X.XXX + ((1 - weights_subdural) * logloss[0.5]) + (weights_subdural * (sympy.Symbol('positives_subdural') * logloss[0] + (78545 - sympy.Symbol('positives_subdural')) * logloss[1]) / 78545)

# Input LB socre: submission_any_{0, 1}.csv, 'lb_score_* = - XX.XXX' is LB SCORE
lb_score_any_0 = - 1.855 + ((1 - weights_any) * logloss[0.5]) + (weights_any * (sympy.Symbol('positives_any') * logloss[1] + (78545 - sympy.Symbol('positives_any')) * logloss[0]) / 78545)
lb_score_any_1 = - 9.002 + ((1 - weights_any) * logloss[0.5]) + (weights_any * (sympy.Symbol('positives_any') * logloss[0] + (78545 - sympy.Symbol('positives_any')) * logloss[1]) / 78545)

# Solve {weights, positives}_epidural
solution = []
solution.append(sympy.solve([lb_score_epidural_0,         lb_score_epidural_1        ])[0].values())
# solution.append(sympy.solve([lb_score_intraparenchymal_0, lb_score_intraparenchymal_1])[0].values())
# solution.append(sympy.solve([lb_score_intraventricular_0, lb_score_intraventricular_1])[0].values())
# solution.append(sympy.solve([lb_score_subarachnoid_0,     lb_score_subarachnoid_1    ])[0].values())
# solution.append(sympy.solve([lb_score_subdural_0,         lb_score_subdural_1        ])[0].values())
solution.append(sympy.solve([lb_score_any_0,              lb_score_any_1             ])[0].values())

# Output results, 0.1428 ≒ 1 / 7, 7 = 5 + 1 * 2, so I think *_any weight is x2 of others
# df_output = pd.DataFrame(solution, columns=['N_positives', '%weight'], index=['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural', 'any'])
df_output = pd.DataFrame(solution, columns=['N_positives', '%weight'], index=['epidural', 'any'])

print(df_output)
