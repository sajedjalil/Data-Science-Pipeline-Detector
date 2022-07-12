import kagglegym
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel)

# Observed with histograns:
low_y_cut = -0.086093
high_y_cut = 0.093497

# mean values from full dataset
full_df_mean = pd.Series({u'derived_0': -4.5360456, u'derived_1': 7.7294358e+11, u'derived_2': -0.33203283, u'derived_3': -0.50460118, u'derived_4': 18.016613, u'fundamental_0': -0.020409383, u'fundamental_1': -5.7037536e+08, u'fundamental_10': 0.44184005, u'fundamental_11': -0.25349072, u'fundamental_12': 61.683414, u'fundamental_13': 0.13335535, u'fundamental_14': 3.135854, u'fundamental_15': 0.21187612, u'fundamental_16': -0.21154669, u'fundamental_17': 7.0874443e+13, u'fundamental_18': -0.82331616, u'fundamental_19': 0.39892805, u'fundamental_2': -0.16229536, u'fundamental_20': -1.4807522, u'fundamental_21': 0.09907262, u'fundamental_22': 0.061406076, u'fundamental_23': 27.403885, u'fundamental_24': 0.076107837, u'fundamental_25': 0.16939202, u'fundamental_26': 216.98228, u'fundamental_27': 4.2248464, u'fundamental_28': 0.042846259, u'fundamental_29': 0.55585134, u'fundamental_3': 0.027802395, u'fundamental_30': 0.066347629, u'fundamental_31': 0.099537916, u'fundamental_32': 33.156422, u'fundamental_33': 616.56512, u'fundamental_34': 1.9188328, u'fundamental_35': 0.34897235, u'fundamental_36': 26.719276, u'fundamental_37': 0.14836955, u'fundamental_38': 0.049366269, u'fundamental_39': 0.27124742, u'fundamental_40': -0.012113965, u'fundamental_41': -469.67477, u'fundamental_42': 132.21361, u'fundamental_43': 0.23444968, u'fundamental_44': 1.7209773, u'fundamental_45': 1.4680398, u'fundamental_46': 0.18678555, u'fundamental_47': 0.065261841, u'fundamental_48': -0.063058481, u'fundamental_49': 0.34341455, u'fundamental_5': 0.77524048, u'fundamental_50': 12.578838, u'fundamental_51': 0.32494286, u'fundamental_52': 0.34409711, u'fundamental_53': -0.16662288, u'fundamental_54': 0.1882488, u'fundamental_55': -0.084851533, u'fundamental_56': 0.046315186, u'fundamental_57': 0.22214222, u'fundamental_58': 0.021671223, u'fundamental_59': 0.14801531, u'fundamental_6': 0.14047608, u'fundamental_60': 0.19425973, u'fundamental_61': 9.3409444e+11, u'fundamental_62': -0.052937128, u'fundamental_63': 0.15223059, u'fundamental_7': 48.656155, u'fundamental_8': 0.08000651, u'fundamental_9': -0.086174473, u'id': 1093.369, u'technical_0': -0.10920228, u'technical_1': 0.00047539145, u'technical_10': -0.78359717, u'technical_11': -0.92170662, u'technical_12': -0.17973705, u'technical_13': 0.00029250374, u'technical_14': -0.99065822, u'technical_16': -0.0013169635, u'technical_17': -0.89924318, u'technical_18': -0.031902265, u'technical_19': -0.05358487, u'technical_2': -0.90304118, u'technical_20': 0.0014537306, u'technical_21': -0.012947327, u'technical_22': -0.0048583783, u'technical_24': 0.0025382061, u'technical_25': 0.00062033773, u'technical_27': -0.075863883, u'technical_28': 0.00067628775, u'technical_29': -0.909805, u'technical_3': 0.0012166852, u'technical_30': 0.0012459407, u'technical_31': 0.00025132595, u'technical_32': -0.086690985, u'technical_33': 0.017549522, u'technical_34': -0.00058132195, u'technical_35': -0.10329999, u'technical_36': -0.085848331, u'technical_37': -0.091033973, u'technical_38': -0.081566848, u'technical_39': -0.072870009, u'technical_40': 0.049083214, u'technical_41': 0.0052362178, u'technical_42': -0.016999658, u'technical_43': -0.97352988, u'technical_44': 0.00038814748, u'technical_5': 0.00079619139, u'technical_6': -0.95168394, u'technical_7': 0.050597329, u'technical_9': -0.023041174, u'timestamp': 945.58252, u'y': 0.00022175087})
full_df_y_std = 0.022421712
full_df_y_mean = 0.00022174742
full_df_y_var = 0.00050273316

def build_model(train, features=[], target_col = 'y'):
    print("Fitting a GP model.")
    y_is_above_cut = (train.y > high_y_cut)
    y_is_below_cut = (train.y < low_y_cut)
    y_is_within_cut = (~y_is_above_cut & ~y_is_below_cut)
    train = train[y_is_within_cut].fillna(full_df_mean)
    
    train = train.sample(2000)
    print(train.index.tolist())
    
    gp_kernel_1 = 3.0e0*full_df_y_std * RBF(length_scale=1e-4, length_scale_bounds=(1e-4, 1e3))
    gp_kernel_2 = 1.0e3 * RBF(length_scale=1.0, length_scale_bounds=(3e0, 3e3))
    gp_kernel_3 = WhiteKernel(noise_level=4e0*full_df_y_var)
    
    gp_kernel = gp_kernel_1 + gp_kernel_2 + gp_kernel_3
    
    m = GaussianProcessRegressor(kernel=gp_kernel)
    
    m.fit(train[features].apply(expit).values.reshape(-1,  len(features)), train[target_col].values.reshape(-1, 1))
    print("Model's log marginal likelihood is :", m.log_marginal_likelihood())
    return m

# The "environment" is our interface for code competitions
env = kagglegym.make()

# We get our initial observation by calling "reset"
observation = env.reset()

# Note that the first observation we get has a "train" dataframe
print("Train has {} rows".format(len(observation.train)))

features = ['technical_30', 'technical_20', 'fundamental_11']
gp_regressor = build_model(observation.train, features)

# The "target" dataframe is a template for what we need to predict:
print("Target column names: {}".format(", ".join(['"{}"'.format(col) for col in list(observation.target.columns)])))

while True:
    target = observation.target
    timestamp = observation.features["timestamp"][0]
    features_values = observation.features.loc[:, features].fillna(full_df_mean).apply(expit).values.reshape((-1, len(features)))
    target['y'] = np.array(gp_regressor.predict(features_values), dtype=np.float64)
    
    if timestamp % 100 == 0:
        print("Timestamp #{}".format(timestamp))

    # We perform a "step" by making our prediction and getting back an updated "observation":
    observation, reward, done, info = env.step(target)
    #print("Reward :", reward)
    if done:
        print("Public score: {}".format(info["public_score"]))
        break