from __future__ import print_function
from __future__ import division

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import math
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from scipy.stats import norm
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.cross_validation import cross_val_score, cross_val_predict
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import SVR, SVC
    import matplotlib.pyplot as plt
    import scipy.interpolate
    from itertools import product, chain
    from sklearn.metrics import make_scorer, mean_squared_error
# UNCOMMENT THE NEXT LINE IF YOU HAVE bayes_opt INSTALLED
#    from bayes_opt import BayesianOptimization

############################################
#
#  Most of the lines below (#48-613)
#  are copied from an excellent library
#  for global optimization with gaussian
#  processes (see below). The only reason
#  for that is because Kaggle does not
#  have this library installed.
#
#  You can delete the section below
#  (everything between lines 24-623)
#  if you have bayes_opt installed
#  on your system. Don't forget to
#  uncomment line #19 above
#
############################################

####################################################################################
#
#  The author of Bayesian Optimization is Fernando https://libraries.io/github/fmfn
#  For installing bayes_opt, see https://github.com/fmfn/BayesianOptimization
#
####################################################################################

    from sklearn.gaussian_process import GaussianProcess
    from scipy.optimize import minimize


def acq_max(ac, gp, y_max, bounds):
    """
    A function to find the maximum of the acquisition function using
    the 'L-BFGS-B' method.

    Parameters
    ----------
    :param ac:
        The acquisition function object that return its point-wise value.

    :param gp:
        A gaussian process fitted to the relevant data.

    :param y_max:
        The current maximum known value of the target function.

    :param bounds:
        The variables bounds to limit the search of the acq max.


    Returns
    -------
    :return: x_max, The arg max of the acquisition function.
    """

    # Start with the lower bound as the argmax
    x_max = bounds[:, 0]
    max_acq = None

    x_tries = np.random.uniform(bounds[:, 0], bounds[:, 1],
                                size=(100, bounds.shape[0]))

    for x_try in x_tries:
        # Find the minimum of minus the acquisition function
        res = minimize(lambda x: -ac(x.reshape(1, -1), gp=gp, y_max=y_max),
                       x_try.reshape(1, -1),
                       bounds=bounds,
                       method="L-BFGS-B")

        # Store it if better than previous minimum(maximum).
        if max_acq is None or -res.fun >= max_acq:
            x_max = res.x
            max_acq = -res.fun

    # Clip output to make sure it lies within the bounds. Due to floating
    # point technicalities this is not always the case.
    return np.clip(x_max, bounds[:, 0], bounds[:, 1])

def matern52(theta, d):
    """
    Matern 5/2 correlation model.::
    
        theta, d --> r(theta, d) = (1+sqrt(5)*r + 5/3*r^2)*exp(-sqrt(5)*r)
        
                               n
            where r = sqrt(   sum  (d_i)^2 / (theta_i)^2 )
                             i = 1
                             
    Parameters
    ----------
    theta : array_like
        An array with shape 1 (isotropic) or n (anisotropic) giving the 
        autocorrelation parameter(s).
        
    d : array_like
        An array with shape (n_eval, n_features) giving the componentwise
        distances between locations x and x' at which the correlation model
        should be evaluated.
        
    Returns
    -------
    r : array_like
        An array with shape (n_eval, ) containing the values of the
        autocorrelation modle.
    """

    theta = np.asarray(theta, dtype=np.float)
    d = np.asarray(d, dtype=np.float)
    
    if d.ndim > 1:
        n_features = d.shape[1]
    else:
        n_features = 1
        
    if theta.size == 1:
        r = np.sqrt(np.sum(d ** 2, axis=1)) / theta[0]
    elif theta.size != n_features:
        raise ValueError("Length of theta must be 1 or %s" % n_features)
    else:
        r = np.sqrt(np.sum(d ** 2 / theta.reshape(1,n_features) ** 2 , axis=1))
        
    return (1 + np.sqrt(5)*r + 5/3.*r ** 2) * np.exp(-np.sqrt(5)*r)
        

class BayesianOptimization(object):

    def __init__(self, f, pbounds, verbose=1):
        """
        :param f:
            Function to be maximized.

        :param pbounds:
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        :param verbose:
            Whether or not to print progress.

        """
        # Store the original dictionary
        self.pbounds = pbounds

        # Get the name of the parameters
        self.keys = list(pbounds.keys())

        # Find number of parameters
        self.dim = len(pbounds)

        # Create an array with parameters bounds
        self.bounds = []
        for key in self.pbounds.keys():
            self.bounds.append(self.pbounds[key])
        self.bounds = np.asarray(self.bounds)

        # Some function to be optimized
        self.f = f

        # Initialization flag
        self.initialized = False

        # Initialization lists --- stores starting points before process begins
        self.init_points = []
        self.x_init = []
        self.y_init = []

        # Numpy array place holders
        self.X = None
        self.Y = None

        # Counter of iterations
        self.i = 0

        # Since scipy 0.16 passing lower and upper bound to theta seems to be
        # broken. However, there is a lot of development going on around GP
        # is scikit-learn. So I'll pick the easy route here and simple specify
        # only theta0.
        self.gp = GaussianProcess(corr=matern52,
                                  theta0=np.random.uniform(0.001, 0.05, self.dim),
                                  thetaL=1e-5 * np.ones(self.dim),
                                  thetaU=1e0 * np.ones(self.dim),
                                  random_start=30)

        # Utility Function placeholder
        self.util = None

        # PrintLog object
        self.plog = PrintLog(self.keys)

        # Output dictionary
        self.res = {}
        # Output dictionary
        self.res['max'] = {'max_val': None,
                           'max_params': None}
        self.res['all'] = {'values': [], 'params': []}

        # Verbose
        self.verbose = verbose

    def init(self, init_points):
        """
        Initialization method to kick start the optimization process. It is a
        combination of points passed by the user, and randomly sampled ones.

        :param init_points:
            Number of random points to probe.
        """

        # Generate random points
        l = [np.random.uniform(x[0], x[1], size=init_points) for x in self.bounds]

        # Concatenate new random points to possible existing
        # points from self.explore method.
        self.init_points += list(map(list, zip(*l)))

        # Create empty list to store the new values of the function
        y_init = []

        # Evaluate target function at all initialization
        # points (random + explore)
        for x in self.init_points:

            y_init.append(self.f(**dict(zip(self.keys, x))))

            if self.verbose:
                self.plog.print_step(x, y_init[-1])

        # Append any other points passed by the self.initialize method (these
        # also have a corresponding target value passed by the user).
        self.init_points += self.x_init

        # Append the target value of self.initialize method.
        y_init += self.y_init

        # Turn it into np array and store.
        self.X = np.asarray(self.init_points)
        self.Y = np.asarray(y_init)

        # Updates the flag
        self.initialized = True

    def explore(self, points_dict):
        """
        Method to explore user defined points

        :param points_dict:
        :return:
        """

        # Consistency check
        param_tup_lens = []

        for key in self.keys:
            param_tup_lens.append(len(list(points_dict[key])))

        if all([e == param_tup_lens[0] for e in param_tup_lens]):
            pass
        else:
            raise ValueError('The same number of initialization points '
                             'must be entered for every parameter.')

        # Turn into list of lists
        all_points = []
        for key in self.keys:
            all_points.append(points_dict[key])

        # Take transpose of list
        self.init_points = list(map(list, zip(*all_points)))

    def initialize(self, points_dict):
        """
        Method to introduce point for which the target function
        value is known

        :param points_dict:
        :return:
        """

        for target in points_dict:

            self.y_init.append(target)

            all_points = []
            for key in self.keys:
                all_points.append(points_dict[target][key])

            self.x_init.append(all_points)

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        :param new_bounds:
            A dictionary with the parameter name and its new bounds

        """

        # Update the internal object stored dict
        self.pbounds.update(new_bounds)

        # Loop through the all bounds and reset the min-max bound matrix
        for row, key in enumerate(self.pbounds.keys()):

            # Reset all entries, even if the same.
            self.bounds[row] = self.pbounds[key]

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ei',
                 kappa=2.576,
                 xi=0.0,
                 **gp_params):
        """
        Main optimization method.

        Parameters
        ----------
        :param init_points:
            Number of randomly chosen points to sample the
            target function before fitting the gp.

        :param n_iter:
            Total number of times the process is to repeated. Note that
            currently this methods does not have stopping criteria (due to a
            number of reasons), therefore the total number of points to be
            sampled must be specified.

        :param acq:
            Acquisition function to be used, defaults to Expected Improvement.

        :param gp_params:
            Parameters to be passed to the Scikit-learn Gaussian Process object

        Returns
        -------
        :return: Nothing
        """
        # Reset timer
        self.plog.reset_timer()

        # Set acquisition function
        self.util = UtilityFunction(kind=acq, kappa=kappa, xi=xi)

        # Initialize x, y and find current y_max
        if not self.initialized:
            if self.verbose:
                self.plog.print_header()
            self.init(init_points)

        y_max = self.Y.max()

        # Set parameters if any was passed
        self.gp.set_params(**gp_params)

        # Find unique rows of X to avoid GP from breaking
        ur = unique_rows(self.X)
        self.gp.fit(self.X[ur], self.Y[ur])

        # Finding argmax of the acquisition function.
        x_max = acq_max(ac=self.util.utility,
                        gp=self.gp,
                        y_max=y_max,
                        bounds=self.bounds)

        # Print new header
        if self.verbose:
            self.plog.print_header(initialization=False)
        # Iterative process of searching for the maximum. At each round the
        # most recent x and y values probed are added to the X and Y arrays
        # used to train the Gaussian Process. Next the maximum known value
        # of the target function is found and passed to the acq_max function.
        # The arg_max of the acquisition function is found and this will be
        # the next probed value of the target function in the next round.
        for i in range(n_iter):
            # Test if x_max is repeated, if it is, draw another one at random
            # If it is repeated, print a warning
            pwarning = False
            if np.any((self.X - x_max).sum(axis=1) == 0):

                x_max = np.random.uniform(self.bounds[:, 0],
                                          self.bounds[:, 1],
                                          size=self.bounds.shape[0])

                pwarning = True

            # Append most recently generated values to X and Y arrays
            self.X = np.vstack((self.X, x_max.reshape((1, -1))))
            self.Y = np.append(self.Y, self.f(**dict(zip(self.keys, x_max))))

            # Updating the GP.
            ur = unique_rows(self.X)
            self.gp.fit(self.X[ur], self.Y[ur])

            # Update maximum value to search for next probe point.
            if self.Y[-1] > y_max:
                y_max = self.Y[-1]

            # Maximize acquisition function to find next probing point
            x_max = acq_max(ac=self.util.utility,
                            gp=self.gp,
                            y_max=y_max,
                            bounds=self.bounds)

            # Print stuff
            if self.verbose:
                self.plog.print_step(self.X[-1], self.Y[-1], warning=pwarning)

            # Keep track of total number of iterations
            self.i += 1

            self.res['max'] = {'max_val': self.Y.max(),
                               'max_params': dict(zip(self.keys,
                                                      self.X[self.Y.argmax()]))
                               }
            self.res['all']['values'].append(self.Y[-1])
            self.res['all']['params'].append(dict(zip(self.keys, self.X[-1])))

        # Print a final report if verbose active.
        if self.verbose:
            self.plog.print_summary()

class UtilityFunction(object):
    """
    An object to compute the acquisition functions.
    """

    def __init__(self, kind, kappa, xi):
        """
        If UCB is to be used, a constant kappa is needed.
        """
        self.kappa = kappa
        
        self.xi = xi

        if kind not in ['ucb', 'ei', 'poi']:
            err = "The utility function " \
                  "{} has not been implemented, " \
                  "please choose one of ucb, ei, or poi.".format(kind)
            raise NotImplementedError(err)
        else:
            self.kind = kind

    def utility(self, x, gp, y_max):
        if self.kind == 'ucb':
            return self._ucb(x, gp, self.kappa)
        if self.kind == 'ei':
            return self._ei(x, gp, y_max, self.xi)
        if self.kind == 'poi':
            return self._poi(x, gp, y_max, self.xi)

    @staticmethod
    def _ucb(x, gp, kappa):
        mean, var = gp.predict(x, eval_MSE=True)
        return mean + kappa * np.sqrt(var)

    @staticmethod
    def _ei(x, gp, y_max, xi):
        mean, var = gp.predict(x, eval_MSE=True)

        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)

        z = (mean - y_max - xi)/np.sqrt(var)
        return (mean - y_max - xi) * norm.cdf(z) + np.sqrt(var) * norm.pdf(z)

    @staticmethod
    def _poi(x, gp, y_max, xi):
        mean, var = gp.predict(x, eval_MSE=True)

        # Avoid points with zero variance
        var = np.maximum(var, 1e-9 + 0 * var)

        z = (mean - y_max - xi)/np.sqrt(var)
        return norm.cdf(z)


def unique_rows(a):
    """
    A functions to trim repeated rows that may appear when optimizing.
    This is necessary to avoid the sklearn GP object from breaking

    :param a: array to trim repeated rows from

    :return: mask of unique rows
    """

    # Sort array and kep track of where things should go back to
    order = np.lexsort(a.T)
    reorder = np.argsort(order)

    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)

    return ui[reorder]


class BColours(object):
    BLUE = '\033[94m'
    CYAN = '\033[36m'
    GREEN = '\033[32m'
    MAGENTA = '\033[35m'
    RED = '\033[31m'
    ENDC = '\033[0m'


class PrintLog(object):

    def __init__(self, params):

        self.ymax = None
        self.xmax = None
        self.params = params
        self.ite = 1

        self.start_time = datetime.now()
        self.last_round = datetime.now()

        # sizes of parameters name and all
        self.sizes = [max(len(ps), 7) for ps in params]

        # Sorted indexes to access parameters
        self.sorti = sorted(range(len(self.params)),
                            key=self.params.__getitem__)

    def reset_timer(self):
        self.start_time = datetime.now()
        self.last_round = datetime.now()

    def print_header(self, initialization=True):

        if initialization:
            print("{}Initialization{}".format(BColours.RED,
                                              BColours.ENDC))
        else:
            print("{}Bayesian Optimization{}".format(BColours.RED,
                                                     BColours.ENDC))

        print(BColours.BLUE + "-" * (29 + sum([s + 5 for s in self.sizes])) + BColours.ENDC)

        print("{0:>{1}}".format("Step", 5), end=" | ")
        print("{0:>{1}}".format("Time", 6), end=" | ")
        print("{0:>{1}}".format("Value", 10), end=" | ")

        for index in self.sorti:
            print("{0:>{1}}".format(self.params[index],
                                    self.sizes[index] + 2),
                  end=" | ")
        print('')

    def print_step(self, x, y, warning=False):

        print("{:>5d}".format(self.ite), end=" | ")

        m, s = divmod((datetime.now() - self.last_round).total_seconds(), 60)
        print("{:>02d}m{:>02d}s".format(int(m), int(s)), end=" | ")

        if self.ymax is None or self.ymax < y:
            self.ymax = y
            self.xmax = x
            print("{0}{2: >10.5f}{1}".format(BColours.MAGENTA,
                                             BColours.ENDC,
                                             y),
                  end=" | ")

            for index in self.sorti:
                print("{0}{2: >{3}.{4}f}{1}".format(BColours.GREEN, BColours.ENDC,
                                                    x[index],
                                                    self.sizes[index] + 2,
                                                    min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")
        else:
            print("{: >10.5f}".format(y), end=" | ")
            for index in self.sorti:
                print("{0: >{1}.{2}f}".format(x[index],
                                              self.sizes[index] + 2,
                                              min(self.sizes[index] - 3, 6 - 2)),
                      end=" | ")

        if warning:
            print("{}Warning: Test point chose at "
                  "random due to repeated sample.{}".format(BColours.RED,
                                                            BColours.ENDC))

        print()

        self.last_round = datetime.now()
        self.ite += 1

    def print_summary(self):
        pass

#######################################
#
#  You can delete the section above
#  (everything between lines #24-623)
#  if you have bayes_opt installed
#  on your system. Don't forget to
#  uncomment line #22
#
#######################################

#######################################
#
#  The actual code starts here
#
#######################################

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        tmin, tsec = divmod((datetime.now() - start_time).total_seconds(), 60)
        print(" Time taken: %i minutes and %s seconds." % (tmin, round(tsec,2)))

def XGbcv( max_depth, gamma, min_child_weight, max_delta_step, subsample, colsample_bytree):

    global RMSEbest
    global ITERbest

    paramt = {
              'booster' : 'gbtree',
              'max_depth' : max_depth.astype(int),
              'gamma' : gamma,
              'eta' : 0.01,
              'objective': 'reg:linear',
              'nthread' : 8,
              'silent' : True,
              'eval_metric': 'rmse',
              'subsample' : subsample,
              'colsample_bytree' : colsample_bytree,
              'min_child_weight' : min_child_weight,
              'max_delta_step' : max_delta_step.astype(int),
              'seed' : 1001
              }

    folds = 5

    xgbr = xgb.cv(
           paramt,
           dtrain,
           num_boost_round = 100000,
#           stratified = True,
           nfold = folds,
           verbose_eval = False,
           early_stopping_rounds = 50,
           metrics = "rmse",
           show_stdv = True
          )

    cv_score = xgbr['test-rmse-mean'].iloc[-1]
    if ( cv_score < RMSEbest ):
        RMSEbest = cv_score
        ITERbest = len(xgbr)

    return (-1.0 * cv_score)

if __name__ == "__main__":

    folds = 5
    RMSEbest = 10.
    ITERbest = 0

    print("\n Please read the comments carefully. The code can be much shorter if you follow directions.")

    train = pd.read_csv("../input/train.csv")
    test = pd.read_csv("../input/test.csv")
    ids = test['Id']

##############################################################################################
#
#  The code below is for manipulating train and test datasets. It was taken from:
#  https://www.kaggle.com/klyusba/house-prices-advanced-regression-techniques/lasso-model-for-regression-problem/notebook
#  You can delete lines #701-1109 and replace it with your own data manipulation code
#
##############################################################################################

    all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                          test.loc[:,'MSSubClass':'SaleCondition']), ignore_index=True)
    # I have no idea how to do it better. Probably, it is better to do nothing
    x = all_data.loc[np.logical_not(all_data["LotFrontage"].isnull()), "LotArea"]
    y = all_data.loc[np.logical_not(all_data["LotFrontage"].isnull()), "LotFrontage"]
    # plt.scatter(x, y)
    t = (x <= 25000) & (y <= 150)
    p = np.polyfit(x[t], y[t], 1)
    all_data.loc[all_data['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, all_data.loc[all_data['LotFrontage'].isnull(), 'LotArea'])
    
    all_data.loc[all_data.Alley.isnull(), 'Alley'] = 'NoAlley'
    all_data.loc[all_data.MasVnrType.isnull(), 'MasVnrType'] = 'None' # no good
    all_data.loc[all_data.MasVnrType == 'None', 'MasVnrArea'] = 0
    all_data.loc[all_data.BsmtQual.isnull(), 'BsmtQual'] = 'NoBsmt'
    all_data.loc[all_data.BsmtCond.isnull(), 'BsmtCond'] = 'NoBsmt'
    all_data.loc[all_data.BsmtExposure.isnull(), 'BsmtExposure'] = 'NoBsmt'
    all_data.loc[all_data.BsmtFinType1.isnull(), 'BsmtFinType1'] = 'NoBsmt'
    all_data.loc[all_data.BsmtFinType2.isnull(), 'BsmtFinType2'] = 'NoBsmt'
    all_data.loc[all_data.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0
    all_data.loc[all_data.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0
    all_data.loc[all_data.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = all_data.BsmtFinSF1.median()
    all_data.loc[all_data.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0
    all_data.loc[all_data.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = all_data.BsmtUnfSF.median()
    all_data.loc[all_data.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0
    all_data.loc[all_data.FireplaceQu.isnull(), 'FireplaceQu'] = 'NoFireplace'
    all_data.loc[all_data.GarageType.isnull(), 'GarageType'] = 'NoGarage'
    all_data.loc[all_data.GarageFinish.isnull(), 'GarageFinish'] = 'NoGarage'
    all_data.loc[all_data.GarageQual.isnull(), 'GarageQual'] = 'NoGarage'
    all_data.loc[all_data.GarageCond.isnull(), 'GarageCond'] = 'NoGarage'
    all_data.loc[all_data.BsmtFullBath.isnull(), 'BsmtFullBath'] = 0
    all_data.loc[all_data.BsmtHalfBath.isnull(), 'BsmtHalfBath'] = 0
    all_data.loc[all_data.KitchenQual.isnull(), 'KitchenQual'] = 'TA'
    all_data.loc[all_data.MSZoning.isnull(), 'MSZoning'] = 'RL'
    all_data.loc[all_data.Utilities.isnull(), 'Utilities'] = 'AllPub'
    all_data.loc[all_data.Exterior1st.isnull(), 'Exterior1st'] = 'VinylSd'
    all_data.loc[all_data.Exterior2nd.isnull(), 'Exterior2nd'] = 'VinylSd'
    all_data.loc[all_data.Functional.isnull(), 'Functional'] = 'Typ'
    all_data.loc[all_data.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'
    all_data.loc[all_data.SaleCondition.isnull(), 'SaleType'] = 'WD'
    all_data.loc[all_data['PoolQC'].isnull(), 'PoolQC'] = 'NoPool'
    all_data.loc[all_data['Fence'].isnull(), 'Fence'] = 'NoFence'
    all_data.loc[all_data['MiscFeature'].isnull(), 'MiscFeature'] = 'None'
    all_data.loc[all_data['Electrical'].isnull(), 'Electrical'] = 'SBrkr'
    # only one is null and it has type Detchd
    all_data.loc[all_data['GarageArea'].isnull(), 'GarageArea'] = all_data.loc[all_data['GarageType']=='Detchd', 'GarageArea'].mean()
    all_data.loc[all_data['GarageCars'].isnull(), 'GarageCars'] = all_data.loc[all_data['GarageType']=='Detchd', 'GarageCars'].median()
    
    # where we have order we will use numeric
    all_data = all_data.replace({'Utilities': {'AllPub': 1, 'NoSeWa': 0, 'NoSewr': 0, 'ELO': 0},
                                 'Street': {'Pave': 1, 'Grvl': 0 },
                                 'FireplaceQu': {'Ex': 5, 
                                                'Gd': 4, 
                                                'TA': 3, 
                                                'Fa': 2,
                                                'Po': 1,
                                                'NoFireplace': 0 
                                                },
                                 'Fence': {'GdPrv': 2, 
                                           'GdWo': 2, 
                                           'MnPrv': 1, 
                                           'MnWw': 1,
                                           'NoFence': 0},
                                 'ExterQual': {'Ex': 5, 
                                                'Gd': 4, 
                                                'TA': 3, 
                                                'Fa': 2,
                                                'Po': 1
                                                },
                                 'ExterCond': {'Ex': 5, 
                                                'Gd': 4, 
                                                'TA': 3, 
                                                'Fa': 2,
                                                'Po': 1
                                                },
                                 'BsmtQual': {'Ex': 5, 
                                                'Gd': 4, 
                                                'TA': 3, 
                                                'Fa': 2,
                                                'Po': 1,
                                                'NoBsmt': 0},
                                 'BsmtExposure': {'Gd': 3, 
                                                'Av': 2, 
                                                'Mn': 1,
                                                'No': 0,
                                                'NoBsmt': 0},
                                 'BsmtCond': {'Ex': 5, 
                                                'Gd': 4, 
                                                'TA': 3, 
                                                'Fa': 2,
                                                'Po': 1,
                                                'NoBsmt': 0},
                                 'GarageQual': {'Ex': 5, 
                                                'Gd': 4, 
                                                'TA': 3, 
                                                'Fa': 2,
                                                'Po': 1,
                                                'NoGarage': 0},
                                 'GarageCond': {'Ex': 5, 
                                                'Gd': 4, 
                                                'TA': 3, 
                                                'Fa': 2,
                                                'Po': 1,
                                                'NoGarage': 0},
                                 'KitchenQual': {'Ex': 5, 
                                                'Gd': 4, 
                                                'TA': 3, 
                                                'Fa': 2,
                                                'Po': 1},
                                 'Functional': {'Typ': 0,
                                                'Min1': 1,
                                                'Min2': 1,
                                                'Mod': 2,
                                                'Maj1': 3,
                                                'Maj2': 4,
                                                'Sev': 5,
                                                'Sal': 6}                             
                                })
    newer_dwelling = all_data.MSSubClass.replace({20: 1, 
                                                30: 0, 
                                                40: 0, 
                                                45: 0,
                                                50: 0, 
                                                60: 1,
                                                70: 0,
                                                75: 0,
                                                80: 0,
                                                85: 0,
                                                90: 0,
                                               120: 1,
                                               150: 0,
                                               160: 0,
                                               180: 0,
                                               190: 0})
    newer_dwelling.name = 'newer_dwelling'
    
    all_data = all_data.replace({'MSSubClass': {20: 'SubClass_20', 
                                                30: 'SubClass_30', 
                                                40: 'SubClass_40', 
                                                45: 'SubClass_45',
                                                50: 'SubClass_50', 
                                                60: 'SubClass_60',
                                                70: 'SubClass_70',
                                                75: 'SubClass_75',
                                                80: 'SubClass_80',
                                                85: 'SubClass_85',
                                                90: 'SubClass_90',
                                               120: 'SubClass_120',
                                               150: 'SubClass_150',
                                               160: 'SubClass_160',
                                               180: 'SubClass_180',
                                               190: 'SubClass_190'}})
    
    # The idea is good quality should rise price, poor quality - reduce price
    overall_poor_qu = all_data.OverallQual.copy()
    overall_poor_qu = 5 - overall_poor_qu
    overall_poor_qu[overall_poor_qu<0] = 0
    overall_poor_qu.name = 'overall_poor_qu'
    
    overall_good_qu = all_data.OverallQual.copy()
    overall_good_qu = overall_good_qu - 5
    overall_good_qu[overall_good_qu<0] = 0
    overall_good_qu.name = 'overall_good_qu'
    
    overall_poor_cond = all_data.OverallCond.copy()
    overall_poor_cond = 5 - overall_poor_cond
    overall_poor_cond[overall_poor_cond<0] = 0
    overall_poor_cond.name = 'overall_poor_cond'
    
    overall_good_cond = all_data.OverallCond.copy()
    overall_good_cond = overall_good_cond - 5
    overall_good_cond[overall_good_cond<0] = 0
    overall_good_cond.name = 'overall_good_cond'
    
    exter_poor_qu = all_data.ExterQual.copy()
    exter_poor_qu[exter_poor_qu<3] = 1
    exter_poor_qu[exter_poor_qu>=3] = 0
    exter_poor_qu.name = 'exter_poor_qu'
    
    exter_good_qu = all_data.ExterQual.copy()
    exter_good_qu[exter_good_qu<=3] = 0
    exter_good_qu[exter_good_qu>3] = 1
    exter_good_qu.name = 'exter_good_qu'
    
    exter_poor_cond = all_data.ExterCond.copy()
    exter_poor_cond[exter_poor_cond<3] = 1
    exter_poor_cond[exter_poor_cond>=3] = 0
    exter_poor_cond.name = 'exter_poor_cond'
    
    exter_good_cond = all_data.ExterCond.copy()
    exter_good_cond[exter_good_cond<=3] = 0
    exter_good_cond[exter_good_cond>3] = 1
    exter_good_cond.name = 'exter_good_cond'
    
    bsmt_poor_cond = all_data.BsmtCond.copy()
    bsmt_poor_cond[bsmt_poor_cond<3] = 1
    bsmt_poor_cond[bsmt_poor_cond>=3] = 0
    bsmt_poor_cond.name = 'bsmt_poor_cond'
    
    bsmt_good_cond = all_data.BsmtCond.copy()
    bsmt_good_cond[bsmt_good_cond<=3] = 0
    bsmt_good_cond[bsmt_good_cond>3] = 1
    bsmt_good_cond.name = 'bsmt_good_cond'
    
    garage_poor_qu = all_data.GarageQual.copy()
    garage_poor_qu[garage_poor_qu<3] = 1
    garage_poor_qu[garage_poor_qu>=3] = 0
    garage_poor_qu.name = 'garage_poor_qu'
    
    garage_good_qu = all_data.GarageQual.copy()
    garage_good_qu[garage_good_qu<=3] = 0
    garage_good_qu[garage_good_qu>3] = 1
    garage_good_qu.name = 'garage_good_qu'
    
    garage_poor_cond = all_data.GarageCond.copy()
    garage_poor_cond[garage_poor_cond<3] = 1
    garage_poor_cond[garage_poor_cond>=3] = 0
    garage_poor_cond.name = 'garage_poor_cond'
    
    garage_good_cond = all_data.GarageCond.copy()
    garage_good_cond[garage_good_cond<=3] = 0
    garage_good_cond[garage_good_cond>3] = 1
    garage_good_cond.name = 'garage_good_cond'
    
    kitchen_poor_qu = all_data.KitchenQual.copy()
    kitchen_poor_qu[kitchen_poor_qu<3] = 1
    kitchen_poor_qu[kitchen_poor_qu>=3] = 0
    kitchen_poor_qu.name = 'kitchen_poor_qu'
    
    kitchen_good_qu = all_data.KitchenQual.copy()
    kitchen_good_qu[kitchen_good_qu<=3] = 0
    kitchen_good_qu[kitchen_good_qu>3] = 1
    kitchen_good_qu.name = 'kitchen_good_qu'
    
    qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,
                         exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu,
                         garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)
    
    bad_heating = all_data.HeatingQC.replace({'Ex': 0, 
                                              'Gd': 0, 
                                              'TA': 0, 
                                              'Fa': 1,
                                              'Po': 1})
    bad_heating.name = 'bad_heating'
                                              
    MasVnrType_Any = all_data.MasVnrType.replace({'BrkCmn': 1,
                                                  'BrkFace': 1,
                                                  'CBlock': 1,
                                                  'Stone': 1,
                                                  'None': 0})
    MasVnrType_Any.name = 'MasVnrType_Any'
    
    SaleCondition_PriceDown = all_data.SaleCondition.replace({'Abnorml': 1,
                                                              'Alloca': 1,
                                                              'AdjLand': 1,
                                                              'Family': 1,
                                                              'Normal': 0,
                                                              'Partial': 0})
    SaleCondition_PriceDown.name = 'SaleCondition_PriceDown'
    
    Neighborhood_Good = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['Neighborhood_Good'])
    Neighborhood_Good[all_data.Neighborhood=='NridgHt'] = 1
    Neighborhood_Good[all_data.Neighborhood=='Crawfor'] = 1
    Neighborhood_Good[all_data.Neighborhood=='StoneBr'] = 1
    Neighborhood_Good[all_data.Neighborhood=='Somerst'] = 1
    Neighborhood_Good[all_data.Neighborhood=='NoRidge'] = 1
    
    # do smth with BsmtFinType1, BsmtFinType2
    
    svm = SVC(C=100)
    # price categories
    pc = pd.Series(np.zeros(train.shape[0]))
    pc[:] = 'pc1'
    pc[train.SalePrice >= 150000] = 'pc2'
    pc[train.SalePrice >= 220000] = 'pc3'
    columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']
    X_t = pd.get_dummies(train.loc[:, columns_for_pc], sparse=True)
    svm.fit(X_t, pc)
    pc_pred = svm.predict(X_t)
    p = train.SalePrice/100000
    
    price_category = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['pc'])
    X_t = pd.get_dummies(all_data.loc[:, columns_for_pc], sparse=True)
    pc_pred = svm.predict(X_t)
    price_category[pc_pred=='pc2'] = 1
    price_category[pc_pred=='pc3'] = 2
    price_category = price_category.to_sparse()
    # Monthes with the lagest number of deals may be significant
    season = all_data.MoSold.replace( {1: 0, 
                                       2: 0, 
                                       3: 0, 
                                       4: 1,
                                       5: 1, 
                                       6: 1,
                                       7: 1,
                                       8: 0,
                                       9: 0,
                                      10: 0,
                                      11: 0,
                                      12: 0})
    season.name = 'season'
    
    # Numer month is not significant
    all_data = all_data.replace({'MoSold': {1: 'Yan', 
                                            2: 'Feb', 
                                            3: 'Mar', 
                                            4: 'Apr',
                                            5: 'May', 
                                            6: 'Jun',
                                            7: 'Jul',
                                            8: 'Avg',
                                            9: 'Sep',
                                            10: 'Oct',
                                            11: 'Nov',
                                            12: 'Dec'}})
    all_data = all_data.replace({'CentralAir': {'Y': 1, 
                                                'N': 0}})
    all_data = all_data.replace({'PavedDrive': {'Y': 1, 
                                                'P': 0,
                                                'N': 0}})
    reconstruct = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['Reconstruct'])
    reconstruct[all_data.YrSold < all_data.YearRemodAdd] = 1
    reconstruct = reconstruct.to_sparse()
    
    recon_after_buy = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['ReconstructAfterBuy'])
    recon_after_buy[all_data.YearRemodAdd >= all_data.YrSold] = 1
    recon_after_buy = recon_after_buy.to_sparse()
    
    build_eq_buy = pd.DataFrame(np.zeros((all_data.shape[0],1)), columns=['Build.eq.Buy'])
    build_eq_buy[all_data.YearBuilt >= all_data.YrSold] = 1
    build_eq_buy = build_eq_buy.to_sparse()
    # I hope this will help
    all_data.YrSold = 2010 - all_data.YrSold
    year_map = pd.concat(pd.Series('YearGroup' + str(i+1), index=range(1871+i*20,1891+i*20)) for i in range(0, 7))
    all_data.GarageYrBlt = all_data.GarageYrBlt.map(year_map)
    all_data.loc[all_data['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 'NoGarage'
    all_data.YearBuilt = all_data.YearBuilt.map(year_map)
    all_data.YearRemodAdd = all_data.YearRemodAdd.map(year_map)
    
    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    
    t = all_data[numeric_feats].quantile(.95)
    use_max_scater = t[t == 0].index
    use_95_scater = t[t != 0].index
    all_data[use_max_scater] = all_data[use_max_scater]/all_data[use_max_scater].max()
    all_data[use_95_scater] = all_data[use_95_scater]/all_data[use_95_scater].quantile(.95)
    t = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
         '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 
         'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']
    
    all_data.loc[:, t] = np.log1p(all_data.loc[:, t])
    # all classes in sklearn requires numeric data only
    # transform categorical variable into binary
    X = pd.get_dummies(all_data, sparse=True)
    X = X.fillna(0)
    X = X.drop('RoofMatl_ClyTile', axis=1) # only one is not zero
    X = X.drop('Condition2_PosN', axis=1) # only two is not zero
    X = X.drop('MSZoning_C (all)', axis=1)
    X = X.drop('MSSubClass_SubClass_160', axis=1)
    # this features definitely couse overfitting
    # add new features
    X = pd.concat((X, newer_dwelling, season, reconstruct, recon_after_buy,
                   qu_list, bad_heating, MasVnrType_Any, price_category, build_eq_buy), axis=1)
    
    def poly(X):
        areas = ['LotArea', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'BsmtUnfSF']
        # t = [s for s in X.axes[1].get_values() if s not in areas]
        t = chain(qu_list.axes[1].get_values(), 
                  ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtCond', 'GarageQual', 'GarageCond',
                   'KitchenQual', 'HeatingQC', 'bad_heating', 'MasVnrType_Any', 'SaleCondition_PriceDown', 'Reconstruct',
                   'ReconstructAfterBuy', 'Build.eq.Buy'])
        for a, t in product(areas, t):
            x = X.loc[:, [a, t]].prod(1)
            x.name = a + '_' + t
            yield x
    
    XP = pd.concat(poly(X), axis=1)
    X = pd.concat((X, XP), axis=1)
    X_train = X[:train.shape[0]]
    X_test = X[train.shape[0]:]
    # the model has become really big
#    print(X_train.shape)
    y = np.log1p(train.SalePrice)
    # this come from iterational model improvment. I was trying to understand why the model gives to the two points much better price
    x_plot = X_train.loc[X_train['SaleCondition_Partial']==1, 'GrLivArea']
    y_plot = y[X_train['SaleCondition_Partial']==1]
    outliers_id = np.array([524, 1299])
    
    outliers_id = outliers_id - 1 # id starts with 1, index starts with 0
    X_train = X_train.drop(outliers_id)
    y = y.drop(outliers_id)
    # There are difinetly more outliers
    
    def rmsle(y, y_pred):
         return np.sqrt((( (np.log1p(y_pred*price_scale)- np.log1p(y*price_scale)) )**2).mean())
    
    # scorer = make_scorer(rmsle, False)
    scorer = make_scorer(mean_squared_error, False)
    
    def rmse_cv(model, X, y):
         return (cross_val_score(model, X, y, scoring=scorer)).mean()
    
##############################################################################################
#
#  Data manipulation code ends here. You are welcome to replace the section above
#  with your own code. Just make sure that train data is in X_train, test data
#  in X_test, and that log1p-modified Sale Prices are in y
#
##############################################################################################

    dtrain = xgb.DMatrix(X_train, label=y)
    dtest = xgb.DMatrix(X_test)

    print("\n Train Set Matrix Dimensions: %d x %d" % (X_train.shape[0], X_train.shape[1]))
    print("\n Test Set Matrix Dimensions: %d x %d\n" % (X_test.shape[0], X_test.shape[1]))

    start_time = timer(None)
    print("# Global Optimization Search for XGboost Parameters")
    print("\n Please note that negative RMSE values will be shown below. This is because")
    print(" RMSE needs to be minimized, while Bayes Optimizer always maximizes the function.\n")

    XGbBO = BayesianOptimization(XGbcv, {'max_depth': (3, 10),
                                     'gamma': (0.00001, 1.0),
                                     'min_child_weight': (0, 5),
                                     'max_delta_step': (0, 5),
                                     'subsample': (0.5, 0.9),
                                     'colsample_bytree' :(0.05, 0.4)
                                    })

    XGbBO.maximize(init_points=10, n_iter=25, acq="ei", xi=0.01)
    print("-" * 53)
    timer(start_time)

    best_RMSE = round((-1.0 * XGbBO.res['max']['max_val']), 6)
    max_depth = XGbBO.res['max']['max_params']['max_depth']
    gamma = XGbBO.res['max']['max_params']['gamma']
    min_child_weight = XGbBO.res['max']['max_params']['min_child_weight']
    max_delta_step = XGbBO.res['max']['max_params']['max_delta_step']
    subsample = XGbBO.res['max']['max_params']['subsample']
    colsample_bytree = XGbBO.res['max']['max_params']['colsample_bytree']

    print("\n Best RMSE value: %f" % best_RMSE)
    print(" Best XGboost parameters:")
    print(" max_depth=%d gamma=%f min_child_weight=%f max_delta_step=%d subsample=%f colsample_bytree=%f" % (int(max_depth), gamma, min_child_weight, int(max_delta_step), subsample, colsample_bytree))

    start_time = timer(None)
    print("\n# Making Prediction")

    paramt = {
              'booster' : 'gbtree',
              'max_depth' : max_depth.astype(int),
              'gamma' : gamma,
              'eta' : 0.01,
              'objective': 'reg:linear',
              'nthread' : 8,
              'silent' : True,
              'eval_metric': 'rmse',
              'subsample' : subsample,
              'colsample_bytree' : colsample_bytree,
              'min_child_weight' : min_child_weight,
              'max_delta_step' : max_delta_step.astype(int),
              'seed' : 1001
              }

    xgbr = xgb.train(paramt, dtrain, num_boost_round=int(ITERbest*(1+(1/folds))))

    x_true = np.expm1(y)
    x_pred = np.expm1(xgbr.predict(dtrain))
# Normalized prediction error clipped to -20% to 20% range
    x_diff = np.clip(100 * ( (x_pred - x_true) / x_true ), -20, 20)
    plt.figure(1)
    plt.title("True vs Predicted Sale Prices")
    plt.scatter(x_true, x_pred, c=x_diff)
    plt.colorbar()
    plt.plot([x_true.min()-5000, x_true.max()+5000], [x_true.min()-5000, x_true.max()+5000], 'k--', lw=1)
    plt.xlabel('Sale Price')
    plt.ylabel('Predicted Sale Price')
    plt.xlim( 0, 800000 )
    plt.ylim( 0, 800000 )
    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    plt.savefig('./HousePrices-XGb-' + str(folds) + 'fold-train-predictions-01-v2.png')
    plt.show(block=False)

    y_pred = np.expm1(xgbr.predict(dtest))
    result = pd.DataFrame(y_pred, columns=['SalePrice'])
    result["Id"] = ids
    result = result.set_index("Id")
    print("\n First 10 Lines of Your Prediction:\n")
    print(result.head(10))
    now = datetime.now()
    sub_file = 'submission_XGb_' + str(best_RMSE) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print("\n Writing Submission File: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='Id')
    timer(start_time)

    history_df = pd.DataFrame(XGbBO.res['all']['params'])
    history_df2 = pd.DataFrame(XGbBO.res['all']['values'])
    history_df = pd.concat((history_df, history_df2), axis=1)
    history_df.rename(columns = { 0 : 'RMSE'}, inplace=True)
    history_df.index.names = ['Iteration']

    x, y, z = history_df['subsample'].values, history_df['colsample_bytree'].values, history_df['RMSE'].values
    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(0.35, 1.05, 100), np.linspace(0, 0.65, 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate
    rbf = scipy.interpolate.Rbf(x, y, z, function='multiquadric', smooth=0.5)
    zi = rbf(xi, yi)

    plt.figure(2)
    plt.title("Interpolated density distribution of C vs gamma")
    plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[0.35, 1.05, 0, 0.65], interpolation = 'lanczos')
    plt.scatter(x, y, c=z)
    plt.colorbar()
    plt.xlabel('subsample')
    plt.ylabel('colsample_bytree')
    plt.savefig('./HousePrices-XGb-' + str(folds) + 'fold-01-v2.png')
    plt.show(block=False)
    print("\n Optimization Plot Saved:  HousePrices-XGb-%dfold-01-v2.png" % folds)

    history_df['RMSE'] = -1.0 * history_df['RMSE']
    history_df.to_csv("./HousePrices-XGb-" + str(folds) + "fold-01-v2-grid.csv")
    print("\n Grid Search Results Saved:  HousePrices-XGb-%dfold-01-v2-grid.csv\n" % folds)
