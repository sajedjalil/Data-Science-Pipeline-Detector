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
    from scipy.sparse import csr_matrix, coo_matrix, hstack
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.cross_validation import cross_val_score, cross_val_predict
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt
    import scipy.interpolate
# UNCOMMENT THE NEXT LINE IF YOU HAVE bayes_opt INSTALLED
#    from bayes_opt import BayesianOptimization

############################################
#
#  Most of the lines below (#46-611)
#  are copied from an excellent library
#  for global optimization with gaussian
#  processes (see below). The only reason
#  for that is because Kaggle does not
#  have this library installed.
#
#  You can delete the section below
#  (everything between lines 22-621)
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
#  (everything between lines #22-621)
#  if you have bayes_opt installed
#  on your system. Don't forget to
#  uncomment line #19
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

def svrcv(log2C, log2gamma):
    cv_score = cross_val_score(
        SVR
        (
        C=math.pow(2,log2C),
        kernel='rbf',
        gamma=math.pow(2,log2gamma),
        cache_size=2000,
        verbose=False,
        max_iter=-1,
        shrinking=False,
        ),
        train_data,
        target,
        'mean_squared_error',
        cv=folds
        ).mean()
    return ( -1.0 * np.sqrt(-cv_score))

def scale_data(X, scaler=None):
    if not scaler:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def sparse_df_to_array(df):
    num_rows = df.shape[0]
    data = []
    row = []
    col = []
    for i, col_name in enumerate(df.columns):
        if isinstance(df[col_name], pd.SparseSeries):
            column_index = df[col_name].sp_index
            if isinstance(column_index, BlockIndex):
                column_index = column_index.to_int_index()
            ix = column_index.indices
            data.append(df[col_name].sp_values)
            row.append(ix)
            col.append(len(df[col_name].sp_values) * [i])
        else:
            data.append(df[col_name].values)
            row.append(np.array(range(0, num_rows)))
            col.append(np.array(num_rows * [i]))
    data_f = np.concatenate(data)
    row_f = np.concatenate(row)
    col_f = np.concatenate(col)
    arr = coo_matrix((data_f, (row_f, col_f)), df.shape, dtype=np.float64)
    return arr.tocsr()

if __name__ == "__main__":

    folds = 5

    CV = CountVectorizer(min_df=1, analyzer='word', max_features=30000, binary=True)

    print("\n Please read the comments carefully. The code can be much shorter if you follow directions.")

# Load data set and target values

    start_time = timer(None)
    print("\n# Reading and Processing Data")
    train = pd.read_csv("../input/train.csv", dtype = {'Id' : np.str} )
    print("\n Initial Train Set Matrix Dimensions: %d x %d" % (train.shape[0], train.shape[1]))
    target = np.log(np.array(train['SalePrice']))
    test = pd.read_csv("../input/test.csv", dtype = {'Id' : np.str} )
    print("\n Initial Test Set Matrix Dimensions: %d x %d" % (test.shape[0], test.shape[1]))
    ids = test['Id']

# Minimal feature engineering - this can be done much better
    train['Age'] = train['YrSold'] - train['YearBuilt']
    test['Age'] = train['YrSold'] - train['YearBuilt']
    train['AgeRemod'] = train['YrSold'] - train['YearRemodAdd']
    test['AgeRemod'] = train['YrSold'] - train['YearRemodAdd']
    train['Baths'] = train['FullBath'] + train['HalfBath']
    test['Baths'] = test['FullBath'] + test['HalfBath']
    train['BsmtBaths'] = train['BsmtFullBath'] + train['BsmtHalfBath']
    test['BsmtBaths'] = test['BsmtFullBath'] + test['BsmtHalfBath']

# Feature fixing - converting numerical to categorical, removing spaces from features, coverting single-letter features
    train['MSSubClass'] = train['MSSubClass'].astype(str)
    test['MSSubClass'] = test['MSSubClass'].astype(str)
    train['Alley'].replace('NA', 'NoAlley', inplace=True)
    test['Alley'].replace('NA', 'NoAlley', inplace=True)
    train['MSZoning'].replace('A', 'AG', inplace=True)
    train['MSZoning'].replace('C', 'CO', inplace=True)
    train['MSZoning'].replace('I', 'IN', inplace=True)
    test['MSZoning'].replace('A', 'AG', inplace=True)
    test['MSZoning'].replace('C', 'CO', inplace=True)
    test['MSZoning'].replace('I', 'IN', inplace=True)
    train['HouseStyle'].replace('1.5Fin', '1_5Fin', inplace=True)
    train['HouseStyle'].replace('1.5Unf', '1_5Unf', inplace=True)
    train['HouseStyle'].replace('2.5Fin', '2_5Fin', inplace=True)
    train['HouseStyle'].replace('2.5Unf', '2_5Unf', inplace=True)
    test['HouseStyle'].replace('1.5Fin', '1_5Fin', inplace=True)
    test['HouseStyle'].replace('1.5Unf', '1_5Unf', inplace=True)
    test['HouseStyle'].replace('2.5Fin', '2_5Fin', inplace=True)
    test['HouseStyle'].replace('2.5Unf', '2_5Unf', inplace=True)
    train['RoofMatl'].replace('Tar&Grv', 'Tar_Grv', inplace=True)
    test['RoofMatl'].replace('Tar&Grv', 'Tar_Grv', inplace=True)
    train['Exterior1st'].replace('Wd Sdng', 'WdSdng', inplace=True)
    test['Exterior1st'].replace('Wd Sdng', 'WdSdng', inplace=True)
    train['Exterior2nd'].replace('Wd Sdng', 'WdSdng', inplace=True)
    test['Exterior2nd'].replace('Wd Sdng', 'WdSdng', inplace=True)
    train['MasVnrType'].replace('None', 'NoVen', inplace=True)
    test['MasVnrType'].replace('None', 'NoVen', inplace=True)
    train['BsmtQual'].replace('NA', 'NB', inplace=True)
    test['BsmtQual'].replace('NA', 'NB', inplace=True)
    train['BsmtCond'].replace('NA', 'NB', inplace=True)
    test['BsmtCond'].replace('NA', 'NB', inplace=True)
    train['BsmtExposure'].replace('NA', 'NB', inplace=True)
    test['BsmtExposure'].replace('NA', 'NB', inplace=True)
    train['BsmtFinType1'].replace('NA', 'NB', inplace=True)
    test['BsmtFinType1'].replace('NA', 'NB', inplace=True)
    train['BsmtFinType2'].replace('NA', 'NB', inplace=True)
    test['BsmtFinType2'].replace('NA', 'NB', inplace=True)
    train['FireplaceQu'].replace('NA', 'NF', inplace=True)
    test['FireplaceQu'].replace('NA', 'NF', inplace=True)
    train['GarageType'].replace('NA', 'NG', inplace=True)
    test['GarageType'].replace('NA', 'NG', inplace=True)
    train['GarageFinish'].replace('NA', 'NG', inplace=True)
    test['GarageFinish'].replace('NA', 'NG', inplace=True)
    train['GarageQual'].replace('NA', 'NG', inplace=True)
    test['GarageQual'].replace('NA', 'NG', inplace=True)
    train['GarageCond'].replace('NA', 'NG', inplace=True)
    test['GarageCond'].replace('NA', 'NG', inplace=True)
    train['PoolQC'].replace('NA', 'NP', inplace=True)
    test['PoolQC'].replace('NA', 'NP', inplace=True)
    train['Fence'].replace('NA', 'NF', inplace=True)
    test['Fence'].replace('NA', 'NF', inplace=True)
    train['MiscFeature'].replace('NA', 'NoF', inplace=True)
    test['MiscFeature'].replace('NA', 'NoF', inplace=True)
    train['CentralAir'].replace('Y', 'Yes', inplace=True)
    train['CentralAir'].replace('N', 'No', inplace=True)
    test['CentralAir'].replace('Y', 'Yes', inplace=True)
    test['CentralAir'].replace('N', 'No', inplace=True)
    train['PavedDrive'].replace('Y', 'Yes', inplace=True)
    train['PavedDrive'].replace('N', 'No', inplace=True)
    train['PavedDrive'].replace('P', 'Partial', inplace=True)
    test['PavedDrive'].replace('Y', 'Yes', inplace=True)
    test['PavedDrive'].replace('N', 'No', inplace=True)
    test['PavedDrive'].replace('P', 'Partial', inplace=True)

# Feature consolidation
    train['Conditions'] = train[['Condition1', 'Condition2']].apply(lambda x: ' '.join(x), axis=1)
    test['Conditions'] = test[['Condition1', 'Condition2']].apply(lambda x: ' '.join(x), axis=1)
    train['MonthYearSold'] = train[['MoSold', 'YrSold']].astype(str).apply(lambda x: '_'.join(x), axis=1)
    test['MonthYearSold'] = test[['MoSold', 'YrSold']].astype(str).apply(lambda x: '_'.join(x), axis=1)

# Makee numerical and non-numerical subsets of data
    numerical = ['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'Age', 'AgeRemod', 'Baths', 'BedroomAbvGr', 'BsmtBaths', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtUnfSF', 'EnclosedPorch', 'Fireplaces', 'FullBath', 'GarageArea', 'GarageCars', 'GarageYrBlt', 'GrLivArea', 'HalfBath', 'KitchenAbvGr', 'LotArea', 'LotFrontage', 'LowQualFinSF', 'MasVnrArea', 'MiscVal', 'MoSold', 'OpenPorchSF', 'OverallCond', 'OverallQual', 'PoolArea', 'ScreenPorch', 'TotalBsmtSF', 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd', 'YrSold']
    non = ['Alley', 'BldgType', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtQual', 'CentralAir', 'Conditions', 'Electrical', 'ExterCond', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'Fence', 'FireplaceQu', 'Foundation', 'Functional', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 'Heating', 'HeatingQC', 'HouseStyle', 'KitchenQual', 'LandContour', 'LandSlope', 'LotConfig', 'LotShape', 'MasVnrType', 'MiscFeature', 'MonthYearSold', 'MSSubClass', 'MSZoning', 'Neighborhood', 'PavedDrive', 'PoolQC', 'RoofMatl', 'RoofStyle', 'SaleCondition', 'SaleType', 'Street', 'Utilities']

    train_num = train[numerical]
    train_num.reset_index(drop=True, inplace=True)
    train_non = train[non]
    train_non.reset_index(drop=True, inplace=True)
    test_num = test[numerical]
    test_num.reset_index(drop=True, inplace=True)
    test_non = test[non]
    test_non.reset_index(drop=True, inplace=True)
    train_non = train_non.replace(np.NaN, '')
    train_num = train_num.replace(np.NaN, -999)
    test_non = test_non.replace(np.NaN, '')
    test_num = test_num.replace(np.NaN, -999)

# Scale only numerical data
    train_test_num = pd.concat((train_num, test_num), ignore_index=True)
    train_test_num, scaler = scale_data( train_test_num )
    train_num, _ = scale_data( train_num, scaler )
    test_num, _ = scale_data( test_num, scaler )
    train_num = pd.DataFrame(train_num, columns=numerical)
    test_num = pd.DataFrame(test_num, columns=numerical)

# Create sparse matrices, first only from numerical data
    train_data = sparse_df_to_array(train_num)
    test_data = sparse_df_to_array(test_num)
    train_test_non = pd.concat((train_non, test_non), ignore_index=True)

# Convert individual categorical columns to sparse matrices, add to existing sparse matrices
    for i, col_name in enumerate(train_test_non.columns):
        CV.fit(train_test_non[col_name])
        train_sparse = CV.transform(train_non[col_name])
        test_sparse = CV.transform(test_non[col_name])
        train_data = hstack((train_data, train_sparse), format='csr')
        test_data = hstack((test_data, test_sparse), format='csr')

    print("\n Sparse Train Set Matrix Dimensions: %d x %d" % (train_data.shape[0], train_data.shape[1]))
    print("\n Sparse Test Set Matrix Dimensions: %d x %d\n" % (test_data.shape[0], test_data.shape[1]))
    timer(start_time)

    start_time = timer(None)
    print("\n# Global Optimization Search for SVR Parameters C and gamma\n")
    print("\n Please note that negative RMSE values will be shown below. This is because")
    print(" RMSE needs to be minimized, while Bayes optimizer always maximizes the function.\n")

    svrBO = BayesianOptimization(svrcv, {
                                         'log2C': (-6, 6),
                                         'log2gamma': (-12, 0)
                                        })

    svrBO.maximize(init_points=50, n_iter=150, acq="ei", xi=0.0)
    print("-" * 53)
    timer(start_time)

    best_RMSE = round((-1.0 * svrBO.res['max']['max_val']), 6)
    C = svrBO.res['max']['max_params']['log2C']
    gamma = svrBO.res['max']['max_params']['log2gamma']

    print("\n Best RMSE value: %f" % best_RMSE)
    print(" Best SVR parameters:  log2(C) = %f  log2(gamma) = %f" % (C, gamma))

    start_time = timer(None)
    print("\n# Making Prediction")

    svr = SVR(kernel='rbf', C=math.pow(2,C), gamma=math.pow(2,gamma), cache_size=2000, verbose=False, max_iter=-1, shrinking=False)

    x_true = np.array(train['SalePrice'])
    x_pred = np.exp(cross_val_predict(svr, X=train_data, y=target, cv=folds))

# Normalized prediction error clipped to -40% to 40% range
    x_diff = np.clip(100 * ( (x_pred - x_true) / x_true ), -40, 40)

# Make a figure showing colored true vs predicted values
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
    plt.savefig('./HousePrices-SVR-' + str(folds) + 'fold-train-predictions-01-v4.png')
    plt.show(block=False)

# Fit with optimized parameters and make a prediction
    svr.fit(train_data, target)
    y_pred = np.exp(svr.predict(test_data))
    result = pd.DataFrame(y_pred, columns=['SalePrice'])
    result["Id"] = ids
    result = result.set_index("Id")
    print("\n First 10 Lines of Your Prediction:\n")
    print(result.head(10))
    now = datetime.now()
    sub_file = 'submission_SVR_' + str(best_RMSE) + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print("\n Writing Submission File: %s" % sub_file)
    result.to_csv(sub_file, index=True, index_label='Id')
    timer(start_time)

# Save all parameters and RMSE values from Bayesian optimization
    history_df = pd.DataFrame(svrBO.res['all']['params'])
    history_df2 = pd.DataFrame(svrBO.res['all']['values'])
    history_df = pd.concat((history_df, history_df2), axis=1)
    history_df.rename(columns = { 'log2C' : 'log2(C)'}, inplace=True)
    history_df.rename(columns = { 'log2gamma' : 'log2(gamma)'}, inplace=True)
    history_df.rename(columns = { 0 : 'RMSE'}, inplace=True)
    history_df.index.names = ['Iteration']
    history_df.to_csv("./HousePrices-SVR-" + str(folds) + "fold-01-v4-grid.csv")
    print("\n Grid Search Results Saved:  HousePrices-SVR-%dfold-01-v4-grid.csv" % folds)

# Make a figure showing interpolated RMSE values from known C vs gamma outcomes
    x, y, z = history_df['log2(C)'].values, history_df['log2(gamma)'].values, history_df['RMSE'].values
# Set up a regular grid of interpolation points
    xi, yi = np.linspace(-6.5, 6.5, 100), np.linspace(-12.5, 0.5, 100)
    xi, yi = np.meshgrid(xi, yi)

# Interpolate
    rbf = scipy.interpolate.Rbf(x, y, z, function='multiquadric', smooth=0.5)
    zi = rbf(xi, yi)

    plt.figure(2)
    plt.title("Interpolated density distribution of C vs gamma")
    plt.imshow(zi, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[-6.5, 6.5, -12.5, 0.5], interpolation = 'lanczos')
    plt.scatter(x, y, c=z)
    plt.colorbar()
    plt.xlabel('log2(C)')
    plt.ylabel('log2(gamma)')
    plt.savefig('./HousePrices-SVR-' + str(folds) + 'fold-01-v4.png')
    plt.show(block=False)
    print("\n Optimization Plot Saved:  HousePrices-SVR-%dfold-01-v4.png\n" % folds)
    
