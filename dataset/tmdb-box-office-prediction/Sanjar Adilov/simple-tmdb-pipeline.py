"""
TMDB Box Office Prediction
**************************

Create an estimator pipeline comprising feature transformations,
feature selection, target transformation, and regressor.
Quantify the quality of the estimator's predictions using n-fold
cross-validation and RMSLE criterion.
[Optional] Tune the estimator's hyper-parameters applying
cross-validated randomized search over a specified parameter grid.
Make predictions and save a submission file.

Functions
---------
main:
    Main function.
parse_options:
    Parse command line arguments.
load_data:
    Load training and testing data.
load_model:
    Create a model pipeline.
evaluate_model:
    Evaluate a score by cross-validation.
optimize_hyperparameters:
    Optimize the parameters of the model by cross-validated randomized search.
make_predictions:
    Make and save predictions.

Classes
-------
BudgetTransformer:
    Complete missing values in *budget*.
CompanyTransformer:
    Parse *production_companies*.
CountryTransformer:
    Parse *production_countries*.
DateTransformer:
    Parse *release_date*.
GenreTransformer:
    Parse *genres*.
KeywordTransformer:
    Parse *Keywords*.
LanguageTransformer:
    Parse *original_language*.
MissingIndicatorTransformer:
    Get missing indicator variables from *belongs_to_collection*,
    *homepage*, and *tagline*.
SpokenLanguageTransformer:
    Parse *spoken_languages*.
"""

import argparse
import pathlib
import re
import shutil
import sys
import tempfile

import nltk
import numpy as np
import pandas as pd
import scipy.stats

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.compose import (
    ColumnTransformer,
    TransformedTargetRegressor,
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import (
    ENGLISH_STOP_WORDS,
    CountVectorizer,
    TfidfVectorizer,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer
from sklearn.model_selection import (
    cross_val_score,
    RandomizedSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
)


COLUMNS = [
    # Features
    'belongs_to_collection',
    'budget',
    'genres',
    'homepage',
    'Keywords',
    'original_language',
    'original_title',
    'popularity',
    'production_countries',
    'production_companies',
    'release_date',
    'spoken_languages',
    'runtime',
    'tagline',
    # Target
    'revenue',
]


def main():
    options = parse_options()

    data, target, test_df = load_data(options.train_file,
                                      options.test_file)

    pipe = load_model(options.cache_directory)

    if options.n_folds != 0:
        evaluate_model(pipe, data, target, options.n_folds)

    if options.n_search != 0:
        pipe = optimize_hyperparameters(pipe, data, target, options.n_search)

        if options.n_folds != 0:
            evaluate_model(pipe, data, target, options.n_folds)

    make_predictions(pipe, data, target, test_df)

    if options.cache_directory is not None:
        shutil.rmtree(options.cache_directory)


def parse_options():
    """usage: run.py [-h] [-c CACHE_DIRECTORY] [-e N_FOLDS] [-r N_SEARCH]
              train_file test_file

    Solve TMDB Box Office Prediction Contest.

    positional arguments:
        train_file            Training data filename.
        test_file             Testing data filename.

    optional arguments:
        -h, --help            show this help message and exit
        -c CACHE_DIRECTORY, --cache-directory CACHE_DIRECTORY
                              Cache directory to store metadata.
                              (default: None)
        -e N_FOLDS, --evaluation N_FOLDS
                              Evaluate model on training data using n-fold CV.
                              (default: 10)
        -r N_SEARCH, --randomized-hyperparam-search N_SEARCH
                              Optimize hyperparameters. (default: 0)
    """

    def cache_directory(argument):
        if argument is None:
            return argument

        if not isinstance(argument, str):
            raise argparse.ArgumentTypeError(f'{argument} is not a valid path')

        if argument == 'temp':
            argument = tempfile.mkdtemp()

        directory = pathlib.Path(argument)
        if not directory.exists():
            raise EnvironmentError(f'Directory {argument} does not exist')

        return str(directory)

    prog = pathlib.Path(sys.argv[0]).name

    parser = argparse.ArgumentParser(
        prog=prog,
        description="""Solve TMDB Box Office Prediction Contest.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--train_file',
                        help='Training data filename.',
                        default='../input/train.csv',
                        required=False)
    parser.add_argument('--test_file',
                        help='Testing data filename.',
                        default='../input/test.csv',
                        required=False)
    parser.add_argument('-c', '--cache-directory',
                        help='Cache directory to store metadata.',
                        type=cache_directory,
                        default=None,
                        dest='cache_directory')
    parser.add_argument('-e', '--evaluation',
                        help='Evaluate model on training data using nfold CV.',
                        type=int,
                        default=10,
                        dest='n_folds')
    parser.add_argument('-r', '--randomized-hyperparam-search',
                        help='Optimize hyperparameters.',
                        type=int,
                        default=0,
                        dest='n_search')

    # Inexplicable argument required by kaggle
    parser.add_argument('-f')

    return parser.parse_args()


def load_data(train_filename, test_filename):
    """Load training and testing data.
    """
    train_df = pd.read_csv(train_filename,
                           usecols=COLUMNS)
    test_df = pd.read_csv(test_filename,
                          usecols=COLUMNS[:-1])

    data = train_df.drop('revenue', axis=1)
    target = train_df['revenue'].copy()

    return data, target, test_df


def load_model(cache_directory):
    """Create model pipeline.

    Parameters
    ----------
    cache_directory : str or path-like or None
        Cache directory to store metadata.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        Pipeline of transformers and estimators.
    """
    # Common transformers
    interaction_trans = PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=False,
    )
    log_trans = FunctionTransformer(
        func=np.log1p,
        inverse_func=np.exp,
        validate=False,
        check_inverse=False,
    )

    # Common imputers
    median_imputer = SimpleImputer(
        strategy='median',
    )

    # Special imputers
    company_imputer = SimpleImputer(
        strategy='constant',
        fill_value='Unknown',
    )
    country_imputer = SimpleImputer(
        strategy='constant',
        fill_value='US',
    )
    date_imputer = SimpleImputer(
        strategy='constant',
        fill_value='20/3/2001',
    )
    genre_imputer = SimpleImputer(
        strategy='constant',
        fill_value='movie',
    )
    keyword_imputer = SimpleImputer(
        strategy='constant',
        fill_value='film',
    )
    spoken_language_imputer = SimpleImputer(
        strategy='constant',
        fill_value='en',
    )

    # Special transformers
    budget_trans = BudgetTransformer(scipy.stats.expon)
    missing_trans = MissingIndicatorTransformer()
    title_trans = TfidfVectorizer(
        max_features=5,
        tokenizer=get_lemmas,
        stop_words=STOP_WORDS,
    )

    # Transformer pipelines
    company_pipe = Pipeline(
        steps=[
            ('imputer', company_imputer),
            ('trans', CompanyTransformer()),
            ('vect', CountVectorizer(max_features=10)),
        ],
        memory=cache_directory,
    )
    country_pipe = Pipeline(
        steps=[
            ('imputer', country_imputer),
            ('trans', CountryTransformer()),
            ('vect', CountVectorizer(min_df=300)),
        ],
        memory=cache_directory,
    )
    date_pipe = Pipeline(
        steps=[
            ('imputer', date_imputer),
            ('trans', DateTransformer()),
        ],
        memory=cache_directory,
    )
    genre_pipe = Pipeline(
        steps=[
            ('imputer', genre_imputer),
            ('trans', GenreTransformer()),
            ('vect', CountVectorizer()),
        ],
        memory=cache_directory,
    )
    keyword_pipe = Pipeline(
        steps=[
            ('imputer', keyword_imputer),
            ('trans', KeywordTransformer()),
            ('vect', TfidfVectorizer(min_df=150)),
        ],
        memory=cache_directory,
    )
    language_pipe = Pipeline(
        steps=[
            ('trans', LanguageTransformer()),
            ('vect', OneHotEncoder(handle_unknown='ignore')),
        ],
        memory=cache_directory,
    )
    spoken_language_pipe = Pipeline(
        steps=[
            ('imputer', spoken_language_imputer),
            ('trans', SpokenLanguageTransformer()),
            ('vect', CountVectorizer(min_df=75)),
        ],
        memory=cache_directory,
    )

    # Column transformer
    column_transformer = ColumnTransformer(
        transformers=[
            ('budget_trans', budget_trans, ['budget']),
            ('company_trans', company_pipe, ['production_companies']),
            ('country_trans', country_pipe, ['production_countries']),
            ('date_trans', date_pipe, ['release_date']),
            ('genre_trans', genre_pipe, ['genres']),
            ('keyword_trans', keyword_pipe, ['Keywords']),
            ('language_trans', language_pipe, ['original_language']),
            ('log_trans', log_trans, [
                 'budget',
                 'popularity',
             ]),
            ('missing_trans', missing_trans, [
                 'belongs_to_collection',
                 'homepage',
                 'tagline',
             ]),
            ('poly_trans', interaction_trans, [
                 'budget',
                 'popularity',
             ]),
            ('runtime_imputer', median_imputer, ['runtime']),
            ('spoken_lang_trans', spoken_language_pipe, ['spoken_languages']),
            ('title_trans', title_trans, 'original_title'),
        ],
        remainder='drop',
        n_jobs=1,
    )

    # Binomial feature selector
    feature_selector = VarianceThreshold(
        threshold=0.98 * (1 - 0.98),
    )

    # Estimators
    regressor = GradientBoostingRegressor(
        n_estimators=330,
        learning_rate=0.115,
        max_depth=3,
        min_samples_split=50,
        max_features=0.7,
        subsample=0.63,
        n_iter_no_change=27,

        random_state=0,
    )
    transformed_regressor = TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.exp,
        check_inverse=False,
    )

    # Main pipeline
    pipe = Pipeline(
        steps=[
            ('union', column_transformer),
            ('selector', feature_selector),
            ('regressor', transformed_regressor),
        ],
        memory=cache_directory,
    )

    return pipe


def evaluate_model(model, data, target, n_folds):
    """Perform n-fold cross-validation on training data with a given model.
    """
    cv_scores = cross_val_score(
        estimator=model,
        X=data,
        y=target,
        scoring=root_mean_squared_log_error_scorer,
        cv=n_folds,
    )
    print(f'Mean RMSLE: {cv_scores.mean():.3f} (+/-{cv_scores.std():.3f})\n')


def optimize_hyperparameters(pipe, data, target, n_search):
    """Perform randomized search and return the best estimator.

    Pipe should be comprised of column transformer ('union') and
    target transformed regressor ('regressor').
    """
    parameter_distributions = {
        'union__company_trans__vect__max_features': range(2, 51),
        'union__country_trans__vect__min_df': range(150, 301),
        'union__keyword_trans__vect__min_df': range(100, 151),
        'union__spoken_lang_trans__vect__min_df': range(20, 121),
        'union__title_trans__max_features': range(2, 16),

        'regressor__regressor__n_estimators': range(250, 351),
        'regressor__regressor__learning_rate': scipy.stats.uniform(0.11, 0.02),
        'regressor__regressor__min_samples_split': range(15, 61),
        'regressor__regressor__max_depth': range(3, 6),
        'regressor__regressor__max_features': scipy.stats.uniform(0.6, 0.4),
        'regressor__regressor__n_iter_no_change': range(13, 31),
    }

    grid = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=parameter_distributions,
        n_iter=n_search,
        scoring=root_mean_squared_log_error_scorer,
        iid=False,
        n_jobs=-1,
        cv=4,
        verbose=True,
        error_score='raise',
        return_train_score=False,
    )
    grid.fit(data, target)

    return grid.best_estimator_


def make_predictions(model, train_data, target, test_data):
    """Fit the model, predict results, and save a submission file.
    """
    model.fit(train_data, target)
    predictions = model.predict(test_data)

    predictions_df = pd.DataFrame({
        'id': test_data.index + 3001,
        'revenue': predictions,
    })
    predictions_df.to_csv('submission.csv', index=False)


# Global constant used for lemmatizing various terms.
LEMMATIZER = nltk.stem.WordNetLemmatizer()
# Stop words ignored by document vectorizers.
STOP_WORDS = ENGLISH_STOP_WORDS.union(['ha', 'le', 'u', 'wa'])


# Tokenizer function
def get_lemmas(document):
    regexp = re.compile(r'(?u)\b\w\w+\b', flags=re.IGNORECASE)
    words = regexp.findall(document)
    return [LEMMATIZER.lemmatize(word.lower()) for word in words]


# RMSLE function
def root_mean_squared_log_error(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred))))

# RMSLE scorer
root_mean_squared_log_error_scorer = make_scorer(root_mean_squared_log_error,
                                                 greater_is_better=False)


class BudgetTransformer(BaseEstimator, TransformerMixin):
    """Imputation transformer for completing missing values in *budget*.

    For a specified distribution (e.g. exponential), applies ML (Maximum
    Likelihood) method on nonempty data and obtains ML estimates. Missing
    values are imputed by generating random numbers according to specified RV
    and obtained MLE.

    Parameters
    ----------
    rv : an instance of scipy.stats.rv_continuous class
        Random variable.
    params : iterable, default: None
        Parameters of random variable rv.
        If not specified (i.e. None), can be calculated using `fit` or
        `fit_transform`.
    """

    def __init__(self, rv, params=None):
        self.rv = rv
        self.params = params

    def fit(self, X, y=None, **fit_params):
        """Apply ML and return MLE.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Input data to fit.
            Expected to contain missing values (zeros).

        Returns
        -------
        self : BudgetTransformer
        """
        items = X.squeeze()

        self.params = self.rv.fit(items[items != 0])

        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Input data to complete.

        Returns
        -------
        X_new : array-like, shape (n_samples, 1)
            Transformed data (missing values filled with random numbers).
        """
        items = X.squeeze()

        items[items == 0] = self.rv.rvs(*self.params,
                                        size=items[items == 0].shape[0])
        items = items.values.reshape(items.shape[0], 1)

        return items


class DateTransformer(BaseEstimator, TransformerMixin):
    """Transformer class that converts string timestamps into a matrix of
    year, month, and date values.
    """

    COLUMNS = ['year', 'month', 'day']

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        """Transform date strings to (year, month, date) matrix.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Input data.

        Returns
        -------
        X_new : numpy.ndarray, shape (n_samples, 3)
            Matrix containing years, months, and dates for every date in X.
        """
        items = X.squeeze()
        if not isinstance(items, pd.Series):
            items = pd.Series(items)
        items_date = items.apply(pd.to_datetime)

        return np.column_stack((
            items_date.apply(lambda t: t.year).values,
            items_date.apply(lambda t: t.month).values,
            items_date.apply(lambda t: t.day).values,
        ))


class LanguageTransformer(BaseEstimator, TransformerMixin):
    """Transformer class that filters languages in *original_language*.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        """Get new language series saving the most frequent ones and
        integrating others into one group.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Input data (series of strings).

        Returns
        -------
        X_new : numpy.ndarray, shape (n_samples, 1)
            Transformed data.
        """
        items = X.squeeze()
        mask = (
            (items != 'en')
            # & (items != ...)
        )
        items[mask] = 'other_lang'
        items = items.values.reshape(items.shape[0], -1)
        return items


class MissingIndicatorTransformer(BaseEstimator, TransformerMixin):
    """Transformer class that assigns indicator values for data containing
    relatively large number of missing values.
    """

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        """Transform data to (0, 1) vector.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Input data (series containing a lot of NaNs).

        Returns
        -------
        X_new : numpy.ndarray, shape (n_samples, 1)
            Transformed data with indicator values.
        """
        X[X.notnull()] = 1
        X[X.isnull()] = 0
        return X


class JSONTransformer(BaseEstimator, TransformerMixin):
    """Base transformer class for parsing JSON-like data.

    Custom transformers are created by inheriting this class and
    re-implementing `parse` classmethod.
    """

    @classmethod
    def parse(cls, item):
        """Parse single item of data.

        Parameters
        ----------
        item : str
            List of key-value items.
        """
        raise NotImplementedError()

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        """Transform data applying self.parse to every instance of the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, 1)
            Input data.

        Returns
        -------
        X_new : numpy.ndarray, shape (n_samples, 1)
            Transformed data.
        """
        items = X.squeeze()
        items = [self.parse(item) for item in items]
        items = np.array(items, dtype=object)
        return items


class CompanyTransformer(JSONTransformer):
    SUFFIXES = [
        'Animation',
        'Animations',
        'Artists',
        'Cinema',
        'Cinéma',
        'Classics',
        'Collection',
        'Communications',
        'Company',
        'Corporation',
        'Distribution',
        'Enterprises',
        'Entertainment',
        'Film',
        'Films',
        'Group',
        'Home',
        'International',
        'Ltd.',
        'Media',
        'Movies',
        'Network',
        'Networks',
        'Pictures',
        'Production',
        'Productions',
        'Studio',
        'Studios',
        'Television',
    ]
    COMPANY_RE = re.compile(
        r'''
        ((?P<name_with_suffix>.+?)
         (?=(\W+({suffixes})))
        |
         (?P<name>.+)
        )
        '''.format(suffixes='|'.join(SUFFIXES)),
        re.VERBOSE
    )

    @classmethod
    def parse(cls, item):
        """Parse list of production company key-values.

        Exclude common terms like 'Entertainment', 'Pictures', etc.
        Return a sequence of companies.

        Applying `eval` raises an exception if `item` is not a string of
        JSON-like items. In this case, simply return `item`
        (since it has already been imputed or parsed).
        """
        try:
            return ','.join(
                re.sub(
                    r'\W+',
                    r'_',
                    cls.COMPANY_RE.match(x['name']).group()
                )
                for x in eval(item)
            )
        except (NameError, TypeError):
            return item


class CountryTransformer(JSONTransformer):
    @classmethod
    def parse(cls, item):
        """Parse list of production countries.

        Return a sequence of countries.
        """
        try:
            return ','.join(x['iso_3166_1'] for x in eval(item))
        except (NameError, TypeError):
            return item


class GenreTransformer(JSONTransformer):
    @classmethod
    def parse(cls, item):
        """Parse list of genres.

        Return a sequence of genres.
        """
        try:
            return ','.join(x['name'] for x in eval(item))
        except (NameError, TypeError):
            return item


class KeywordTransformer(JSONTransformer):
    @classmethod
    def parse(cls, item):
        """Parse list of keywords.

        Exclude non-alphabetical symbols.

        Return a sequence of keywords.
        """
        try:
            return ','.join(
                LEMMATIZER.lemmatize(re.sub(r'[\W]+', r'_', x['name'].lower()))
                for x in eval(item)
            )
        except (NameError, TypeError):
            return item


class SpokenLanguageTransformer(JSONTransformer):
    @classmethod
    def parse(cls, item):
        """Parse list of spoken languages.

        Return a sequence of spoken languages.
        """
        try:
            return ','.join(x['iso_639_1'] for x in eval(item))
        except (NameError, TypeError):
            return item


main()
