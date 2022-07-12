import os; os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import pandas as pd
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten, Activation
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K

# Load data from files.
print("Loading data from files...")
train_df = pd.read_table('../input/train.tsv')
test_df = pd.read_table('../input/test_stg2.tsv')
print(train_df.shape, test_df.shape)

# Handle missing data.
def fill_missing_values(df):
    df.category_name.fillna(value="Other", inplace=True)
    df.brand_name.fillna(value="Missing", inplace=True)
    df.item_description.fillna(value="None", inplace=True)
    return df

train_df = fill_missing_values(train_df)
test_df = fill_missing_values(test_df)

# Replace 123 gb to 123gb with case insensitive.
train_df['item_description'] = train_df['item_description'].str.replace(r'\b(\d+)\s(gb)\b', r'\1\2', case=False)
train_df['name'] = train_df['name'].str.replace(r'\b(\d+)\s(gb)\b', r'\1\2', case=False)
test_df['item_description'] = test_df['item_description'].str.replace(r'\b(\d+)\s(gb)\b', r'\1\2', case=False)
test_df['name'] = test_df['name'].str.replace(r'\b(\d+)\s(gb)\b', r'\1\2', case=False)

# Scale target variable to log.
train_df["target"] = np.log1p(train_df.price)

# Split training examples into train/dev examples.
train_df, dev_df = train_test_split(train_df, random_state=123, train_size=0.99)

#############
# RNN Model #
#############
# Handle categorical data.
print("Processing categorical data...")
le = LabelEncoder()

le.fit(np.hstack([train_df.category_name, dev_df.category_name, test_df.category_name]))
train_df.category_name = le.transform(train_df.category_name)
dev_df.category_name = le.transform(dev_df.category_name)
test_df.category_name = le.transform(test_df.category_name)

le.fit(np.hstack([train_df.brand_name, dev_df.brand_name, test_df.brand_name]))
train_df.brand_name = le.transform(train_df.brand_name)
dev_df.brand_name = le.transform(dev_df.brand_name)
test_df.brand_name = le.transform(test_df.brand_name)

del le

# Handle text data.
print("Transforming name to sequences...")
print("   Fitting tokenizer...")
tok_raw = Tokenizer()
tok_raw.fit_on_texts(train_df.name)

print("   Transforming text to sequences...")
train_df['seq_name'] = tok_raw.texts_to_sequences(train_df.name.str.lower())
dev_df['seq_name'] = tok_raw.texts_to_sequences(dev_df.name.str.lower())
test_df['seq_name'] = tok_raw.texts_to_sequences(test_df.name.str.lower())

print("Transforming item description to sequences...")
print("    Fitting tokenizer...")
tok_raw.fit_on_texts(train_df.item_description)

print("    Transforming text to sequences...")
train_df['seq_item_description'] = tok_raw.texts_to_sequences(train_df.item_description.str.lower())
dev_df['seq_item_description'] = tok_raw.texts_to_sequences(dev_df.item_description.str.lower())
test_df['seq_item_description'] = tok_raw.texts_to_sequences(test_df.item_description.str.lower())

del tok_raw

MAX_NAME_SEQ = 10
MAX_ITEM_DESC_SEQ = 75
MAX_NAME = np.max(train_df.seq_name.apply(lambda x: np.max(x) if len(x) > 0 else 0)) + 1
MAX_ITEM_DESC = np.max(train_df.seq_item_description.apply(lambda x: np.max(x) if len(x) > 0 else 0)) + 1
MAX_CATEGORY = np.max([train_df.category_name.max(), dev_df.category_name.max(), test_df.category_name.max()]) + 1
MAX_BRAND = np.max([train_df.brand_name.max(), dev_df.brand_name.max(), test_df.brand_name.max()]) + 1
MAX_CONDITION = np.max([train_df.item_condition_id.max(), dev_df.item_condition_id.max(), test_df.item_condition_id.max()]) + 1

# Get data for training the model.
def get_keras_data(df):
    X = {
        'name': pad_sequences(df.seq_name, maxlen=MAX_NAME_SEQ),
	    'item_desc': pad_sequences(df.seq_item_description, maxlen=MAX_ITEM_DESC_SEQ),
        'brand_name': np.array(df.brand_name),
        'category_name': np.array(df.category_name),
        'item_condition': np.array(df.item_condition_id),
        'num_vars': np.array(df[["shipping"]])}
    return X

X_train = get_keras_data(train_df)
Y_train = train_df.target.values.reshape(-1, 1)

X_dev = get_keras_data(dev_df)
Y_dev = dev_df.target.values.reshape(-1, 1)

X_test = get_keras_data(test_df)

# Define RNN model to solve the problem.
def new_rnn_model(lr=0.001, decay=0.0):    
    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(MAX_NAME, 20)(name)
    emb_item_desc = Embedding(MAX_ITEM_DESC, 60)(item_desc)
    emb_brand_name = Embedding(MAX_BRAND, 10)(brand_name)
    emb_category_name = Embedding(MAX_CATEGORY, 10)(category_name)

    # rnn layers
    rnn_layer1 = GRU(16) (emb_item_desc)
    rnn_layer2 = GRU(8) (emb_name)

    # main layers
    main_l = concatenate([
        Flatten() (emb_brand_name),
        Flatten() (emb_category_name),
        item_condition,
        rnn_layer1,
        rnn_layer2,
        num_vars,
    ])

    main_l = Dense(256)(main_l)
    main_l = Activation('elu')(main_l)

    main_l = Dense(128)(main_l)
    main_l = Activation('elu')(main_l)

    main_l = Dense(64)(main_l)
    main_l = Activation('elu')(main_l)

    # the output layer.
    output = Dense(1, activation="linear") (main_l)

    model = Model([name, item_desc, brand_name , category_name, item_condition, num_vars], output)

    optimizer = Adam(lr=lr, decay=decay)
    model.compile(loss="mse", optimizer=optimizer)

    return model

print("Defining RNN model...")

# Model hyper parameters.
BATCH_SIZE = 1024
epochs = 2

# Calculate learning rate decay.
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_train['name']) / BATCH_SIZE) * epochs
lr_init, lr_fin = 0.007, 0.0005
lr_decay = exp_decay(lr_init, lr_fin, steps)

rnn_model = new_rnn_model(lr=lr_init, decay=lr_decay)

print("Fitting RNN model to training examples...")
rnn_model.fit(
        X_train, Y_train, epochs=epochs, batch_size=BATCH_SIZE,
        validation_data=(X_dev, Y_dev), verbose=2,
)

# Evaluate the model.
def rmsle(Y, Y_pred):
    # Y and Y_red have already been in log scale.
    assert Y.shape == Y_pred.shape
    return np.sqrt(np.mean(np.square(Y_pred - Y )))

print("Evaluating the model on validation data...")
Y_dev_preds_rnn = rnn_model.predict(X_dev, batch_size=BATCH_SIZE)
v_rmsle = rmsle(Y_dev, Y_dev_preds_rnn)
print(" RMSLE error:", v_rmsle)

# Make prediction for test data.
print("Making prediction for test data...")
rnn_preds = rnn_model.predict(X_test, batch_size=BATCH_SIZE)
rnn_preds = np.expm1(rnn_preds)

del rnn_model, X_train, X_dev, X_test
gc.collect()

###############
# Ridge Model #
###############
full_df = pd.concat([train_df, dev_df, test_df])

# Convert values to string for later processing.
print("Handling missing values...")
full_df['shipping'] = full_df['shipping'].astype(str)
full_df['item_condition_id'] = full_df['item_condition_id'].astype(str)
full_df['category_name'] = full_df['category_name'].astype(str)
full_df['brand_name'] = full_df['brand_name'].astype(str)

train_df = full_df.iloc[:train_df.shape[0], :]
dev_df = full_df.iloc[train_df.shape[0]:train_df.shape[0]+dev_df.shape[0], :]
test_df = full_df.iloc[train_df.shape[0]+dev_df.shape[0]:, :]
print(train_df.shape, dev_df.shape, test_df.shape)

del full_df
gc.collect()

# Transform data to vectorization.
default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(train_df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        token_pattern=r'(?u)\b(?:\w\w+)|(?:\d+\.|/\d+)\b',
        preprocessor=build_preprocessor('name'))),
    ('category_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('category_name'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        token_pattern=r'(?u)\b(?:\w\w+)|(?:\d+\.|/\d+)\b',
        preprocessor=build_preprocessor('item_description'))),
])

print("Vecterizing training data...")
X_train = vectorizer.fit_transform(train_df.values)
Y_train = train_df.target.values.reshape(-1, 1)
del train_df
gc.collect()

print("Vectorizing develop data...")
X_dev = vectorizer.transform(dev_df.values)
Y_dev = dev_df.target.values.reshape(-1, 1)
del dev_df
gc.collect()

print("Vectorizing test data...")
X_test = vectorizer.transform(test_df.values)
test_id = test_df.test_id
del test_df
gc.collect()

print(X_train.shape, X_dev.shape, X_test.shape)

ridge_model = Ridge(
    solver='auto', fit_intercept=True, alpha=0.5,
    max_iter=100, normalize=False, tol=0.05,
)
ridge_model.fit(X_train, Y_train)

Y_dev_preds_ridge = ridge_model.predict(X_dev)
Y_dev_preds_ridge = Y_dev_preds_ridge.reshape(-1, 1)
print("RMSL error on dev set:", rmsle(Y_dev, Y_dev_preds_ridge))

print("Predicting test data using Ridge model...")
ridge_preds = ridge_model.predict(X_test)
ridge_preds = np.expm1(ridge_preds)

# Evaluate RNN + Ridge models.
def aggregate_predicts(Y1, Y2):
    assert Y1.shape == Y2.shape
    ratio = 0.63
    return Y1 * ratio + Y2 * (1.0 - ratio)

Y_dev_preds = aggregate_predicts(Y_dev_preds_rnn, Y_dev_preds_ridge)
print("RMSL error for RNN + Ridge on dev set:", rmsle(Y_dev, Y_dev_preds))

# Create submission
preds = aggregate_predicts(rnn_preds, ridge_preds)
submission = pd.DataFrame({
        "test_id": test_id,
        "price": preds.reshape(-1),
})
submission.to_csv("./rnn_submission.csv", index=False)

print("Finished!!!")