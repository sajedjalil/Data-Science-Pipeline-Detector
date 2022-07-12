# %% [code]
from time import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR
from xgboost import XGBRegressor as XBGR
from sklearn.externals import joblib
import tensorflow as tf
import wget,json,os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential

# pip install wget
# pip install tensorflow-gpu==2.0.0


print(tf.__verion__)
print(os.listdir('.'))
print(os.getcwd())