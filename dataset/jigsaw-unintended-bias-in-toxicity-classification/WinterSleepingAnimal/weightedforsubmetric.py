import numpy as np
import pandas as pd
import os
import sys
import random
import keras
import tensorflow as tf
import json

sys.path.insert(0, '../input/pretrained-bert-including-scripts/master/bert-master')
sys.path.insert(0, '../input/kerasbert/keras_bert')
# os.system("cp -r '../input/kerasbert/keras_bert' '/kaggle/working'")
print(os.listdir('.'))
BERT_PRETRAINED_DIR = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12'
# print('***** BERT pretrained directory: {} *****'.format(BERT_PRETRAINED_DIR))
NUM_MODELS = 2
maxlen = 72
nb_epochs= 2
bsz = 128

# ## Load raw model
from keras_bert.bert import get_model
from keras_bert.loader import load_trained_model_from_checkpoint
from keras.optimizers import Adam
adam = Adam(lr=2e-5, decay=0.01)
# print('load bert_model')

# config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
# checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
# model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, seq_len=maxlen)
# model.summary(line_length=120)


# ## Build classification model
# As the Extract layer extracts only the first token where "['CLS']" used to be, we just take the layer and connect to the single neuron output.
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
import keras.backend as K
import re
# import codecs

def custom_loss(y_true, y_pred):
    return binary_crossentropy(K.reshape(y_true[:,0],(-1,1)), y_pred) * y_true[:,1]

# sequence_output  = model.layers[-6].output
# pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
# model3  = Model(inputs=model.input, outputs=pool_output)
# model3.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer=adam)
# model3.summary()
def build_model(num_aux_targets, loss_weight):
    config_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_config.json')
    checkpoint_file = os.path.join(BERT_PRETRAINED_DIR, 'bert_model.ckpt')
    model = load_trained_model_from_checkpoint(config_file, checkpoint_file, training=True, seq_len=maxlen)
    sequence_output  = model.layers[-6].output
    result = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
    aux_result = Dense(num_aux_targets, activation='sigmoid')(sequence_output)
    model3  = Model(inputs=model.input, outputs=[result, aux_result])
    del model, config_file, checkpoint_file
    model3.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer=adam)
    return model3



# ## Prepare Data, Training, Predicting
# 
# First the model need train data like [token_input,seg_input,masked input], here we set all segment input to 0 and all masked input to 1.
# 
# Still I am finding a more efficient way to do token-convert-to-ids

# def convert_lines(example, max_seq_length, tokenizer):
#     max_seq_length -= 2
#     all_tokens = []
#     longer = 0
#     for i in range(example.shape[0]):
#       tokens_a = tokenizer.tokenize(example[i])
#       if len(tokens_a)>max_seq_length:
#         tokens_a = tokens_a[:max_seq_length]
#         longer += 1
#       one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
#       all_tokens.append(one_token)
#     print(longer)
#     return np.array(all_tokens)
    

dict_path = os.path.join(BERT_PRETRAINED_DIR, 'vocab.txt')
import tokenization  #Actually keras_bert contains tokenization part, here just for convenience
tokenizer = tokenization.FullTokenizer(vocab_file=dict_path, do_lower_case=True)
print('build tokenizer done')
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
# train_df = train_df.sample(frac=0.01, random_state = 42)
#train_df['comment_text'] = train_df['comment_text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)

train_lines, train_labels = train_df['comment_text'].values, train_df.target.values 
identity_columns = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
# Overall
weights = np.ones((len(train_labels),)) / 4
# Subgroup
weights += (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) / 4
# Background Positive, Subgroup Negative
weights += (( (train_df.target.values>=0.5).astype(bool).astype(np.int) +
   (train_df[identity_columns].fillna(0).values<0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
# Background Negative, Subgroup Positive
weights += (( (train_df.target.values<0.5).astype(bool).astype(np.int) +
   (train_df[identity_columns].fillna(0).values>=0.5).sum(axis=1).astype(bool).astype(np.int) ) > 1 ).astype(bool).astype(np.int) / 4
loss_weight = 1.0 / weights.mean()
train_labels = np.vstack([(train_df.target.values>=0.5).astype(np.int), weights]).T
y_aux_train = train_df[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values

print('sample used', train_lines.shape)
# token_input = convert_lines(train_lines, maxlen, tokenizer)
# sampleindx = np.random.choice(token_input[0], size=(902437,))
# token_input = np.load('../input/datanpy/token.npy')
# sampleindx = np.random.choice(token_input[0], size=(451219,))
# token_input = token_input[sampleindx]
# train_labels = train_labels[sampleindx]
# y_aux_train = y_aux_train[sampleindx]
# seg_input = np.zeros((token_input.shape[0], maxlen))
# mask_input = np.ones((token_input.shape[0], maxlen))
# print('coverted !')
# print(token_input.shape)
# print(seg_input.shape)
# print(mask_input.shape)

# sequence_output  = model.layers[-6].output
# pool_output = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
# result = Dense(1, activation='sigmoid',kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.02),name = 'real_output')(sequence_output)
# aux_result = Dense(num_aux_targets, activation='sigmoid')(sequence_output)
# model3  = Model(inputs=model.input, outputs=pool_output)
# model3  = Model(inputs=model.input, outputs=[result, aux_result])
# model3.compile(loss=[custom_loss,'binary_crossentropy'], loss_weights=[loss_weight, 1.0], optimizer=adam)
# model3.summary()
#load test data
# test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
#test_df['comment_text'] = test_df['comment_text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)
# eval_lines = test_df['comment_text'].values
# print(eval_lines.shape)

# token_input2 = convert_lines(eval_lines, maxlen, tokenizer)
# token_input2 = np.load('../input/datanpy/token_test.npy')
# seg_input2 = np.zeros((token_input2.shape[0], maxlen))
# mask_input2 = np.ones((token_input2.shape[0], maxlen))
# print('test data done')
# print(token_input2.shape)
# print(seg_input2.shape)
# print(mask_input2.shape)
# hehe = model3.predict([token_input2, seg_input2, mask_input2], verbose=1, batch_size=2048)


# import pickle
import gc

# from keras.callbacks import ModelCheckpoint
from keras.callbacks import LearningRateScheduler
# checkpointer = ModelCheckpoint(filepath="checkpoint-{epoch:02d}.hdf5", verbose=1, period=1)
# model3.fit([token_input, seg_input, mask_input], train_labels, 
#     batch_size=bsz, epochs=nb_epochs, verbose=1, callbacks=[checkpointer])
del identity_columns, weights, tokenizer, train_lines, train_df
gc.collect()

checkpoint_predictions = []
weights = []

for model_idx in range(NUM_MODELS):
    model3 = build_model(y_aux_train.shape[-1], loss_weight)
    for global_epoch in range(nb_epochs):
        token_input = np.load('../input/datanpy/token.npy')
        sampleindx = np.random.choice(token_input[0], size=(451219,))
        token_input = token_input[sampleindx]
        train_labels3 = train_labels[sampleindx]
        y_aux_train3 = y_aux_train[sampleindx]
        seg_input = np.zeros((token_input.shape[0], maxlen))
        mask_input = np.ones((token_input.shape[0], maxlen))
        steps_per_epoch = np.ceil(len(token_input)/bsz)
        print('steps_per_epoch: ', steps_per_epoch)
        # print(gc.collect())
        model3.fit([token_input, seg_input, mask_input], [train_labels3, y_aux_train3], 
                    batch_size=bsz, epochs=1, verbose=2, 
                    callbacks=[LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch))])
        del token_input, sampleindx, train_labels3, y_aux_train3, seg_input, mask_input, steps_per_epoch 
        token_input2 = np.load('../input/datanpy/token_test.npy')
        seg_input2 = np.zeros((token_input2.shape[0], maxlen))
        mask_input2 = np.ones((token_input2.shape[0], maxlen))
        checkpoint_predictions.append(model3.predict([token_input2, seg_input2, mask_input2], batch_size=2048)[0].flatten())
        del token_input2, seg_input2, mask_input2
        gc.collect()
        weights.append(2 ** global_epoch)
    del model3
    gc.collect()


predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv', index_col='id')
submission.prediction = predictions
submission.reset_index(drop=False, inplace=True)
submission.to_csv('submission.csv', index=False)