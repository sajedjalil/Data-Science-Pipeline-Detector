from typing import Callable, Dict, Optional, List, Tuple
import os
import sys
from functools import partial
import random

from scipy.stats import spearmanr
from scipy.stats import ks_2samp
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedShuffleSplit

import torch
import tensorflow as tf
import tensorflow.keras.backend as K
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel


def seed_everything(seed: int = 42):
    # Python/TF Seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)

    # Torch Seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def spearmanr_ignore_nan(trues: np.array, preds: np.array):
    return np.nanmean(
        [spearmanr(ta, np.nan_to_num(pa) + 1e-7).correlation for ta, pa in
         zip(np.transpose(trues), np.transpose(preds))])


from tensorflow.keras.callbacks import Callback
import logging


class CustomMetricEarlyStoppingCallback(Callback):
    def __init__(
            self, data: Tuple[np.array], training_data: Optional[Tuple[np.array]] = None,
            score_func: Callable = None, min_delta: float = 0, patience: int = 0, verbose: int = 0, mode: str = 'auto',
            baseline: float = None, restore_best_weights: bool = False):

        super().__init__()

        self.x_train: Optional[np.arary] = None
        self.y_train: Optional[np.arary] = None
        if training_data is not None:
            self.x_train, self.y_train = training_data

        self.x_valid, self.y_valid = data
        self.score_func = score_func

        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logging.warning(f'EarlyStopping mode {mode} is unknown, fallback to auto mode.')
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            self.monitor_op = np.greater

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = spearmanr_ignore_nan(self.y_valid, self.model.predict(self.x_valid))
        if self.y_train is not None and self.x_train is not None:
            current_train = spearmanr_ignore_nan(self.y_train, self.model.predict(self.x_train))
            diff = current_train - current
            print(
                f'\nEarlyStopping Metric: {current:.3f}, training: {current_train:.3f}, fitting diff: {diff:.3f}\n')
        else:
            print(
                f'\nEarlyStopping Metric: {current:.3f}, best: {self.best:.3f}\n')

        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_monitor_value(self, logs):
        logs = logs or {}
        monitor_value = None
        return monitor_value


# TF model block BEGIN
########################################################################################################################
def _configure_pretrained_model_block(model, max_seq_length: int, is_distilled: bool = False):
    # if config.output_hidden_states = True, obtain hidden states via bert_model(...)[-1]
    embedding_index = 0
    if model.config.output_hidden_states:
        embedding_index = -1

    input_ids = tf.keras.layers.Input((max_seq_length,), dtype=tf.int32)
    attention_mask = tf.keras.layers.Input((max_seq_length,), dtype=tf.int32)
    if is_distilled:
        embedding = model(input_ids, attention_mask=attention_mask)[embedding_index]
        return (input_ids, attention_mask), tf.keras.layers.GlobalAveragePooling1D()(embedding)

    token_type_ids = tf.keras.layers.Input((max_seq_length,), dtype=tf.int32)
    embedding = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[embedding_index]
    return (input_ids, attention_mask, token_type_ids), tf.keras.layers.GlobalAveragePooling1D()(embedding)


def create_model_from_pretrained(
        model, max_seq_length_question: int, max_seq_length_answer: int, output_size: int = 30,
        is_distilled: bool = False):
    q_inputs, q_embed = _configure_pretrained_model_block(model, max_seq_length_question, is_distilled=is_distilled)
    a_inputs, a_embed = _configure_pretrained_model_block(model, max_seq_length_answer, is_distilled=is_distilled)

    if is_distilled:
        q_input_ids, q_attention_mask = q_inputs
        a_input_ids, a_attention_mask = a_inputs
        inputs = [q_input_ids, q_attention_mask, a_input_ids, a_attention_mask]
    else:
        q_input_ids, q_attention_mask, q_token_type_ids = q_inputs
        a_input_ids, a_attention_mask, a_token_type_ids = a_inputs
        inputs = [q_input_ids, q_attention_mask, q_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids]

    x = tf.keras.layers.Concatenate()([q_embed, a_embed])
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_size, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    # model.summary()  # debug purpose
    return model


########################################################################################################################
# TF model block END  # TODO: move this TF model block section out from main script


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler


def learning_rate_scheduler(epoch: int, lr: float, max_lr: float = 5e-4, factor: float = .5):
    lr_scheduled = tf.math.minimum(max_lr, lr * tf.math.exp(factor * epoch))
    if epoch > 0:
        print(f"\nNext epoch {epoch + 1}: previous learning rate: {lr:.6f} - scheduled to {lr_scheduled: .6f}")
    return lr_scheduled


def _ckeck_dir_path_exist(working_path: str):
    if os.path.exists(working_path) and not os.path.isfile(working_path):
        return True

    return False


def _mkdir_safe(working_path: str):
    if not _ckeck_dir_path_exist(working_path):
        os.makedirs(working_path)


class ISolver:
    @property
    def fine_tuned_model_weights_file_path_(self) -> str:
        raise NotImplementedError()

    def run(self, data: Dict, fit_params: Optional[Dict] = None, inference_only: bool = False, **kwargs):
        raise NotImplementedError()

    def analyze(self):
        raise NotImplementedError()

    @property
    def test_prediction_(self):
        raise NotImplementedError()

    @property
    def valid_trues_(self):
        raise NotImplementedError()

    @property
    def valid_prediction_(self):
        raise NotImplementedError()

    @property
    def valid_score_(self) -> float:
        raise NotImplementedError()


class MixinTransformerSolver(ISolver):
    def __init__(self, fine_tuned_dir: str, pretrained_dir: str, model_weights_filename: str):

        self.model_weights_filename: str = model_weights_filename  # consider move to configs

        _mkdir_safe(fine_tuned_dir)
        self.fine_tuned_dir_path: str = fine_tuned_dir
        self.pretrained_dir_path: str = pretrained_dir
        if not _ckeck_dir_path_exist(pretrained_dir):
            err_msg = f"pretrained dir path is not exists{pretrained_dir}"
            raise ValueError(err_msg)

        # results
        self.preds_test: Optional[pd.DataFrame] = None
        self.preds_valid: Optional[pd.DataFrame] = None
        self.trues_valid: Optional[pd.DataFrame] = None
        self.valid_score: Optional[float] = None

        self.is_executed: bool = False

    @property
    def fine_tuned_model_weights_file_path_(self) -> str:
        return os.path.join(self.fine_tuned_dir_path, self.model_weights_filename)

    def run(self, data: Dict, fit_params: Optional[Dict] = None, inference_only: bool = False, **kwargs):
        raise NotImplementedError()

    def analyze(self):
        import pdb;
        pdb.set_trace()
        return self

    @property
    def test_prediction_(self):
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        return self.preds_test

    @property
    def valid_trues_(self):
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        if self.trues_valid is None:
            print('no model validation in this run')

        return self.trues_valid

    @property
    def valid_prediction_(self):
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        if self.preds_valid is None:
            print('no model validation in this run')

        return self.preds_valid

    @property
    def valid_score_(self) -> float:
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        if self.valid_score is None:
            print('no model validation while run')

        return self.valid_score


class BaseTransformerTFSolver(MixinTransformerSolver):
    def __init__(self, fine_tuned_dir: str, score_func: Callable, batch_encode_func: Callable, configs: Dict, ):
        super().__init__(
            fine_tuned_dir=fine_tuned_dir, pretrained_dir=configs['pretrained_model_dir'],
            model_weights_filename=configs["model_weights_filename"])

        self.score_func = score_func
        self._batch_encode_func = batch_encode_func

        self.configs_question: Dict = configs['question']
        self.configs_answer: Dict = configs['answer']

        self.is_distilled: bool = configs.get("is_distilled", False)

        self.tokenizer: AutoTokenizer = None
        self.model = None
        self.target_columns: Optional[List[str]] = None

        self.max_seq_length_question = self.configs_question["tokenize"]['max_length']
        self.max_seq_length_answer = self.configs_answer["tokenize"]['max_length']

    def _pipeline_factory(self, load_model_from_fine_tuned: bool = False, output_size: int = None):
        # FIXME: AutoTokenizer, AutoConfig, AutoTFModel has unexpected issue while load from self.fine_tuned_dir_path
        load_from_dir_path: str = self.pretrained_dir_path

        tokenizer = AutoTokenizer.from_pretrained(load_from_dir_path)
        tokenizer.save_pretrained(self.fine_tuned_dir_path)

        if output_size is None:
            raise ValueError("need to specified output size for create model")

        # init a new model for this
        model_configs = AutoConfig.from_pretrained(load_from_dir_path)
        model_configs.output_hidden_states = False  # Set to True to obtain hidden states

        print(f"load pretrained weights for transformer from: {load_from_dir_path}")
        K.clear_session()
        model_block = TFAutoModel.from_pretrained(load_from_dir_path, config=model_configs)
        model_block.save_pretrained(self.fine_tuned_dir_path)
        model = create_model_from_pretrained(
            model_block, max_seq_length_question=self.max_seq_length_question,
            max_seq_length_answer=self.max_seq_length_answer, output_size=output_size, is_distilled=self.is_distilled)

        if load_model_from_fine_tuned:
            print(f"load fine-tuned wieghts from : {self.fine_tuned_model_weights_file_path_}")
            model.load_weights(self.fine_tuned_model_weights_file_path_)

        return tokenizer, model

    def _batch_encode(self, x: pd.DataFrame):
        inputs = self._batch_encode_func(
            x, self.tokenizer, self.configs_question["column"], self.configs_question["tokenize"],
            self.configs_answer["column"], self.configs_answer["tokenize"], is_distilled=self.is_distilled)
        return inputs

    def _run_inference(self, x: pd.DataFrame):
        if self.model is None or self.tokenizer is None:
            self.tokenizer, self.model = self._pipeline_factory(
                load_model_from_fine_tuned=True, output_size=len(self.target_columns))
        else:
            print("inference using current loaded tokenizer and model")
        return pd.DataFrame(self.model.predict(self._batch_encode(x)), index=x.index, columns=self.target_columns)

    def _run_model_fine_tune(self, data: Dict, fit_params: Optional[Dict] = None, **kwargs):
        raise NotImplementedError()

    def _analyze_score_dist(self, data: Dict):
        train_groups = data.get('train_groups', None)

        # validation-test overall diff
        ks_result = self.preds_test.apply(lambda x: ks_2samp(x.values, self.preds_valid[x.name].values), axis=0)
        ks_stats, p_value = list(zip(*(ks_result.tolist())))
        stats_diff = pd.concat([
            self.preds_test.mean().rename('test_mean'), self.preds_valid.mean().rename('valid_mean'),
            (self.preds_test.mean() - self.preds_valid.mean()).rename('mean_diff'),
            self.preds_test.mean().rename('test_std'), self.preds_valid.mean().rename('valid_std'),
            pd.Series(ks_stats, index=self.preds_test.columns).rename('ks_stats'),
            pd.Series(p_value, index=self.preds_test.columns).rename('p_value'), ], axis=1).sort_values('mean_diff')
        print(f'valid-test difference:\n{stats_diff.round(6)}\n')

        # validation performance
        valid_metrics = pd.concat([
            (self.trues_valid - self.preds_valid).mean(axis=0).rename('bias'),
            (self.trues_valid - self.preds_valid).abs().mean(axis=0).rename('mae'),
            ((self.trues_valid - self.preds_valid) / self.trues_valid.mean()).abs().mean(axis=0).rename('mape'),
            self.trues_valid.apply(lambda x: x.corr(self.preds_valid[x.name]), axis=0).rename('corr'),
            ], axis=1).sort_values('corr', ascending=False)
        print(f'validation metrics:\n{valid_metrics.round(6)}\n')

        #
        output_categories_question = data.get('output_categories_question', None)
        output_categories_answer = data.get('output_categories_answer', None)
        if output_categories_question is not None and output_categories_answer is not None:
            valid_score_question = self.score_func(
                self.trues_valid[output_categories_question].values,
                self.preds_valid[output_categories_question].values)
            valid_score_answer = self.score_func(
                self.trues_valid[output_categories_answer].values, self.preds_valid[output_categories_answer].values)
            print(f"valid score on question: {valid_score_question:.3f}, answer: {valid_score_answer:.3f}\n")

        groupby_obj = train_groups.groupby('category')
        group_valid_score = groupby_obj.apply(lambda x: self.score_func(
            self.trues_valid.reindex(index=x.index).values,
            self.preds_valid.reindex(index=x.index).values))
        if output_categories_question is not None and output_categories_answer is not None:
            valid_score_question = groupby_obj.apply(lambda x: self.score_func(
                self.trues_valid.reindex(index=x.index, columns=output_categories_question).values,
                self.preds_valid.reindex(index=x.index, columns=output_categories_question).values))
            valid_score_answer = groupby_obj.apply(lambda x: self.score_func(
                self.trues_valid.reindex(index=x.index, columns=output_categories_answer).values,
                self.preds_valid.reindex(index=x.index, columns=output_categories_answer).values))
            group_valid_score = pd.concat(
                [group_valid_score.rename('overall'), valid_score_question.rename('question'),
                 valid_score_answer.rename('answer')], axis=1)
            print(f"group valid score: \n{group_valid_score}\n")

        stats_dict = {
            'valid_test_stats_diff': stats_diff,
            'valid_metrics': valid_metrics,
            'valid_group_score': group_valid_score,
        }

        file_path = os.path.join(self.fine_tuned_dir_path, 'performance_stats.hdf5')
        with pd.HDFStore(file_path) as store:
            for k, v in stats_dict.items():
                store.put(key=k, value=v)

        return self

    def run(self, data: Dict, fit_params: Optional[Dict] = None, inference_only: bool = False, **kwargs):
        test_x = data.get('test_x', None)
        self.target_columns = data['output_categories']

        if inference_only:
            self.is_executed = True
            self.preds_test = self._run_inference(test_x)
            print(f'test dist:\n{self.preds_test.describe()}')
            return self

        self._run_model_fine_tune(data=data, fit_params=fit_params, **kwargs)
        self.preds_test = self._run_inference(test_x)
        print(f'test dist:\n{self.preds_test.describe()}')
        self.is_executed = True

        self._analyze_score_dist(data)
        for k, v in self.calib_reg_dict.items():
            self.preds_valid[k] = v.predict(self.preds_valid[k])
            self.preds_test[k] = v.predict(self.preds_test[k])
        self._analyze_score_dist(data)
        return self


class BaselineTransformerSolver(BaseTransformerTFSolver):
    def __init__(
            self, fine_tuned_dir: str, score_func: Callable, encode_func: Callable, configs: Dict,
            cv_splitter: Optional = None, ):
        super().__init__(
            fine_tuned_dir=fine_tuned_dir, score_func=score_func, batch_encode_func=encode_func, configs=configs)

        self.cv_splitter = cv_splitter
        self.loss_direction: str = configs.get("loss_direction", 'auto')
        self.eval_metric = 'val_loss'

    def _run_model_fine_tune(self, data: Dict, fit_params: Optional[Dict] = None, **kwargs):
        train_x = data.get('train_x', None)
        train_y = data.get('train_y', None)
        train_groups = data.get('train_groups', None)

        # TODO: run HPO
        # FIXME: for now it only runs single model not cv models
        for fold, (train_idx, valid_idx) in enumerate(self.cv_splitter.split(
                X=train_y, y=train_groups['category'].cat.codes, groups=None), start=1):
            self.tokenizer, model = self._pipeline_factory(
                load_model_from_fine_tuned=False, output_size=len(self.target_columns))

            # TODO: generator and data augmentation
            train_outputs = train_y.iloc[train_idx].values
            train_inputs = self._batch_encode(train_x.iloc[train_idx])

            valid_outputs = train_y.iloc[valid_idx].values
            valid_inputs = self._batch_encode(train_x.iloc[valid_idx])

            # training
            optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6)
            # callbacks
            warmup_lr_scheduler = partial(learning_rate_scheduler, max_lr=1e-4, factor=.25)
            lr_schedule = LearningRateScheduler(warmup_lr_scheduler)
            reduce_lr = ReduceLROnPlateau(
                monitor=self.eval_metric, factor=0.5, patience=2, min_lr=1e-6, model=self.loss_direction)
            early_stopping = CustomMetricEarlyStoppingCallback(
                data=(valid_inputs, valid_outputs), training_data=(train_inputs, train_outputs),
                score_func=self.score_func, patience=5, verbose=1, mode=self.loss_direction, restore_best_weights=True)
            callbacks = [early_stopping, reduce_lr, lr_schedule]  # model_checkpoint: got NotImplementedError with
            model.compile(
                loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_crossentropy', 'mse', 'mae'])

            model.fit(
                train_inputs, train_outputs, validation_data=(valid_inputs, valid_outputs), callbacks=callbacks,
                **fit_params)

            model.save_weights(self.fine_tuned_model_weights_file_path_)

            preds = model.predict(valid_inputs)
            self.valid_score = self.score_func(valid_outputs, preds)
            self.preds_valid = pd.DataFrame(preds, index=train_x.iloc[valid_idx].index, columns=self.target_columns)
            self.trues_valid = train_y.iloc[valid_idx]
            print(f'best validation metric score: {self.valid_score:.3f}')

            from sklearn.isotonic import IsotonicRegression
            self.calib_reg_dict = {
                col: IsotonicRegression(y_min=0., y_max=1.).fit(self.trues_valid[col], self.preds_valid[col]) for col in
                self.target_columns}
            break  # FIXME: for now it only runs single model not cv models
        #
        return self


def batch_encode_sequence(
        df: pd.DataFrame, tokenizer, column_question: str, tokenize_config_question: Dict,
        column_answer: str, tokenize_config_answer: Dict, is_distilled: bool = False):
    tokenized_question = tokenizer.batch_encode_plus(df[column_question], **tokenize_config_question)
    q_input_ids = tokenized_question["input_ids"]  # .numpy()
    q_attention_mask = tokenized_question["attention_mask"]  # .numpy()
    q_token_type_ids = tokenized_question["token_type_ids"]  # .numpy()

    tokenized_answer = tokenizer.batch_encode_plus(df[column_answer], **tokenize_config_answer)
    a_input_ids = tokenized_answer["input_ids"]  # .numpy()
    a_attention_mask = tokenized_answer["attention_mask"]  # .numpy()
    a_token_type_ids = tokenized_answer["token_type_ids"]  # .numpy()

    if is_distilled:
        return q_input_ids, q_attention_mask, a_input_ids, a_attention_mask

    return q_input_ids, q_attention_mask, q_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids


def process_read_dataframe(df: pd.DataFrame):
    bins = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    # group
    group_columns = ['category', 'host_stem']
    df['host_stem'] = df['host'].str.split('.').apply(lambda x: ".".join(x[-2:]))
    df[group_columns] = df[group_columns].astype('category')

    # corpus
    columns = ['question_title', 'question_body', 'answer']
    for col in columns:
        df[f"count_{col}"] = df[col].str.split(' ').apply(lambda x: len(x)).astype(np.int32)

    df["count_question_title_body"] = (df['count_question_title'] + df['count_question_body']).astype(np.int32)
    stats_columns = [f"count_{col}" for col in columns] + ["count_question_title_body"]

    df_stats = df[stats_columns].describe(bins)
    df_stats_split = df.groupby('category')[stats_columns].apply(lambda x: x.describe(bins)).unstack(0).T

    # concat
    df['question_title_body'] = df['question_title'].str.cat(others=df['question_body'], sep=' ')
    columns = columns + ['question_title_body']
    return df[columns], df[group_columns], df_stats, df_stats_split


def read_train_test(data_dir: str, index_name: str, inference_only: bool = False):
    # output_categories
    target_columns = [
        'question_asker_intent_understanding', 'question_body_critical', 'question_conversational',
        'question_expect_short_answer', 'question_fact_seeking', 'question_has_commonly_accepted_answer',
        'question_interestingness_others', 'question_interestingness_self', 'question_multi_intent',
        'question_not_really_a_question', 'question_opinion_seeking', 'question_type_choice', 'question_type_compare',
        'question_type_consequence', 'question_type_definition', 'question_type_entity', 'question_type_instructions',
        'question_type_procedure', 'question_type_reason_explanation', 'question_type_spelling',
        'question_well_written', 'answer_helpful', 'answer_level_of_information', 'answer_plausible',
        'answer_relevance', 'answer_satisfaction', 'answer_type_instructions', 'answer_type_procedure',
        'answer_type_reason_explanation', 'answer_well_written'
    ]
    output_categories_question = list(
        filter(lambda x: x.startswith('question_'), target_columns))
    output_categories_answer = list(filter(lambda x: x.startswith('answer_'), target_columns))
    output_categories = output_categories_question + output_categories_answer

    df_test = pd.read_csv(os.path.join(data_dir, 'test.csv')).set_index(index_name)
    test_x, test_groups, test_stats, test_stats_split = process_read_dataframe(df_test)
    print(f'test shape = {df_test.shape}\n{test_stats}\n')

    data = {
        'test_x': test_x, 'test_groups': test_groups, 'output_categories_question': output_categories_question,
        'output_categories_answer': output_categories_answer, 'output_categories': output_categories
    }
    if inference_only:
        return data

    # training
    df_train = pd.read_csv(os.path.join(data_dir, 'train.csv')).set_index(index_name)

    # labels
    df_train[target_columns] = df_train[target_columns].astype(np.float32)
    train_y = df_train[output_categories]
    train_x, train_groups, train_stats, train_stats_split = process_read_dataframe(df_train)
    print(f'train shape = {df_train.shape}\n{train_stats}\n')
    print(f'Split by category: \n{train_stats_split}\nResorted\n{train_stats_split.swaplevel().sort_index()}\n')
    data.update({
        'train_x': train_x, 'train_y': train_y, 'train_groups': train_groups, 'train_stats': train_stats,
        'train_stats_split': train_stats_split, 'output_categories_question': output_categories_question,
        'output_categories_answer': output_categories_answer}
    )
    return data


def make_submission(preds: np.array, data_dir: str, index_name: str):
    df_sub = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv')).set_index(index_name)
    df_sub[df_sub.columns] = preds[df_sub.columns]
    preds.index.name = index_name
    preds.to_csv('submission.csv', index=True)
    return preds


def main():
    configs_repo = {
        'BaselineBert': {
            "solver_gen": "BaselineTransformerSolver",
            "model_weights_filename": "tf_model_fine-tuned.h5",
            "question": {
                "column": "question_title_body",
                "tokenize": {
                    "add_special_tokens": True,
                    "max_length": 256,
                    "stride": 0,
                    "truncation_strategy": "longest_first",
                    "return_tensors": "tf",
                    "return_input_lengths": False,
                    "return_attention_masks": True,
                },
            },

            "answer": {
                "column": "answer",
                "tokenize": {
                    "add_special_tokens": True,
                    "max_length": 384,
                    "stride": 0,
                    "truncation_strategy": "longest_first",
                    "return_tensors": "tf",
                    "return_input_lengths": False,
                    "return_attention_masks": True,
                },
            },
        },
    }

    fit_params_repo = {
        'distilbert-base-uncased': {
            'batch_size': 16,
            'epochs': 10,
            'verbose': 1,
            # 'callbacks': None,
            'shuffle': True,
            'steps_per_epoch': None,
            'validation_steps': None,
            'validation_freq': 1,
            'max_queue_size': 128,
            'workers': 4,
            'use_multiprocessing': True,
        },
    }

    # set configs to run
    inference_only = False
    fit_params = fit_params_repo['distilbert-base-uncased']
    configs = configs_repo['BaselineBert']
    configs["pretrained_model_dir"] = '../input/hugging-face-pretrained/distilbert-base-uncased'
    configs["is_distilled"] = True

    # data reading setup
    WORKING_DIR = './'
    DATA_DIR = '../input/google-quest-challenge/'
    INDEX_NAME = 'qa_id'

    splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.2)
    data = read_train_test(data_dir=DATA_DIR, index_name=INDEX_NAME, inference_only=inference_only)
    solver = BaselineTransformerSolver(
        fine_tuned_dir=WORKING_DIR, cv_splitter=splitter, score_func=spearmanr_ignore_nan,
        encode_func=batch_encode_sequence, configs=configs)
    solver.run(data, fit_params=fit_params, inference_only=inference_only)
    test_result = solver.test_prediction_

    make_submission(test_result, data_dir=DATA_DIR, index_name=INDEX_NAME)
    return


if "__main__" == __name__:
    print(f"tensorflow version: {tf.__version__}")
    print(f"transformers version: {transformers.__version__}")
    seed_everything()
    np.set_printoptions(suppress=True)
    main()

