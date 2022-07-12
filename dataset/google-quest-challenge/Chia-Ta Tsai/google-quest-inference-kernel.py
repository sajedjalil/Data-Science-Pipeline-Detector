from typing import Callable, Dict, Optional, List, Tuple
import os
import sys
import random

from scipy.stats import spearmanr
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold, StratifiedShuffleSplit

import torch
import tensorflow as tf
import transformers

is_kaggle_server: bool = "kaggle" in os.getcwd().split("/")  # check if in kaggle server


def seed_everything(seed: int = 42):
    # Python/TF Seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.random.set_seed(seed)

    # Torch Seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def spearmanr_ignore_nan(trues: np.array, preds: np.array):
    return np.nanmean(
        [spearmanr(ta, pa).correlation for ta, pa in
         zip(np.transpose(trues), np.transpose(np.nan_to_num(preds)) + 1e-7)])


## model
########################################################################################################################
import tensorflow.keras.backend as K
from transformers import AutoConfig, AutoTokenizer, AutoModel, TFAutoModel


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

    subtracted = tf.keras.layers.Subtract()([q_embed, a_embed])
    x = tf.keras.layers.Concatenate()([q_embed, a_embed, subtracted])
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(output_size, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=x)
    # model.summary()  # debug purpose
    return model


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


from scipy.stats import ks_2samp


def _ckeck_dir_path_exist(working_path: str):
    if os.path.exists(working_path) and not os.path.isfile(working_path):
        return True

    return False


def _mkdir_safe(working_path: str) -> bool:
    if not _ckeck_dir_path_exist(working_path):
        os.makedirs(working_path)
        return True

    return False


class MixinTransformerSolver(ISolver):
    def __init__(
            self, score_func: Callable, fine_tuned_dir: str, pretrained_dir: str, model_weights_filename: str,
            model_stats_filename: str = "model_stats.hdf5"):

        self.score_func: Callable = score_func

        self.model_weights_filename: str = model_weights_filename  # consider move to configs
        self.model_stats_filename: str = model_stats_filename

        _mkdir_safe(fine_tuned_dir)
        self.fine_tuned_dir_path: str = fine_tuned_dir
        self.pretrained_dir_path: str = pretrained_dir
        if not _ckeck_dir_path_exist(pretrained_dir):
            err_msg = f"pretrained dir path is not exists: {pretrained_dir}"
            raise ValueError(err_msg)

        self.target_columns: Optional[List[str]] = None

        # results
        self.preds_test: Optional[pd.DataFrame] = None
        self.preds_valid: Optional[pd.DataFrame] = None
        self.trues_valid: Optional[pd.DataFrame] = None
        self.valid_score: Optional[float] = None

        self.is_executed: bool = False

    def _analyze_score_dist(self, data: Dict):
        train_groups = data.get("train_groups", None)

        # validation-test overall diff
        ks_result = self.preds_test.apply(lambda x: ks_2samp(x.values, self.preds_valid[x.name].values), axis=0)
        ks_stats, p_value = list(zip(*(ks_result.tolist())))
        stats_diff = pd.concat([
            self.preds_test.mean().rename("test_mean"), self.preds_valid.mean().rename("valid_mean"),
            (self.preds_test.mean() - self.preds_valid.mean()).rename("mean_diff"),
            self.preds_test.mean().rename("test_std"), self.preds_valid.mean().rename("valid_std"),
            pd.Series(ks_stats, index=self.preds_test.columns).rename("ks_stats"),
            pd.Series(p_value, index=self.preds_test.columns).rename("p_value"), ], axis=1).sort_values("mean_diff")
        print(f"valid-test difference:\n{stats_diff.round(6)}\n")

        # validation performance
        valid_breakdown_metrics = pd.concat([
            (self.trues_valid - self.preds_valid).mean(axis=0).rename("bias"),
            (self.trues_valid - self.preds_valid).abs().mean(axis=0).rename("mae"),
            ((self.trues_valid - self.preds_valid) / self.trues_valid.mean()).abs().mean(axis=0).rename("mape"),
            self.trues_valid.apply(
                lambda x: x.corr(self.preds_valid[x.name], method="pearson"), axis=0).rename("pearson"),
            self.trues_valid.apply(
                lambda x: x.corr(self.preds_valid[x.name], method="spearman"), axis=0).rename("spearman"),
        ], axis=1).sort_values("spearman", ascending=True)
        print(f"validation breakdown metrics:\n{valid_breakdown_metrics.round(6)}\n")

        valid_overall_metrics = valid_breakdown_metrics.describe()
        print(f"validation overall metrics:\n{valid_overall_metrics.round(6)}\n")

        #
        output_categories_question = data.get("output_categories_question", None)
        output_categories_answer = data.get("output_categories_answer", None)
        if output_categories_question is not None and output_categories_answer is not None:
            y_valid_q = self.trues_valid[output_categories_question]
            p_valid_q = self.preds_valid[output_categories_question]
            valid_score_question = self.score_func(y_valid_q.values, p_valid_q.values)

            y_valid_a = self.trues_valid[output_categories_answer]
            p_valid_a = self.preds_valid[output_categories_answer]
            valid_score_answer = self.score_func(y_valid_a.values, p_valid_a.values)
            print(f"valid score on question: {valid_score_question:.3f}, answer: {valid_score_answer:.3f}\n")

        # analysis by groups
        groupby_obj = train_groups.reindex(index=self.trues_valid.index).groupby("category")
        group_valid_score = groupby_obj.apply(lambda x: self.score_func(
            self.trues_valid.reindex(index=x.index).values, self.preds_valid.reindex(index=x.index).values))

        if output_categories_question is not None and output_categories_answer is not None:
            valid_score_question = groupby_obj.apply(lambda x: self.score_func(
                y_valid_q.reindex(index=x.index).values, y_valid_a.reindex(index=x.index).values)).rename("question")
            valid_score_answer = groupby_obj.apply(lambda x: self.score_func(
                y_valid_a.reindex(index=x.index).values, p_valid_a.reindex(index=x.index).values)).rename("answer")
            group_valid_score = pd.concat([
                group_valid_score.rename("overall"), valid_score_question, valid_score_answer], axis=1)
            print(f"group valid score: \n{group_valid_score}\n")

        stats_dict = {
            "valid_test_stats_diff": stats_diff,
            "valid_breakdown_metrics": valid_breakdown_metrics,
            "valid_overall_metrics": valid_overall_metrics,
            "valid_group_score": group_valid_score,
        }

        file_path = self.fine_tuned_model_stats_file_path_
        with pd.HDFStore(file_path, mode="w") as store:
            for k, v in stats_dict.items():
                store.put(key=k, value=v)

        return self

    @property
    def fine_tuned_model_weights_file_path_(self) -> str:
        return os.path.join(self.fine_tuned_dir_path, self.model_weights_filename)

    @property
    def fine_tuned_model_stats_file_path_(self) -> str:
        return os.path.join(self.fine_tuned_dir_path, self.model_stats_filename)


    def run(self, data: Dict, fit_params: Optional[Dict] = None, inference_only: bool = False, **kwargs):
        test_x = data.get("test_x", None)
        self.target_columns = data["output_categories"]

        if inference_only:
            self.is_executed = True
            self.preds_test = self._run_inference(test_x)
            print(f"test dist:\n{self.preds_test.describe()}")
            return self

        self._run_model_fine_tune(data=data, fit_params=fit_params, **kwargs)
        self.preds_test = self._run_inference(test_x)
        print(f"test dist:\n{self.preds_test.describe()}")
        self.is_executed = True

        self._analyze_score_dist(data)
        return self

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
            print("no model validation in this run")

        return self.trues_valid

    @property
    def valid_prediction_(self):
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        if self.preds_valid is None:
            print("no model validation in this run")

        return self.preds_valid

    @property
    def valid_score_(self) -> float:
        if not self.is_executed:
            raise ValueError("need to run solver before get results")

        if self.valid_score is None:
            print("no model validation while run")

        return self.valid_score

    def _run_inference(self, test_x):
        raise NotImplementedError()

    def _run_model_fine_tune(self, data: Dict, fit_params: Dict, **kwargs):
        raise NotImplementedError()


class BaseTransformerTFSolver(MixinTransformerSolver):
    def __init__(self, fine_tuned_dir: str, score_func: Callable, batch_encode_func: Callable, configs: Dict, ):
        super().__init__(
            score_func=score_func, fine_tuned_dir=fine_tuned_dir, pretrained_dir=configs["pretrained_model_dir"],
            model_weights_filename=configs["model_weights_filename"])

        self._batch_encode_func = batch_encode_func

        self.configs_question: Dict = configs["question"]
        self.configs_answer: Dict = configs["answer"]

        self.is_distilled: bool = configs.get("is_distilled", False)

        self.special_tokens_dict = configs.get("special_tokens_dict", dict())
        self.tokenizer: AutoTokenizer = None
        self.model = None

        self.max_seq_length_question = self.configs_question["tokenize"]["max_length"]
        self.max_seq_length_answer = self.configs_answer["tokenize"]["max_length"]

    def _pipeline_factory(self, load_model_from_fine_tuned: bool = False, output_size: int = None):
        # FIXME: AutoTokenizer, AutoConfig, AutoTFModel has unexpected issue while load from self.fine_tuned_dir_path
        load_from_dir_path: str = self.pretrained_dir_path

        tokenizer = AutoTokenizer.from_pretrained(load_from_dir_path)
        # tokenizer.save_pretrained(self.fine_tuned_dir_path)
        num_added_toks = tokenizer.add_special_tokens(self.special_tokens_dict)
        if len(self.special_tokens_dict) > 0:
            print(f"adding special {num_added_toks} tokens: {self.special_tokens_dict}")

        if output_size is None:
            raise ValueError("need to specified output size for create model")

        # init a new model for this
        model_configs = AutoConfig.from_pretrained(load_from_dir_path)
        model_configs.output_hidden_states = False  # Set to True to obtain hidden states

        print(f"load pretrained weights for transformer from: {load_from_dir_path}")
        K.clear_session()
        model_block = TFAutoModel.from_pretrained(load_from_dir_path, config=model_configs)
        # model_block.resize_token_embeddings(len(tokenizer))  # FIXME: not implemented in TF
        # model_block.save_pretrained(self.fine_tuned_dir_path)
        model = create_model_from_pretrained(
            model_block, max_seq_length_question=self.max_seq_length_question,
            max_seq_length_answer=self.max_seq_length_answer, output_size=output_size, is_distilled=self.is_distilled)

        if load_model_from_fine_tuned:
            print(f"load fine-tuned wieghts from : {self.fine_tuned_model_weights_file_path_}")
            model.load_weights(self.fine_tuned_model_weights_file_path_)

        return tokenizer, model

    def _batch_encode(self, x: pd.DataFrame):
        inputs = self._batch_encode_func(
            x, self.tokenizer, column_question=self.configs_question["column"],
            column_question_pair=self.configs_question.get("column_pair", None),
            tokenize_config_question=self.configs_question["tokenize"], column_answer=self.configs_answer["column"],
            column_answer_pair=self.configs_answer.get("column_pair", None),
            tokenize_config_answer=self.configs_answer["tokenize"], is_distilled=self.is_distilled)
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
########################################################################################################################


def batch_encode_sequence(
        df: pd.DataFrame, tokenizer, column_question: str, column_answer: str,
        column_question_pair: Optional[str] = None, tokenize_config_question: Optional[Dict] = None,
        column_answer_pair: Optional[str] = None, tokenize_config_answer: Optional[Dict] = None,
        is_distilled: bool = False):
    encode_sequence = df[column_question]
    if column_question_pair is not None:
        encode_sequence = zip(df[column_question], df[column_question_pair])
    if tokenize_config_question is None:
        tokenize_config_question = dict()

    tokenized_question = tokenizer.batch_encode_plus(encode_sequence, **tokenize_config_question)
    q_input_ids = tokenized_question["input_ids"].numpy()
    q_attention_mask = tokenized_question["attention_mask"].numpy()
    q_token_type_ids = tokenized_question["token_type_ids"].numpy()

    # fix?
    max_length = tokenize_config_question["max_length"]
    if max_length != q_attention_mask.shape[1]:
        appended_length = max_length - q_attention_mask.shape[1]
        q_attention_mask = np.pad(q_attention_mask, ((0, 0), (0, appended_length)), constant_values=0)

    encode_sequence = df[column_answer]
    if column_answer_pair is not None:
        encode_sequence = zip(df[column_answer], df[column_answer_pair])
    if tokenize_config_answer is None:
        tokenize_config_answer = dict()

    tokenized_answer = tokenizer.batch_encode_plus(encode_sequence, **tokenize_config_answer)
    a_input_ids = tokenized_answer["input_ids"].numpy()
    a_attention_mask = tokenized_answer["attention_mask"].numpy()
    a_token_type_ids = tokenized_answer["token_type_ids"].numpy()

    # fix?
    max_length = tokenize_config_answer["max_length"]
    if max_length != a_attention_mask.shape[1]:
        appended_length = max_length - a_attention_mask.shape[1]
        a_attention_mask = np.pad(a_attention_mask, ((0, 0), (0, appended_length)), constant_values=0)

    # print(q_input_ids.shape, q_attention_mask.shape, a_input_ids.shape, a_attention_mask.shape)
    if is_distilled:
        return q_input_ids, q_attention_mask, a_input_ids, a_attention_mask

    return q_input_ids, q_attention_mask, q_token_type_ids, a_input_ids, a_attention_mask, a_token_type_ids


def process_read_dataframe(df: pd.DataFrame):
    bins = [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    # group
    group_columns = ["category", "host_stem"]
    df["host_stem"] = df["host"].str.split(".").apply(lambda x: ".".join(x[-2:]))
    df[group_columns] = df[group_columns].astype("category")

    # corpus
    columns = ["question_title", "question_body", "answer"]
    for col in columns:
        df[f"count_{col}"] = df[col].str.split(" ").apply(lambda x: len(x)).astype(np.int32)

    df["count_question_title_body"] = (df["count_question_title"] + df["count_question_body"]).astype(np.int32)
    df["count_question_title_body_answer"] = (df["count_question_title_body"] + df["count_answer"]).astype(np.int32)
    stats_columns = [f"count_{col}" for col in columns] + [
        "count_question_title_body", "count_question_title_body_answer"]

    df_stats = df[stats_columns].describe(bins)
    df_stats_split = df.groupby("category")[stats_columns].apply(lambda x: x.describe(bins)).unstack(0).T

    # concat
    # df["question_title_body"] = df["question_title"].str.cat(others=df["question_body"], sep=" ")
    # columns = columns + ["question_title_body"]
    return df[columns], df[group_columns], df_stats, df_stats_split


def read_train_test(data_dir: str, index_name: str, inference_only: bool = False):
    # output_categories
    target_columns = [
        "question_asker_intent_understanding", "question_body_critical", "question_conversational",
        "question_expect_short_answer", "question_fact_seeking", "question_has_commonly_accepted_answer",
        "question_interestingness_others", "question_interestingness_self", "question_multi_intent",
        "question_not_really_a_question", "question_opinion_seeking", "question_type_choice", "question_type_compare",
        "question_type_consequence", "question_type_definition", "question_type_entity", "question_type_instructions",
        "question_type_procedure", "question_type_reason_explanation", "question_type_spelling",
        "question_well_written", "answer_helpful", "answer_level_of_information", "answer_plausible",
        "answer_relevance", "answer_satisfaction", "answer_type_instructions", "answer_type_procedure",
        "answer_type_reason_explanation", "answer_well_written"
    ]
    output_categories_question = list(
        filter(lambda x: x.startswith("question_"), target_columns))
    output_categories_answer = list(filter(lambda x: x.startswith("answer_"), target_columns))
    output_categories = output_categories_question + output_categories_answer

    df_test = pd.read_csv(os.path.join(data_dir, "test.csv")).set_index(index_name)
    test_x, test_groups, test_stats, test_stats_split = process_read_dataframe(df_test)
    print(f"test shape = {df_test.shape}\n{test_stats}\n")

    data = {
        "test_x": test_x, "test_groups": test_groups, "output_categories_question": output_categories_question,
        "output_categories_answer": output_categories_answer, "output_categories": output_categories
    }
    if inference_only:
        return data

    # training
    df_train = pd.read_csv(os.path.join(data_dir, "train.csv")).set_index(index_name)

    # labels
    df_train[target_columns] = df_train[target_columns].astype(np.float32)
    train_y = df_train[output_categories]
    train_x, train_groups, train_stats, train_stats_split = process_read_dataframe(df_train)
    print(f"train shape = {df_train.shape}\n{train_stats}\n")
    print(f"Split by category: \n{train_stats_split}\nResorted\n{train_stats_split.swaplevel().sort_index()}\n")
    data.update({
        "train_x": train_x, "train_y": train_y, "train_groups": train_groups, "train_stats": train_stats,
        "train_stats_split": train_stats_split, "output_categories_question": output_categories_question,
        "output_categories_answer": output_categories_answer}
    )
    return data


def make_submission(preds: np.array, data_dir: str, index_name: str):
    df_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv")).set_index(index_name)
    df_sub[df_sub.columns] = preds[df_sub.columns]
    preds.index.name = index_name
    preds.to_csv("submission.csv", index=True)
    return preds


def main():
    configs_repo = {
        'BaselineBert': {
            "solver_gen": "BaselineTransformerSolver",
            "model_weights_filename": "tf_model_fine-tuned.h5",

            'special_tokens_dict': {},
            "question": {
                "column": "question_title",
                "column_pair": "question_body",
                "tokenize": {
                    "add_special_tokens": True,
                    "max_length": 384,  # 256,
                    "stride": 0,
                    "truncation_strategy": "longest_first",
                    "return_tensors": "tf",
                    "return_input_lengths": False,
                    "return_attention_masks": True,
                    "pad_to_max_length": True,
                },
            },

            "answer": {
                "column": "question_title",
                "column_pair": "answer",
                "tokenize": {
                    "add_special_tokens": True,
                    "max_length": 512,  # 384,
                    "stride": 0,
                    "truncation_strategy": "longest_first",
                    "return_tensors": "tf",
                    "return_input_lengths": False,
                    "return_attention_masks": True,
                    "pad_to_max_length": True,
                },
            },
        },

        'BaselineAlBert': {
            "solver_gen": "BaselineTransformerSolver",
            "model_weights_filename": "tf_model_fine-tuned.h5",

            'special_tokens_dict': {},
            "question": {
                "column": "question_title",
                "column_pair": "question_body",
                "tokenize": {
                    "add_special_tokens": True,
                    "max_length": 256,  # 256,
                    "stride": 0,
                    "truncation_strategy": "longest_first",
                    "return_tensors": "tf",
                    "return_input_lengths": False,
                    "return_attention_masks": True,
                    "pad_to_max_length": True,
                },
            },

            "answer": {
                "column": "question_title",
                "column_pair": "answer",
                "tokenize": {
                    "add_special_tokens": True,
                    "max_length": 384,  # 384,
                    "stride": 0,
                    "truncation_strategy": "longest_first",
                    "return_tensors": "tf",
                    "return_input_lengths": False,
                    "return_attention_masks": True,
                    "pad_to_max_length": True,
                },
            },
        },

        'BaselineRoBerta': {
            "solver_gen": "BaselineTransformerSolver",
            "model_weights_filename": "tf_model_fine-tuned.h5",

            'special_tokens_dict': {},
            "question": {
                "column": "question_title",
                "column_pair": "question_body",
                "tokenize": {
                    "add_special_tokens": True,
                    "max_length": 256,  # 256,
                    "stride": 0,
                    "truncation_strategy": "longest_first",
                    "return_tensors": "tf",
                    "return_input_lengths": False,
                    "return_attention_masks": True,
                    "pad_to_max_length": True,
                },
            },

            "answer": {
                "column": "question_title",
                "column_pair": "answer",
                "tokenize": {
                    "add_special_tokens": True,
                    "max_length": 384,  # 384,
                    "stride": 0,
                    "truncation_strategy": "longest_first",
                    "return_tensors": "tf",
                    "return_input_lengths": False,
                    "return_attention_masks": True,
                    "pad_to_max_length": True,
                },
            },
        },
    }

    # need to
    solver_gen = BaseTransformerTFSolver
    inference_only = True

    # pretrained_model_type: str = os.path.basename(os.path.normpath(pretrained_model_dir))
    fit_params = dict()

    DATA_DIR = '../input/google-quest-challenge/'
    INDEX_NAME = 'qa_id'

    ##
    data = read_train_test(data_dir=DATA_DIR, index_name=INDEX_NAME, inference_only=inference_only)
    inference_working_dir = "../input/google-quest-challenge-weights/"

    if True:
        generated_working_dir = "distilroberta-base_q384_a512"
        pretrained_model_type: str = "distilroberta-base"
        configs = configs_repo["BaselineRoBerta"].copy()
        
        configs["pretrained_model_dir"] = os.path.join("../input/hugging-face-pretrained/", pretrained_model_type)
        if pretrained_model_type.find("distil") >= 0:
            configs["is_distilled"] = True

        WORKING_DIR = os.path.join(inference_working_dir, generated_working_dir)
        splitter = StratifiedShuffleSplit(n_splits=3, test_size=0.2)
        solver = solver_gen(
            fine_tuned_dir=WORKING_DIR, score_func=spearmanr_ignore_nan, batch_encode_func=batch_encode_sequence, configs=configs)
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
