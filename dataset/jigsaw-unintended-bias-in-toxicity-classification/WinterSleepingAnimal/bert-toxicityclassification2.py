# Thanks for taindow https://www.kaggle.com/taindow/bert-a-fine-tuning-example
# Thanks for EliasCai https://github.com/EliasCai/bert-toxicity-classification
import os
import sys
import collections
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import pandas as pd
import numpy as np
import time
print(os.listdir("../input"))
# print(os.listdir("../input/pretrained-bert-including-scripts/master/bert-master"))
# print(os.listdir("../input/berttoxicityclassification/repository/keyingwu2-c-bert-toxicity-classification-6935b16"))
# print(os.listdir("."))
print(os.getcwd())
sys.path.insert(0, '../input/berttoxicityclassification/repository/keyingwu2-c-bert-toxicity-classification-6935b16')

from run_classifier import *
import modeling
import optimization
import tokenization

train=pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test=pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

train['comment_text'] = train['comment_text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)
test['comment_text'] = test['comment_text'].replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\n',  ' ', regex=True)

# print(train.columns)
train['target'] = np.where(train['target']>=0.5,1,0)
# train = train.sample(frac=0.33)

# train.to_csv('../input/train.tsv', sep='\t', index=False, header=False)
# print(os.listdir("../input"))
# test.to_csv('test.tsv', sep='\t', index=False, header=True)

task_name = 'toxic'
bert_config_file = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_config.json'
vocab_file = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/vocab.txt'
init_checkpoint = '../input/pretrained-bert-including-scripts/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/bert_model.ckpt'
init_checkpoint = '../input/checkpoint2/model.ckpt-20000'
data_dir = '../input/jigsaw-unintended-bias-in-toxicity-classification/'
output_dir = './'
upload_dir = '../input/berttoxicity2/'
do_lower_case = True
max_seq_length = 72
do_train = True
do_eval = True
do_predict = True
train_batch_size = 32
eval_batch_size = 32
predict_batch_size = 32
learning_rate = 2e-5
num_train_epochs = 1.0
warmup_proportion = 0.1
use_tpu = False
master = None
save_checkpoints_steps = 35000 
iterations_per_loop = 100
num_tpu_cores = 8
tpu_cluster_resolver = None

start = time.time()
print("--------------------------------------------------------")
print("Starting training ...")
print("--------------------------------------------------------")

bert_config = modeling.BertConfig.from_json_file(bert_config_file)

processor = ToxicProcessor()
label_list = processor.get_labels('')

tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

tpu_cluster_resolver = None
is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

run_config = tf.contrib.tpu.RunConfig(
  cluster=tpu_cluster_resolver,
  master=master,
  model_dir=output_dir,
  save_checkpoints_steps=save_checkpoints_steps,
  keep_checkpoint_every_n_hours = 1,
  tpu_config=tf.contrib.tpu.TPUConfig(
      iterations_per_loop=iterations_per_loop,
      num_shards=num_tpu_cores,
      per_host_input_for_training=is_per_host))

train_examples = processor.get_train_examples(data_dir)[0]
# num_train_steps = int(len(train_examples) / train_batch_size * num_train_epochs)
num_train_steps = 10000
num_warmup_steps = int(num_train_steps * warmup_proportion)

model_fn = model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_list),
    init_checkpoint=init_checkpoint,
    learning_rate=learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=use_tpu,
    use_one_hot_embeddings=use_tpu)

estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=train_batch_size)

train_file = os.path.join(upload_dir, "train_full.tf_record")

# file_based_convert_examples_to_features(
#     train_examples, label_list, max_seq_length, tokenizer, train_file)

tf.logging.info("***** Running training *****")
tf.logging.info("  Num examples = %d", len(train_examples))
tf.logging.info("  Batch size = %d", train_batch_size)
tf.logging.info("  Num steps = %d", num_train_steps)

train_input_fn = file_based_input_fn_builder(
    input_file=train_file,
    seq_length=max_seq_length,
    is_training=True,
    drop_remainder=True)
print("then start the train function !")
# estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
estimator.train(input_fn=train_input_fn, steps=num_train_steps)
print("after the train function !")
end = time.time()
print("--------------------------------------------------------")
print("Training complete in ", end - start, " seconds")
print("--------------------------------------------------------")

def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

#   name_to_features = {
#       "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
#       "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
#       "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
#       "label_ids": tf.FixedLenFeature([], tf.int64),
#       "is_real_example": tf.FixedLenFeature([], tf.int64),
#   }

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64)
  }
  def _decode_record(record, name_to_features):
      """Decodes a record to a TensorFlow example."""
      example = tf.parse_single_example(record, name_to_features)
      for name in list(example.keys()):
          t = example[name]
          if t.dtype == tf.int64:
              t = tf.to_int32(t)
          example[name] = t
      return example

  def input_fn(params):
      # batch_size = params["batch_size"]
      batch_size = 64
      d = tf.data.TFRecordDataset(input_file)
      if is_training:
          d = d.repeat()
          d = d.shuffle(buffer_size=100)
      d = d.apply(
          tf.contrib.data.map_and_batch(
              lambda record: _decode_record(record, name_to_features),
              batch_size=batch_size,
              drop_remainder=drop_remainder))
      return d

  return input_fn


start = time.time()
print("--------------------------------------------------------")
print("Starting inference ...")
print("--------------------------------------------------------")
predict_examples = processor.get_test_examples(data_dir)[0]
num_actual_predict_examples = len(predict_examples)

predict_file = os.path.join(upload_dir, "predict.tf_record")

# file_based_convert_examples_to_features(predict_examples, label_list,
#                                         max_seq_length, tokenizer,
#                                         predict_file)
tf.logging.info("***** Running prediction*****")
tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                len(predict_examples), num_actual_predict_examples,
                len(predict_examples) - num_actual_predict_examples)
tf.logging.info("  Batch size = %d", predict_batch_size)
predict_drop_remainder = True if use_tpu else False
predict_input_fn = file_based_input_fn_builder(
    input_file=predict_file,
    seq_length=max_seq_length,
    is_training=False,
    drop_remainder=predict_drop_remainder)

result = estimator.predict(input_fn=predict_input_fn)

output_predict_file = os.path.join(output_dir, "test_results.tsv")

with tf.gfile.GFile(output_predict_file, "w") as writer:
    num_written_lines = 0
    tf.logging.info("***** Predict results *****")
    for (i, prediction) in enumerate(result):
        # probabilities = prediction["probabilities"]
        probabilities = prediction
        if i >= num_actual_predict_examples:
            break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1

end = time.time()
print("--------------------------------------------------------")
print("Inference complete in ", end - start, " seconds")
print("--------------------------------------------------------")
sample_submission = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')
predictions = pd.read_csv('./test_results.tsv', header=None, sep='\t')

submission = pd.concat([sample_submission.iloc[:,0], predictions.iloc[:,1]], axis=1)
submission.columns = ['id','prediction']
submission.to_csv('submission.csv', index=False, header=True)
submission.head()