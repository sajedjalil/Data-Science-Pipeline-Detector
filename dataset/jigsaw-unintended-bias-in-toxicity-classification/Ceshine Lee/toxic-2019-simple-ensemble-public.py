# CV 0.9361 custom weight config
import os

os.environ['SEED'] = '11309'

from subprocess import run, PIPE
from pathlib import Path

print([x.name for x in Path("../input/").iterdir()])
print([x.name for x in Path("../input/ceshine-toxic-2019-codebase/").iterdir()])
print([x.name for x in Path("../input/jigsaw-unintended-bias-in-toxicity-classification/").iterdir()])

run(["pip install --no-deps ../input/ceshine-toxic-2019-codebase/delegator/delegator.py-master/."], shell=True, check=True)

import delegator

c = delegator.run('nvidia-smi', block=True)
print(c.out)

c = delegator.run('pip install --no-deps ../input/ceshine-toxic-2019-codebase/helperbot/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/ceshine-toxic-2019-codebase/pytorch-pretrained-bert-master/pytorch-pretrained-BERT-master/.', block=True)
assert c.return_code == 0

c = delegator.run('pip install --no-deps ../input/ceshine-toxic-2019-codebase/toxic/.', block=True)
assert c.return_code == 0

print("Installing apex...")
c = delegator.run('pip install -v --no-deps --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/apex-master/apex-master/apex-master/.', block=True)
assert c.return_code == 0
print("Done.")

import pandas as pd
import numpy as np
from scipy.stats import rankdata

POWER = 3.5
arr_subs = []

def predict(name, tokenizer, max_len, mode, df_ref=None):
    print(f"Predicting...{name}")
    c = delegator.run(
        'python -m toxic.inference_bert --batch-size 128 --model-path ../input/ceshine-toxic-2019-models '
        f'--model-name {name} '
        f'--tokenizer-name {tokenizer} --maxlen {max_len} '
        f'--mode {mode}', block=True)
    if c.return_code != 0:
        print(c.err)
    print(c.out)
    assert c.return_code == 0
    print("Done.")
    df_sub = pd.read_csv("submission.csv").set_index("id")
    df_sub = df_sub ** POWER
    if df_ref is not None:
        df_sub = df_sub.loc[df_ref.index]
    return df_sub
    
df_ref = predict(
    "bert-base-uncased_-1_yuval_220_f0",
    "bert-base-uncased_tokenizer",
    220, "both"
)
arr_subs.append(df_ref.values)

df_tmp = predict(
    "gpt2_-1_yuval_250_f4",
    "gpt2_tokenizer",
    250, "head",
    df_ref
)
arr_subs.append(df_tmp.values)

df_tmp = predict(
    "gpt2_-1_yuval_250_f5",
    "gpt2_tokenizer",
    250, "head",
    df_ref
)
arr_subs.append(df_tmp.values)

df_tmp = predict(
    "bert-base-uncased_-1_yuval_220_f1",
    "bert-base-uncased_tokenizer",
    220, "both",
    df_ref
)
arr_subs.append(df_tmp.values)

df_tmp = predict(
    "bert-base-uncased_-1_yuval_220_f2",
    "bert-base-uncased_tokenizer",
    250, "both",
    df_ref
)
arr_subs.append(df_tmp.values)

df_tmp = predict(
    "bert-base-uncased_-1_yuval_220_f3",
    "bert-base-uncased_tokenizer",
    250, "both",
    df_ref
)
arr_subs.append(df_tmp.values)

df_tmp = predict(
    "bert-base-uncased_-1_yuval_220_f4",
    "bert-base-uncased_tokenizer",
    250, "both",
    df_ref
)
arr_subs.append(df_tmp.values)


def ensemble_predictions(predictions, weights, type_="linear"):
    """Taken from a public kernel.
    
    TODO: finds out which one it was from and gives proper credit.
    """
    assert np.isclose(np.sum(weights), 1.0)
    if type_ == "linear":
        res = np.average(predictions, weights=weights, axis=0)
    elif type_ == "harmonic":
        res = np.average([1 / p for p in predictions], weights=weights, axis=0)
        return 1 / res
    elif type_ == "geometric":
        numerator = np.average(
            [np.log(p) for p in predictions], weights=weights, axis=0
        )
        res = np.exp(numerator / sum(weights))
        return res
    elif type_ == "rank":
        res = np.average([rankdata(p) for p in predictions], weights=weights, axis=0)
        return res / (len(res) + 1)
    return res


df_sub = df_ref.copy()
weights = np.ones(len(arr_subs), dtype=np.float32) / len(arr_subs)
df_sub["prediction"] = ensemble_predictions(
    arr_subs, weights=weights, type_="linear")
df_sub.to_csv("submission.csv")


np.set_printoptions(precision=4)
arr = np.concatenate(arr_subs, axis=1).transpose()
print(arr.shape)
print(np.corrcoef(arr))