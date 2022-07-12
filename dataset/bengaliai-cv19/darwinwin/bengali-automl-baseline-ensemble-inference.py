# I trained three separate models for grapheme_root, consonant_diacritic, and vowel_diacritic.
# The images are encoded into PNG files before feeding to the models.
# This kernel can be run using either GPU or CPU-only mode.
from pathlib import Path

print([x.name for x in Path("../input/").iterdir()])
print([x.name for x in Path("../input/bengali-automl-models/").iterdir()])
print([x.name for x in Path("../input/bengaliai-cv19/").iterdir()])


import io
import glob
import joblib
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

HEIGHT = 137
WIDTH = 236
N_CHANNELS = 1
FLATTEN_IMAGE = [str(x) for x in range(32332)]

# Avoids cudnn initialization problem
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def predict_file(model_path, parquet_file, batch_size):
    print(f"Reading {parquet_file}")
    df = pd.read_parquet(parquet_file)

    models = {}
    for label_type in ("grapheme_root", "consonant_diacritic", "vowel_diacritic"):
        models[label_type+ "_medium"] = tf.saved_model.load(
            str(Path(model_path) / (label_type + "_medium"))
        ).signatures["serving_default"]
        models[label_type] = tf.saved_model.load(
            str(Path(model_path) / label_type)
        ).signatures["serving_default"]

    buffer, preds, ids = [], [], []

    for name in df["image_id"].values:
        ids.extend([
            f'{name}_grapheme_root',
            f'{name}_consonant_diacritic',
            f'{name}_vowel_diacritic'
        ])

    def predict(buffer, preds):
        tmp = []
        for label_type in ("grapheme_root", "consonant_diacritic", "vowel_diacritic"):
            labels = None
            scores = []
            for model_type in ("", "_medium"):
                outputs = models[label_type + model_type](
                    image_bytes=tf.convert_to_tensor(buffer),
                    key=tf.convert_to_tensor(
                        [str(x) for x in range(len(buffer))])
                )
                if labels is None:
                    labels = outputs["labels"][0].numpy()
                else:
                    assert np.array_equal(labels, outputs["labels"][0].numpy())
                scores.append(outputs["scores"].numpy())
            picked = np.argmax(np.mean(scores, axis=0), axis=1)
            tmp.append(np.asarray([int(labels[x]) for x in picked]))
        preds.extend(np.stack(tmp, axis=1).reshape(-1).tolist())

    def encode(arr):
        image = Image.fromarray(
            255 - arr.reshape(HEIGHT, WIDTH),
            mode="L"
        )
        with io.BytesIO() as output:
            image.save(output, format="PNG")
            output.seek(0)
            encoded = output.read()
        return encoded
    # Use n_jobs=2 for GPU and n_jobs=4 for CPU-only mode
    with joblib.Parallel(n_jobs=4) as parallel:
        for i in range(0, df.shape[0], batch_size):
            batch = df.iloc[
                i:i+batch_size
            ][FLATTEN_IMAGE].values.astype(
                "uint8"
            )
            predict(
                parallel(
                    joblib.delayed(encode)(arr)
                    for arr in batch
                ),
                preds
            )
    return ids, preds


def main(
    model_path: str = "../input/bengali-automl-models/",
    parquet_pattern: str = "../input/bengaliai-cv19/test_image_data_*.parquet",
    batch_size: int = 128
):
    predictions: List[float] = []
    ids: List[str] = []
    for parquet_file in glob.glob(parquet_pattern):
        ids_tmp, preds_tmp = predict_file(
            model_path, parquet_file, batch_size
        )
        predictions += preds_tmp
        ids += ids_tmp
    df_sub = pd.DataFrame({
        'row_id': ids,
        'target': predictions
    })
    df_sub.to_csv("submission.csv", index=False)
    
main()
