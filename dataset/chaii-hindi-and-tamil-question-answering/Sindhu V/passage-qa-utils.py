# %% [code]
#NOTE: Transformers used here should be version-agnostic. Let notebooks handle versioning mismatch, etc
import pandas as pd, numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer, default_data_collator
from datasets import Dataset
from hf_qa_utils import *

def generate_output(mnames, df, max_seq_len, doc_stride, \
                    batch_size=128, return_feats=False, pp_cleanup=True, tokenizer_name=None):
    if df.shape[0]==0:
        assert not return_feats, "Can't return tokenized features from empty dataframe"
        return {}, {} 
    args = TrainingArguments(
        f"dummy",
        report_to=["tensorboard"],
        per_device_eval_batch_size=batch_size,
    )        
    
    data_collator = default_data_collator

    ds = Dataset.from_pandas(df)
    tname = tokenizer_name if tokenizer_name is not None else mnames[0]
    tokenizer = AutoTokenizer.from_pretrained(tname)
    tkwargs = {"tokenizer": tokenizer, "max_length": max_seq_len, "doc_stride": doc_stride}
    ds_feats = ds.map(prepare_validation_features, batched=True, remove_columns=ds.column_names, fn_kwargs=tkwargs)
    
    print("#"*50)
    starts, ends = None, None
    for mname in mnames:
        model = AutoModelForQuestionAnswering.from_pretrained(mname)
        print(mname)
        trainer = Trainer(model, args, data_collator=data_collator, tokenizer=tokenizer)
        raw_vals = trainer.predict(ds_feats)
        if starts is None:
            starts, ends = raw_vals.predictions
        else:
            starts += raw_vals.predictions[0]
            ends += raw_vals.predictions[1]
    starts /= len(mnames)
    ends /= len(mnames)

    ds_feats.set_format(type=ds_feats.format["type"], columns=list(ds_feats.features.keys()))

    final_predictions, candidates = postprocess_qa_predictions(ds, ds_feats, (starts, ends), \
                                                               cls_token_id=tokenizer.cls_token_id, n_best_size=5, \
                                                               pp_cleanup=pp_cleanup, return_candidates=True)
    if return_feats:
        return final_predictions, candidates, ds_feats
    return final_predictions, candidates

def get_context_offset(ofs):
    ofs = np.stack([of for of in ofs if of is not None])
    return ofs[:, 0].min(), ofs[:, 1].max()
    
def get_char_offsets_df(ds_feats):
    tdf = ds_feats.to_pandas()
    #Generate offsets for required info only
    tdf["char_offset"] = tdf["offset_mapping"].apply(get_context_offset)
    tdf["min_ix"] = tdf["char_offset"].apply(lambda x: x[0])
    tdf["max_ix"] = tdf["char_offset"].apply(lambda x: x[1])
    tdf = tdf[["example_id", "min_ix", "max_ix"]].copy()
    tdf.rename(columns={"example_id": "id"}, inplace=True)
    tdf.index.name = "feature_index"
    cois = ["id", "feature_index", "min_ix", "max_ix"]
    tdf = tdf.reset_index()[cois]
    return tdf