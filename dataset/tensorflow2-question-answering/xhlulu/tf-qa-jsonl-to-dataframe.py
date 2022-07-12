"""
# Updates

V12:
    - Set default `n_rows` to be -1, i.e. load all data. This prevents
      submission error due to difference in test set. Thanks @fschilder!
V11:
    - document_text is now a string instead of list
    - Update truncation method
    - Added example_id
"""
import json

import numpy as np
import pandas as pd
from tqdm import tqdm


def jsonl_to_df(file_path, n_rows=-1, load_annotations=True, truncate=True, offset=200):
    """
    Simple utility function to load the .jsonl files for the 
    TF2.0 QA competition. It creates a dataframe of the dataset.
    
    To use, click "File" > "Add utility script", search the name of this 
    notebook, then run:
    
    >>> from tf_qa_jsonl_to_dataframe import jsonl_to_df
    >>> train = jsonl_to_df("/kaggle/...train.jsonl")
    >>> test = jsonl_to_df("/kaggle/...test.jsonl", load_annotations=False)
    
    Parameters:
        * file_path (str): The path to your json_file.
        * n_rows (int): The number of rows you are importing. Set value to -1 if you want to import everything. [Default=-1]
        * load_annotations (bool): Whether to load annotations (for training data) or not (test set does not have
          annotations). [Default=True]
        * truncate (bool): Whether to cut the text before the first answer (long or short) [Default=True]
          and after the last answer (long or short), leaving a space for the offset
        * offset (int): If offset = k, then keep only keep the interval (answer_start - k, answer_end + k) [Default=True]
        
    Returns:
        A Dataframe containing the following columns:
            * document_text (str): The document split by whitespace, possibly truncated
            * question_text (str): the question posed
            * yes_no_answer (str): Could be "YES", "NO", or "NONE"
            * short_answer_start (int): Start index of token, -1 if does not exist
            * short_answer_end (int): End index of token, -1 if does not exist
            * long_answer_start (int): Start index of token, -1 if does not exist
            * long_answer_end (int): End index of token, -1 if does not exist
            * example_id (str): ID representing the string.
    
    Author: @xhlulu
    Source: https://www.kaggle.com/xhlulu/tf-qa-jsonl-to-dataframe
    """
    json_lines = []
    
    with open(file_path) as f:
        for i, line in tqdm(enumerate(f)):
            if i == n_rows:
                break
            
            line = json.loads(line)
            last_token = line['long_answer_candidates'][-1]['end_token']

            out_di = {
                'document_text': line['document_text'],
                'question_text': line['question_text']
            }
            
            if 'example_id' in line:
                out_di['example_id'] = line['example_id']
            
            if load_annotations:
                annot = line['annotations'][0]
                
                out_di['yes_no_answer'] = annot['yes_no_answer']
                out_di['long_answer_start'] = annot['long_answer']['start_token']
                out_di['long_answer_end'] = annot['long_answer']['end_token']

                if len(annot['short_answers']) > 0:
                    out_di['short_answer_start'] = annot['short_answers'][0]['start_token']
                    out_di['short_answer_end'] = annot['short_answers'][0]['end_token']
                else:
                    out_di['short_answer_start'] = -1
                    out_di['short_answer_end'] = -1

                if truncate:
                    if out_di['long_answer_start'] == -1:
                        start_threshold = out_di['short_answer_start'] - offset
                    elif out_di['short_answer_start'] == -1:
                        start_threshold = out_di['long_answer_start'] - offset
                    else:
                        start_threshold = min(out_di['long_answer_start'], out_di['short_answer_start']) - offset
                        
                    start_threshold = max(0, start_threshold)
                    end_threshold = max(out_di['long_answer_end'], out_di['short_answer_end']) + offset + 1
                    
                    out_di['document_text'] = " ".join(
                        out_di['document_text'].split(' ')[start_threshold:end_threshold]
                    )

            json_lines.append(out_di)

    df = pd.DataFrame(json_lines).fillna(-1)
    
    return df


if __name__ == '__main__':
    directory = '/kaggle/input/tensorflow2-question-answering/'
    train = jsonl_to_df(directory + 'simplified-nq-train.jsonl', n_rows=1000)
    test = jsonl_to_df(directory + 'simplified-nq-test.jsonl', load_annotations=False, n_rows=-1)
    
    print(train.shape)
    print(test.shape)
    
    print(train.columns)
    print(test.columns)