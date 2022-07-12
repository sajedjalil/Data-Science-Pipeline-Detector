import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import riiideducation
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# This is a framework for RIIID competation

# Entrypoint for training model
# [TODO] Change this function to switch different methods
# Put the actual logic into another function instead. Only call here.
def train_model(test_df):
    return sample_method_always_predict_1(test_df)
    
# A sample method, always return 1 for prediction
def sample_method_always_predict_1(test_df):
    test_df['answered_correctly'] = 1
    prediction_result = test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']]
    return prediction_result

# for viewing all files
def checkInputFiles():
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

# main process
def main():
    env = riiideducation.make_env()

    # Training data is in the competition dataset as 
    # nrows can be changed within the limit of memory usage
    train_df = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv',low_memory=False,
                           nrows=10**5, 
                           dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 'int32', 'content_id': 'int16', 'content_type_id': 'int8',
                                  'task_container_id': 'int16', 'user_answer': 'int8', 'answered_correctly': 'int8', 'prior_question_elapsed_time': 'float32', 
                                 'prior_question_had_explanation': 'boolean',
                                 })
    
    iter_test = env.iter_test()
    batch = 0
    results = []
    for (test_df, sample_prediction_df) in iter_test:
        result = train_model(test_df)
        env.predict(result)
        batch += 1
        
        # print out the process
        print ("Batch "+ str(batch)+" completed.")
        
        # save the result
        results.append(result)
        
    # combine all prediction results into a single pd.dataframe
    output = pd.concat(results)
    
    # save prediction result. file name must be "submission.csv"
    output.to_csv('submission.csv', index=False)

# run run run
main()