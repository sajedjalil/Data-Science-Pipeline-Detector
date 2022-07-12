#imports
import riiideducation
import pandas as pd
import os
import gc
import time
import lightgbm as lgb

print("finish import")

# constants
DATA_DIR = "/kaggle/input/intermediate-values"
USER_QUESTION_FEATURE_PATH = os.path.join(DATA_DIR, "user_question_feature.csv")
USER_LECTURE_FEATURE_PATH = os.path.join(DATA_DIR, "user_lecture_feature.csv")
CONTENT_FEATURE_PATH = os.path.join(DATA_DIR, "content_feature.csv")
PART_FEATURE_PATH = os.path.join(DATA_DIR, "part_feature.csv")
TAG_FEATURE_PATH = os.path.join(DATA_DIR, "tag_feature.csv")
MODEL_FILE = os.path.join(DATA_DIR, "model1.txt")
QUESTION_DATA_PATH = "/kaggle/input/riiid-test-answer-prediction/questions.csv"
print("finish constant")

# feature engineering
user_question_feature = pd.read_csv(USER_QUESTION_FEATURE_PATH)
user_lecture_feature = pd.read_csv(USER_LECTURE_FEATURE_PATH)
content_feature = pd.read_csv(CONTENT_FEATURE_PATH)
part_feature = pd.read_csv(PART_FEATURE_PATH)
tag_feature = pd.read_csv(TAG_FEATURE_PATH)
average = 0.6572355454194717
prior_time = 25424
values = {'prior_question_elapsed_time': prior_time, 
          'prior_question_had_explanation': False, 
          'total_user_question': 0, 
          'user_correct_rate': average,
          'total_user_lecture': 0,
          'total_content_question': 0,
          'content_correct_rate': average,
          'total_part_question': 0,
          'part_correct_rate': average
         }
print("finish feature")
model = lgb.Booster(model_file=MODEL_FILE)
print("finish model")

#test
env = riiideducation.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:
    test_df = test_df.join(pd.read_csv(QUESTION_DATA_PATH).set_index('question_id'), on="content_id")
    test_df['tag_count'] = [len(str(x).split(" ")) for x in test_df['tags']]
    test_df = test_df.join(user_question_feature.set_index("user_id"), on = "user_id")
    print(test_df.columns)
    print(user_lecture_feature.columns)
    test_df = test_df.join(user_lecture_feature.set_index("user_id"), on = "user_id")
    test_df = test_df.join(content_feature.set_index('content_id'), on = "content_id")
    test_df = test_df.join(part_feature.set_index("part"), on = "part")
    test_df = test_df.join(tag_feature.set_index("tag_count"), on = "tag_count")
    test_df = test_df.fillna(value=values).astype({'prior_question_had_explanation': 'int8'})
    test_df['answered_correctly'] =  model.predict(test_df[['timestamp','prior_question_elapsed_time','prior_question_had_explanation',\
                                                            'total_user_question', 'user_correct_rate','total_user_lecture', 'total_content_question', \
                                                            'content_correct_rate', 'total_part_question', 'part_correct_rate', 'total_tag_question',\
                                                            'tag_correct_rate']])
    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])
    print("finish 1 testing set")