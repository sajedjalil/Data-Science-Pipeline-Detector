from vowpalwabbit import pyvw
from collections import defaultdict, deque
from tqdm import tqdm
import riiideducation

env = riiideducation.make_env()

vw = pyvw.vw(quiet=True,                # Turn of VW logging to console
             power_t=0.5,
             loss_function="logistic",
             link="logistic",
             noconstant=True,           # IRT / Elo does not have a constant bias
             binary=False,              # Output probabilities
             classweight="1:0.9",       # More questions answered right than wrong.
             bit_precision=30,          # Avoid hash collisions when creating lots of features
             learning_rate=0.52,
             )


def parse_uir(line):
    e = line.split(',')
    return e[2], e[3], e[4], e[5], e[7], int(e[1])


def response_vw(r):
    # VW needs responses of 1, -1 rather than 1, 0 for right and wrong answers
    return "-1" if r == "0" else r


def categorical_vw(salt, val):
    # make sure numerical categoricals are neither treated as continuous
    # nor hashed to a common value when they overlap with other features in
    # the same namespace
    return salt + ":" + salt + val


def example_vw(user, item, response, importance, last_seen_ms, curr_ms):
    
    if last_seen_ms:
        lag = curr_ms - last_seen_ms
    else:
        lag = 0
        
    
    return " ".join([
        response_vw(response),
        importance,
        "|u",
        categorical_vw("user", user),
        "|i",
        categorical_vw("item", item),
        categorical_vw("seenlast1d", "T") if lag != 0 and lag <= 1000 * 3600 * 24 else "",
        categorical_vw("seenitem7d", "T") if lag > 1000 * 3600 * 24 and lag <= 1000 * 3600 * 24 * 7 else "",
                   
    ])


def predict(df):
    predictions = []
    for user, item, ts in zip(df.user_id.values, df.content_id.values, df.timestamp):
        exp = example_vw(str(user), str(item), "", "", last_ts[user][item], ts)
        predictions.append(vw.predict(exp))
    return predictions


def update(df):
    for user, item, response, ts in zip(df.user_id.values,
                                    df.content_id.values,
                                    df.answered_correctly.values,
                                    df.timestamp):
        importance = "1"
        ex = example_vw(str(user), str(item), str(response), str(importance), last_ts[user][item], ts)
        vw.learn(ex)
        last_ts[user][item] = ts
            
    return 0


train_file = '/kaggle/input/riiid-test-answer-prediction/train.csv'
prev_user = 0
prev_task_id = 0
ex_pending = []
last_ts = defaultdict(lambda: defaultdict(lambda:0))
previous_test_df = None
iter_test = env.iter_test()

i=0
with open(train_file, "r") as fileHandler:
    next(fileHandler)  # skip header
    for line in tqdm(fileHandler, mininterval=20):
        [user, item, type_id, task_id, response, ts] = parse_uir(line.strip())

#         i=i+1
#         if i > 10000:
#             break
        
        if type_id == "1":  # skip lectures
            continue

        importance = "1"  # Every example has same importance weight
        ex = example_vw(user, item, response, importance, last_ts[user][item], ts)

        if user == prev_user and task_id == prev_task_id:
            ex_pending.append([user, item, response, ts, ex])  # Save up examples until the end of the bundle
        else:
            for [usr, itm, res, t_ts, e] in ex_pending:
                vw.learn(e)  # Learn from examples every time we reach a new bundle
                last_ts[user][item] = t_ts
            ex_pending.clear()
            ex_pending.append([user, item, response, ts, ex])

        prev_user = user
        prev_task_id = task_id

for (test_df, sample_prediction_df) in tqdm(iter_test, mininterval=20):

    # Don't learn or evaluate previous responses on the first iteration
    if previous_test_df is not None:

        previous_test_df['answered_correctly'] = eval(test_df["prior_group_answers_correct"].iloc[0])
        previous_test_df = previous_test_df[previous_test_df.content_type_id == 0]
        update(previous_test_df)

    previous_test_df = test_df.copy()
    test_df = test_df[test_df.content_type_id == 0]
    test_df['answered_correctly'] = predict(test_df)
    env.predict(test_df[['row_id', 'answered_correctly']])

vw.finish()
print("End")