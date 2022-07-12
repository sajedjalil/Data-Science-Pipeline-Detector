# Credit goes to Grzegorz Sionkowski for his awesome kernel
# https://www.kaggle.com/sionek/mod-dbscan-x-100-parallel;
# to Heng CherKeng for a python version and all improvements;
# to Konstantin Lopuhin for pulishing the idea in
# https://www.kaggle.com/c/trackml-particle-identification/discussion/57180

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

from sklearn.cluster.dbscan_ import dbscan
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import timeit
import multiprocessing
from multiprocessing import Pool

def find_labels(params):
    hits, dz = params
    a = hits['phi'].values
    z = hits['z'].values
    zr = hits['zr'].values
    aa = a + np.sign(z) * dz * z

    f0 = np.cos(aa)
    f1 = np.sin(aa)
    f2 = zr
    X = StandardScaler().fit_transform(np.column_stack([f0, f1, f2]))

    _, l = dbscan(X, eps=0.0045, min_samples=1, n_jobs=4)
    return l + 1

def add_count(l):
    unique, reverse, count = np.unique(l, return_counts=True, return_inverse=True)
    c = count[reverse]
    c[np.where(l == 0)] = 0
    c[np.where(c > 20)] = 0
    return (l, c)

def do_dbscan_predict(hits):
    start_time = timeit.default_timer()

    hits['r'] = np.sqrt(hits['x'] ** 2 + hits['y'] ** 2)
    hits['zr'] = hits['z'] / hits['r']
    hits['phi'] = np.arctan2(hits['y'], hits['x'])

    params = []
    for i in range(0, 20):
        dz = i * 0.00001
        params.append((hits, dz))
        if i > 0:
             params.append((hits, -dz))
    # Kernel time is limited. So we skip some angles.
    for i in range(20, 60):
        dz = i * 0.00001
        if i % 2 == 0:
            params.append((hits, dz))
        else:
             params.append((hits, -dz))
             
    pool = Pool(processes=4)
    labels_for_all_steps = pool.map(find_labels, params)
    results = [add_count(l) for l in labels_for_all_steps]
    pool.close()

    labels, counts = results[0]
    for i in range(1, len(results)):
        l, c = results[i]
        idx = np.where((c - counts > 0))[0]
        labels[idx] = l[idx] + labels.max()
        counts[idx] = c[idx]

    print('time spent:', timeit.default_timer() - start_time)

    return labels

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits, labels))
    submission = pd.DataFrame(data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission

def run_dbscan():
    data_dir = '../input/train_1'

    event_ids = ['000001000']
    sum = 0
    sum_score = 0
    for i, event_id in enumerate(event_ids):
        hits, cells, particles, truth = load_event(data_dir + '/event' + event_id)
        labels = do_dbscan_predict(hits)
        submission = create_one_event_submission(0, hits['hit_id'].values, labels)
        score = score_event(truth, submission)
        print('[%2d] score : %0.8f' % (i, score))
        sum_score += score
        sum += 1

    print('--------------------------------------')
    print(sum_score / sum)

if __name__ == '__main__':
    print('estimate score by known events')
    run_dbscan()

    path_to_test = "../input/test"
    test_dataset_submissions = []

    create_submission = True  # True for submission
    if create_submission:
        print('process test events')
        for event_id, hits in load_dataset(path_to_test, parts=['hits']):
            print('Event ID: ', event_id)
            labels = do_dbscan_predict(hits)
            # Prepare submission for an event
            one_submission = create_one_event_submission(event_id, hits['hit_id'].values, labels)
            test_dataset_submissions.append(one_submission)

        # Create submission file
        submussion = pd.concat(test_dataset_submissions, axis=0)
        submussion.to_csv('submission_final.csv', index=False)
