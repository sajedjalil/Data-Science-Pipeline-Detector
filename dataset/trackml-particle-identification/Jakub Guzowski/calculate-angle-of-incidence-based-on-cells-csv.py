# inspired by https://www.kaggle.com/asalzburger/pixel-detector-cells
# by Andreas Salzburger

import pandas as pd
import numpy as np
from trackml.dataset import load_event
from math import degrees

print()

# h stands for hits

def add_p_phi_theta_truth(h, truth):
    truth = truth.apply(abs)
    truth['tpr'] = np.sqrt(truth.tpx ** 2 + truth.tpy ** 2)
    truth['p_phi_truth'] = np.arctan2(truth.tpy, truth.tpx)
    truth['p_theta_truth'] = np.arctan2(truth.tpr, truth.tpz)
    return h.merge(truth[['hit_id', 'p_phi_truth', 'p_theta_truth']],
                   on='hit_id', how='left')

def add_cells(h, cells):
    h_clusters = cells.groupby('hit_id') \
                      .agg({'ch0': ['min', 'max'],
                            'ch1': ['min', 'max']}) \
                      .reset_index()
    h_clusters.columns = ['_'.join(column).strip('_')
                          for column in h_clusters.columns.values]
    return h.merge(h_clusters, on='hit_id', how='left')

def add_detectors(h, detectors):
    return h.merge(detectors,
                   on=['volume_id', 'layer_id', 'module_id'], how='left')

def add_p_phi_theta_simplistic(h):
    h['cluster_size_u'] = h.ch0_max - h.ch0_min
    h['cluster_size_v'] = h.ch1_max - h.ch1_min
    # i think in general it should be 0.5 below, 
    # but other values like 0. or 0.2 might give better results
    # h.loc[h.cluster_size_u == 0, 'cluster_size_u'] = 0.2
    # h.loc[h.cluster_size_v == 0, 'cluster_size_v'] = 0.2

    h['pu'] = h.cluster_size_u * h.pitch_u
    h['pv'] = h.cluster_size_v * h.pitch_v
    h['pw'] = 2 * h.module_t
    h['px'] = abs(h.rot_xu * h.pu + h.rot_xv * h.pv + h.rot_xw * h.pw)
    h['py'] = abs(h.rot_yu * h.pu + h.rot_yv * h.pv + h.rot_yw * h.pw)
    h['pz'] = abs(h.rot_zu * h.pu + h.rot_zv * h.pv + h.rot_zw * h.pw)
    h['pr'] = np.sqrt(h.px ** 2 + h.py ** 2)
    h['p_phi'] = np.arctan2(h.py, h.px)
    h['p_theta'] = np.arctan2(h.pr, h.pz)
    
def add_p_phi_theta(h):
    h['cluster_size_u_max'] = h.ch0_max - h.ch0_min + 1
    h['cluster_size_u_min'] = h.cluster_size_u_max - 2
    h.loc[h.cluster_size_u_min < 0, 'cluster_size_u_min'] = 0

    h['cluster_size_v_max'] = h.ch1_max - h.ch1_min + 1
    h['cluster_size_v_min'] = h.cluster_size_v_max - 2
    h.loc[h.cluster_size_v_min < 0, 'cluster_size_v_min'] = 0

    h['pu_max'] = h.cluster_size_u_max * h.pitch_u
    h['pu_min'] = h.cluster_size_u_min * h.pitch_u

    h['pv_max'] = h.cluster_size_v_max * h.pitch_v
    h['pv_min'] = h.cluster_size_v_min * h.pitch_v

    h['pw'] = 2 * h.module_t

    h['angle_u_max'] = np.arctan2(h.pu_max, h.pw)
    h['angle_u_min'] = np.arctan2(h.pu_min, h.pw)
    h['angle_u_avg'] = 0.5 * (h.angle_u_max + h.angle_u_min)
    h['pu'] = h.pw * np.tan(h.angle_u_avg)

    h['angle_v_max'] = np.arctan2(h.pv_max, h.pw)
    h['angle_v_min'] = np.arctan2(h.pv_min, h.pw)
    h['angle_v_avg'] = 0.5 * (h.angle_v_max + h.angle_v_min)
    h['pv'] = h.pw * np.tan(h.angle_v_avg)

    h['px'] = abs(h.rot_xu * h.pu + h.rot_xv * h.pv + h.rot_xw * h.pw)
    h['py'] = abs(h.rot_yu * h.pu + h.rot_yv * h.pv + h.rot_yw * h.pw)
    h['pz'] = abs(h.rot_zu * h.pu + h.rot_zv * h.pv + h.rot_zw * h.pw)
    h['pr'] = np.sqrt(h.px ** 2 + h.py ** 2)
    h['p_phi'] = np.arctan2(h.py, h.px)
    h['p_theta'] = np.arctan2(h.pr, h.pz)

def info_errors(h):
    h['error_p_phi'] = abs(h.p_phi - h.p_phi_truth)
    h['error_p_theta'] = abs(h.p_theta - h.p_theta_truth)

    print(degrees(h.error_p_phi.mean()), degrees(h.error_p_phi.std()))
    print(degrees(h.error_p_theta.mean()), degrees(h.error_p_theta.std()))

def info_angles(h):
    print(degrees(h.p_phi.min()), degrees(h.p_phi.max()))
    print(degrees(h.p_theta.min()), degrees(h.p_theta.max()))
    print(degrees(h.p_phi_truth.min()), degrees(h.p_phi_truth.max()))
    print(degrees(h.p_theta_truth.min()), degrees(h.p_theta_truth.max()))

def run_sample_event(event_path):
    h, cells, _, truth = load_event(event_path)
    detectors = pd.read_csv('./input/detectors.csv')
    
    h = add_cells(h, cells)
    h = add_detectors(h, detectors)
    add_p_phi_theta(h)

    # much lower theta error below
    # h = h.loc[h.volume_id <= 9].copy()

    h = add_p_phi_theta_truth(h, truth)
    info_errors(h)
    
    info_angles(h)

run_sample_event('../input/train_100_events/event000001000')