# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 14:19:01 2020

@author: yusufengin.tetik
"""

import numpy as np
import operator
import cv2
import time
import math
import datetime
import matplotlib.pyplot as plt
from matplotlib import colors


cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


def plot_one(task, ax, i, train_or_test, input_or_output):
    try:
        input_matrix = task[train_or_test][i][input_or_output]
    except:
        print('no', train_or_test, ' ', input_or_output, ' to plot!')
        return
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('\n\n\n' + train_or_test + ' ' + input_or_output)


def plot_task(task):
    num_train = len(task['train'])
    fig, axs = plt.subplots(num_train, 2, figsize=(3 * 2, 3 * num_train))
    for i in range(num_train):
        plot_one(task, axs[i, 0], i, 'train', 'input')
        plot_one(task, axs[i, 1], i, 'train', 'output')
    fig.tight_layout()
    # plt.title(training_tasks[task_no])
    # plt.interactive(True)
    # plt.show()

    num_test = len(task['test'])
    fig, axs = plt.subplots(num_test, 2, figsize=(3 * 2, 3 * num_test))
    if num_test == 1:
        plot_one(task, axs[0], 0, 'test', 'input')
        plot_one(task, axs[1], 0, 'test', 'output')
    else:
        for i in range(num_test):
            plot_one(task, axs[i, 0], i, 'test', 'input')
            plot_one(task, axs[i, 1], i, 'test', 'output')

    plt.tight_layout()
    # plt.interactive(True)
    # plt.title(training_tasks[i])
    plt.show()


def plot_matrix(matrix_1):
    fig = plt.figure()
    plt.yticks([x - 0.5 for x in range(1 + len(matrix_1))])
    plt.xticks([x - 0.5 for x in range(1 + len(matrix_1[0]))])
    plt.imshow(matrix_1, cmap=cmap, norm=norm)
    plt.show()


def plot_pair(matrix_1, matrix_2):
    fig = plt.figure()

    plt.subplot(1, 2, 1)

    plt.yticks([x - 0.5 for x in range(1 + len(matrix_1))])
    plt.xticks([x - 0.5 for x in range(1 + len(matrix_1[0]))])
    plt.grid(True, which='both', linewidth=0.5)
    plt.imshow(matrix_1, cmap=cmap, norm=norm)

    plt.subplot(1, 2, 2)
    plt.yticks([x - 0.5 for x in range(1 + len(matrix_2))])
    plt.xticks([x - 0.5 for x in range(1 + len(matrix_2[0]))])
    plt.grid(True, which='both', linewidth=0.5)
    plt.imshow(matrix_2, cmap=cmap, norm=norm)

    plt.show()


def plot_tri(matrix_1, matrix_2, matrix_3, title=''):
    plt.title(title)

    plt.subplot(1, 3, 1)
    plt.yticks([x - 0.5 for x in range(1 + len(matrix_1))])
    plt.xticks([x - 0.5 for x in range(1 + len(matrix_1[0]))])
    plt.grid(True, which='both', linewidth=0.5)
    plt.axis('off')
    plt.imshow(matrix_1, cmap=cmap, norm=norm)

    plt.subplot(1, 3, 2)
    plt.yticks([x - 0.5 for x in range(1 + len(matrix_2))])
    plt.xticks([x - 0.5 for x in range(1 + len(matrix_2[0]))])
    plt.grid(True, which='both', linewidth=0.5)
    plt.axis('off')
    plt.imshow(matrix_2, cmap=cmap, norm=norm)

    plt.subplot(1, 3, 3)
    plt.yticks([x - 0.5 for x in range(1 + len(matrix_3))])
    plt.xticks([x - 0.5 for x in range(1 + len(matrix_3[0]))])
    plt.grid(True, which='both', linewidth=0.5)
    plt.axis('off')
    plt.imshow(matrix_3, cmap=cmap, norm=norm)

    plt.show()


def plot_quad(matrix_1, matrix_2, matrix_3, matrix_4, title=''):
    fig = plt.figure()

    plt.title(title)

    plt.subplot(1, 4, 1)
    plt.xticks([]), plt.yticks([])
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.imshow(matrix_1, cmap=cmap, norm=norm)

    plt.subplot(1, 4, 2)
    plt.xticks([]), plt.yticks([])
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.imshow(matrix_2, cmap=cmap, norm=norm)

    plt.subplot(1, 4, 3)
    plt.xticks([]), plt.yticks([])
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.imshow(matrix_3, cmap=cmap, norm=norm)

    plt.subplot(1, 4, 4)
    plt.xticks([]), plt.yticks([])
    plt.grid(True, which='both', color='lightgrey', linewidth=0.5)
    plt.imshow(matrix_4, cmap=cmap, norm=norm)

    plt.show()


def helper_get_bg_color_single(x):
    def most_frequent(List):
        return max(set(List), key=List.count)

    a = np.asarray(x).flatten()
    b = a.tolist()
    s = set(b)
    if 0 in s:
        return 0
    else:
        return most_frequent(b)


def helper_get_color(a, bg=0):
    e = set(np.array(a).flatten())
    e.discard(bg)
    f = e.pop()
    return f


def helper_get_color_cnt(a):
    e = set(np.array(a).flatten())
    return len(e)


def helper_get_shapes_cc_colors_ignored(a, cc=4):
    try:
        shapes = []
        a = a.astype(np.uint8)
        a0 = a.copy()
        bgc = helper_get_bg_color_single(a)
        # print(a0)
        if bgc != 0:
            a0[a == bgc] = 0
        # print('a0', a0)
        num_labels, b = cv2.connectedComponents(a0, connectivity=cc)
        # print('labels',b)
        for c in range(1, num_labels):
            coords = np.argwhere(b == c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            d = a[x_min:x_max + 1, y_min:y_max + 1]
            shapes.append(d)
            # print('d', d)
        if len(shapes) > 25:
            return []
    except Exception as e:
        # print(e)
        return []

    return shapes


def helper_get_shapes_cc_colors_considered(a, cc=4, coords_req=False):
    try:
        a = a.astype(np.uint8)
        clrs = set(np.array(a).flatten())
        clrs.discard(0)
        clrs = list(clrs)
        shapes = []
        for clr in clrs:
            i = np.zeros(a.shape, dtype=np.uint8)
            i[a == clr] = clr
            num_labels, b = cv2.connectedComponents(i, cc)
            for c in range(1, num_labels):
                coords = np.argwhere(b == c)
                x_min, y_min = coords.min(axis=0)
                x_max, y_max = coords.max(axis=0)
                d = a[x_min:x_max + 1, y_min:y_max + 1]
                if coords_req:
                    shapes.append([d, coords])
                else:
                    shapes.append(d)

        if len(shapes) > 15:
            return []
    except:
        return []

    return shapes


def helper_mats_are_equal(a_mats, b_mats):
    try:
        correct_cnt = 0
        for a, b in zip(a_mats, b_mats):
            if a.tolist() == b.tolist():
                correct_cnt = correct_cnt + 1

        # e = all(a.tolist() == b.tolist() for a, b in zip(a_mats, b_mats))
        if correct_cnt > 1:
            return True
        return False
    except:
        return False


def helper_patterns_are_equal(a_mats, b_mats):
    try:
        correct_cnt = 0
        for a, b in zip(a_mats, b_mats):
            if helper_patterns_are_equal_single(a, b):
                correct_cnt = correct_cnt + 1
        if correct_cnt > 1:
            return True
        return False
    except Exception as e:
        print(e)
        return False
    # au.plot_quad(a, b, c, d)
    return True


def helper_patterns_are_equal_single(a, b):
    SF = 4
    try:
        a1 = a.copy()
        b1 = b.copy()
        a1 = a1.repeat(SF, axis=0).repeat(SF, axis=1).copy()
        b1 = b1.repeat(SF, axis=0).repeat(SF, axis=1).copy()
        c = cv2.Laplacian(a1.astype(np.uint8), cv2.CV_64F, ksize=3)
        d = cv2.Laplacian(b1.astype(np.uint8), cv2.CV_64F, ksize=3)
        if c is None or d is None:
            return False
        c[c != 0] = 1
        d[d != 0] = 1
        # au.plot_quad(a, b, c, d)
        if c.tolist() != d.tolist():
            return False
    except Exception as e:
        # print(e)
        return False
    # au.plot_quad(a, b, c, d)
    return True


def helper_mats_are_equal_single(a, b):
    if a is None or b is None:
        return False
    e = a.tolist() == b.tolist()
    return e


# compare shapes in different rotations
def helper_shapes_are_equal_single(a, b):
    # au.plot_pair(a, b)
    if a.tolist() == b.tolist():
        return True

    if np.rot90(a).tolist() == b.tolist():
        return True

    return False


# shapes are in same colors
def helper_shapes_are_in_same_color_single(a, b):
    # au.plot_pair(a, b)
    if np.unique(np.array(a)).tolist() == np.unique(np.array(b)).tolist():
        return True


def helper_select_most_colorful(panels, pw=None, ph=None):
    max_cc = 0
    max_panel = None
    for panel in panels:
        cc = len(np.unique(panel))
        if cc > max_cc:
            max_cc = cc
            max_panel = panel
    return max_panel


def helper_select_least_colorful(panels, pw=None, ph=None):
    min_cc = 1000
    min_panel = None
    for panel in panels:
        cc = len(np.unique(panel))
        if cc < min_cc:
            min_cc = cc
            min_panel = panel
    return min_panel


def helper_get_bg_color(mats):
    c = []
    for x in mats:
        a = np.asarray(x).flatten()
        b = a.tolist()
        c = c + b
    return max(set(c), key=c.count)


def helper_select_most_colored(panels, pw=None, ph=None):
    max_cc = 0
    max_panel = None
    bgc = helper_get_bg_color(panels)
    for panel in panels:
        cc = np.count_nonzero(panel != bgc)
        if cc > max_cc:
            max_cc = cc
            max_panel = panel
    return max_panel


def helper_select_least_colored(panels, pw=None, ph=None):
    min_cc = 1000
    min_panel = None
    bgc = helper_get_bg_color(panels)
    for panel in panels:
        cc = np.count_nonzero(panel != bgc)
        if cc < min_cc:
            min_cc = cc
            min_panel = panel
    return min_panel



def helper_generate_color_cnt_mat(panels, pw=None, ph=None):
    try:
        a = np.zeros((pw, ph), dtype=int)
        i = 0
        j = 0
        for panel in panels:
            a[i, j] = np.sum(panel != 0)
            j = j + 1
            if j == ph:
                i = i + 1
                j = 0
        return a
    except:
        return None


def helper_get_inner_panels(x):
    objects = helper_get_shapes_cc_colors_ignored(x)
    if objects is None or len(objects) == 0:
        return None
    s = objects[0].shape
    for o in objects:
        if o.shape != s:
            return None
    return [objects, None, None]


def helper_get_panels(x):
    H, W = x.shape
    b = np.array(x)

    rows = []
    ch = 0  # cell height
    for i in range(H):
        row = b[i, :]
        if len(np.unique(row)) != 1:
            rows.append(row)
        elif ch == 0:
            ch = i
    # au.plot_matrix(rows)
    nogrid = np.array(rows)

    cols_to_remove = []
    cw = 0  # cell width
    for j in range(W):
        col = b[:, j]
        if len(np.unique(col)) == 1:
            cols_to_remove.append(j)
            if cw == 0:
                cw = j

    if cw < 2 or ch < 2:
        return None

    cols = []
    cols_to_remove = set(cols_to_remove)
    for j in range(W):
        col = nogrid[:, j]
        if not j in cols_to_remove:
            cols.append(col)

    nogrid = np.array(cols)
    nogrid = np.transpose(nogrid)
    # au.plot_matrix(nogrid)

    H, W = nogrid.shape
    panels = []
    for i in range(0, H, ch):
        for j in range(0, W, cw):
            panel = nogrid[i:i + ch, j:j + cw]
            panels.append(panel)
    pw = W // cw
    ph = H // ch
    return [panels, pw, ph]


def helper_overlap_panels_in_order(panels, pw=None, ph=None):
    try:
        res_panel = panels[0].copy()
        for i in range(1,len(panels)):
            res_panel[res_panel == 0] = panels[i][res_panel == 0]
        # au.plot_matrix(res_panel)
        return res_panel
    except:
        return None


class DslLabel:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a) for a in a_mats]

    def pred(self, a, x=None):
        try:
            a = a.astype(np.uint8)
            num_labels, b = cv2.connectedComponents(a)
            # au.plot_matrix(b)
            if b is not None:
                return b
        except:
            return None
        return None


class DslLabelAfterRecoloring:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a) for a in a_mats]

    def pred(self, a, x=None):
        try:
            def get_bg_color(a):
                b = np.asarray(a).flatten().tolist()
                return max(set(b), key=b.count)

            b = a.astype(np.uint8).copy()
            bgc = get_bg_color(b)
            if bgc != 0:
                b[b==0] = 11
                b[b==bgc] = 0
            num_labels, c = cv2.connectedComponents(b)
            # au.plot_matrix(b)
            return c
        except:
            return None



class DslGetMostFreqShape:
    def learn(self, a_mats, x_mats, y_mats=None):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        try:
            b = helper_get_shapes_cc_colors_considered(a)
            d = {}
            lst = []
            for o in b:
                f = tuple(np.array(o).flatten())
                d[f] = o
                lst.append(f)

            m = max(set(lst), key=lst.count)
            return d[m]
        except Exception as e:
            # print(e)
            return None
        

class DslGetLeastFreqShape:
    def learn(self, a_mats, x_mats, y_mats=None):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        try:
            b = helper_get_shapes_cc_colors_considered(a)
            d = {}
            lst = []
            for o in b:
                f = tuple(np.array(o).flatten())
                d[f] = o
                lst.append(f)

            m = min(set(lst), key=lst.count)
            return d[m]
        except:
            # print("DslGetLeastFreqShape exception")
            return None


class DslGetWidestShape:
    def learn(self, a_mats, x_mats, y_mats=None):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        b = helper_get_shapes_cc_colors_considered(a)
        max_w = 0
        max_o = None
        for o in b:
            d = o.shape[1]
            if d > max_w:
                max_w = d
                max_o = o

        return max_o


class DslGetLongestShape:
    def learn(self, a_mats, x_mats, y_mats=None):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        b = helper_get_shapes_cc_colors_considered(a)
        # print('num of found objects', len(b))
        max_h = 0
        max_o = None
        for o in b:
            d = o.shape[0]
            if d > max_h:
                max_h = d
                max_o = o

        return max_o


class DslGetLongestShapeCC:
    def learn(self, a_mats, x_mats, y_mats=None):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        b = helper_get_shapes_cc_colors_ignored(a)
        # print('num of found objects', len(b))
        max_h = 0
        max_o = None
        for o in b:
            d = o.shape[0]
            if d > max_h:
                max_h = d
                max_o = o
        return max_o


class DslGetMostColoredObject:
    def learn(self, a_mats, x_mats, y_mats=None):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        b = helper_get_shapes_cc_colors_ignored(a)
        max_c = 0
        max_o = None
        for o in b:
            d = len(np.unique(o.tolist()))
            if d > max_c:
                max_c = d
                max_o = o
        # au.plot_matrix(max_o)

        return max_o


class DslRemoveNoise:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, p, x=None):
        def most_frequent(List):
            return max(set(List), key = List.count)

        bgc = helper_get_bg_color_single(p)
        a = np.pad(p, (1, 1), 'constant', constant_values=(bgc, bgc))
        try:
            for i in range(1, a.shape[0]-1):
                for j in range(1, a.shape[1]-1):
                    nbh = a[i-1:i+2, j-1:j+2].flatten()
                    b = nbh.tolist()
                    c = b.count(a[i,j])
                    if c == 1:
                        if b.count(bgc) == 8:  # remove noise on background
                            a[i,j] = bgc
                        else:
                            b = [bi for bi in b if bi != bgc]
                            a[i,j] = most_frequent(b)
            return a[1:-1,1:-1]
        except:
            return None

class DslRemoveNoiseMostFreq:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, p, x=None):
        def most_frequent(List):
            return max(set(List), key = List.count)

        bgc = helper_get_bg_color_single(p)
        a = np.pad(p, (1, 1), 'constant', constant_values=(bgc, bgc))
        try:
            for i in range(1, a.shape[0]-1):
                for j in range(1, a.shape[1]-1):
                    nbh = a[i-1:i+2, j-1:j+2].flatten()
                    b = nbh.tolist()
                    c = b.count(a[i,j])
                    if c == 1 or c == 2:
                        if b.count(bgc) == 8:  # remove noise on background
                            a[i,j] = bgc
                        else:
                            if b[0] == b[1] and b[1] == b[3]:
                                a[i, j] = b[0]
                            elif b[3] == b[6] and b[6] == b[7]:
                                a[i, j] = b[3]
                            elif b[1] == b[2] and b[2] == b[5]:
                                a[i, j] = b[1]
                            elif b[5] == b[7] and b[7] == b[8]:
                                a[i, j] = b[5]
                            else:
                                a[i,j] = most_frequent(b)
            return a[1:-1,1:-1]
        except Exception as e:
            print(e)
            return None


class DslRemoveNoiseBW:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, p, x=None):

        def most_frequent(List):
            return max(set(List), key = List.count)

        bgc = helper_get_bg_color_single(p)
        a = np.pad(p, (1,1), 'constant', constant_values=(bgc, bgc))

        c = helper_get_color_cnt(a)
        if c != 2:
            return None

        try:
            for i in range(1, a.shape[0]-1):
                for j in range(1, a.shape[1]-1):
                    nbh = a[i-1:i+2, j-1:j+2].flatten()
                    b = nbh.tolist()
                    if b[3] == b[5] :
                        a[i, j] = b[3]
                    elif b[1] == b[7]:
                        a[i, j] = b[1]
            return a[1:-1,1:-1]
        except Exception as e:
            print(e)
            return None


class DslGetBiggestShape:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        b = helper_get_shapes_cc_colors_considered(a)
        # print('num of found objects', len(b))
        max_s = 0
        max_o = None
        for o in b:
            d = o.size
            if d > max_s:
                max_s = d
                max_o = o

        return max_o


class DslUpscale:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):
        preds = []
        for p, x, y in zip(p_mats, x_mats, y_mats):
            try:
                a = y.shape[0] / p.shape[0]
                b = y.shape[1] / p.shape[1]
                c = int(a)
                d = int(b)
                if c == a and d == b:
                    if "w_r" not in self.m:
                        self.m["w_r"] = c
                    if "h_r" not in self.m:
                        self.m["h_r"] = d
                preds.append(self.pred(p, x))
            except Exception as e:
                # print(e)
                return None
        return preds

    def pred(self, p, x):
        if "w_r" not in self.m:
            return None
        if p.shape[0] * self.m["w_r"] > 30 or p.shape[1] * self.m["h_r"] > 30:
            return None
        return p.repeat(self.m["w_r"], axis=0).repeat(self.m["h_r"], axis=1)


class DslDownscale:
    m = {}
    dims = {4:2, 9:3, 16:4, 25:5}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):
        preds = []
        for p, x, y in zip(p_mats, x_mats, y_mats):
            try:
                a = p.shape[0]
                b = p.shape[1]
                c = 1  # no downscaling
                d = 1
                if a in self.dims:
                    c = self.dims[a]
                if b in self.dims:
                    d = self.dims[b]

                if a == c and b == d:  # no downscaling
                    return None

                if "w_r" not in self.m:
                    self.m["w_r"] = c
                if "h_r" not in self.m:
                    self.m["h_r"] = d
                preds.append(self.pred(p, x))
            except Exception as e:
                # print(e)
                return None
        return preds

    def pred(self, a, x):
        if "w_r" not in self.m:
            return None
        p = a[::self.m["w_r"], ::self.m["h_r"]].copy()
        return p


class DslDivide:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):
        preds = []
        for p, x, y in zip(p_mats, x_mats, y_mats):
            try:
                a = p.shape[0] / y.shape[0]
                b = p.shape[1] / y.shape[1]
                c = int(a)
                d = int(b)
                # print(" c ve d", c, d)
                if c != 0 and d != 0 and c == a and d == b:
                    if "divide_long_dim_by" not in self.m:
                        self.m['divide_long_dim_by'] = max(c, d)
                    elif self.m['divide_long_dim_by'] != max(c, d):
                        return None
                else:
                    return None
                preds.append(self.pred(p, x))
            except Exception as e:
                # print(e)
                return None
        return preds

    def pred(self, p, x=None):
        if "divide_long_dim_by" not in self.m:
            return None
        if p.shape[0] > p.shape[1]:
            h = int(p.shape[0] / self.m['divide_long_dim_by'])
            return p[0:h, :]
        else:
            w = int(p.shape[1] / self.m['divide_long_dim_by'])
            return p[:, 0:w]


class DslTile:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):
        preds = []
        for p, x, y in zip(p_mats, x_mats, y_mats):
            try:
                a = y.shape[0] / p.shape[0]
                b = y.shape[1] / p.shape[1]
                c = int(a)
                d = int(b)
                if c == a and d == b:
                    if "w_r" not in self.m:
                        self.m["w_r"] = c
                    if "h_r" not in self.m:
                        self.m["h_r"] = d
                preds.append(self.pred(p, x))
            except Exception as e:
                # print(e)
                return None
        return preds

    def pred(self, p, x=None):
        if "w_r" not in self.m:
            return None
        if p.shape[0] * self.m["w_r"] > 30 or p.shape[1] * self.m["h_r"] > 30:
            return None
        return np.tile(p, (self.m["w_r"] , self.m["h_r"]))


class DslDownscaleToObjectSize:
    def learn(self, p_mats, o_mats, x_mats=None, y_mats=None):
        return [self.pred(p, o, x) for p, o, x  in zip(p_mats, o_mats, x_mats)]

    def pred(self, p, o, x=None):
        hr = p.shape[0] // o.shape[0]
        wr = p.shape[1] // o.shape[1]
        if hr < 2 or wr < 2:
            return None
        a = p[::hr,::wr]
        return a


class DslCopyObjectsColors:
    def learn(self, p_mats, o_mats, x_mats, y=None):
        return [self.pred(p, o, x) for p, o, x in zip(p_mats, o_mats, x_mats)]

    def pred(self, p, o, x):
        if p.shape != o.shape:
            return None
        a = p.copy()
        bgc = helper_get_bg_color_single(a)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j] != bgc:
                    a[i, j] = o[i, j]
        return a


class DslXor:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        if any(a.shape != b.shape for a, b in zip(a_mats, b_mats)):
            return None
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape != b.shape:
            return None
        return operator.xor(a, b)


class DslOr:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        if any(a.shape != b.shape for a, b in zip(a_mats, b_mats)):
            return None
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape != b.shape:
            return None
        return a | b


class DslNotaOrb:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        if any(a.shape != b.shape for a, b in zip(a_mats, b_mats)):
            return None
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape != b.shape:
            return None
        nota = np.zeros(a.shape).astype(np.uint8)
        nota[a == 0] = 1
        return nota | b


class DslNotaOrNotb:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        if any(a.shape != b.shape for a, b in zip(a_mats, b_mats)):
            return None
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape != b.shape:
            return None
        nota = np.zeros(a.shape).astype(np.uint8)
        nota[a == 0] = 1
        notb = np.zeros(b.shape).astype(np.uint8)
        notb[b == 0] = 1
        return nota | notb


class DslAnd:
    def learn(self, a_mats, b_mats, x_mats=None, y=None):
        if any(a.shape != b.shape for a, b in zip(a_mats, b_mats)):
            return None
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape != b.shape:
            return None
        return a & b



class DslNotaAndb:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        if any(a.shape != b.shape for a, b in zip(a_mats, b_mats)):
            return None
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape != b.shape:
            return None
        nota = np.zeros(a.shape).astype(np.uint8)
        nota[a == 0] = 1
        return nota & b


class DslaAndNotb:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        if any(a.shape != b.shape for a, b in zip(a_mats, b_mats)):
            return None
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape != b.shape:
            return None
        notb = np.zeros(b.shape).astype(np.uint8)
        notb[b == 0] = 255
        return a & notb


class DslNotaAndNotb:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        if any(a.shape != b.shape for a, b in zip(a_mats, b_mats)):
            return None
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape != b.shape:
            return None
        nota = np.zeros(a.shape).astype(np.uint8)
        notb = np.zeros(b.shape).astype(np.uint8)
        nota[a == 0] = 1
        notb[b == 0] = 1
        return nota & notb


class DslFirst:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        return a


class DslSecond:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        return b


class DslConcatV:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape[1] != b.shape[1]:
            return None
        c = np.concatenate((a, b), axis=0)
        return c


class DslConcatH:
    def learn(self, a_mats, b_mats, x_mats, y=None):
        return [self.pred(a, b, x) for a, b, x in zip(a_mats, b_mats, x_mats)]

    def pred(self, a, b, x=None):
        if a.shape[0] != b.shape[0]:
            return None
        return np.concatenate((a, b), axis=1)


class DslFlipH:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        return np.flip(a, 1)


class DslFlipV:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        return np.flip(a, 0)


class DslRotate270:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        return np.rot90(a, 3)


class DslRotate180:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        return np.rot90(a, 2)


class DslRotate90:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        return np.rot90(a)


class DslClone:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        return a.copy()


class DslNot:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        b = np.unique(a)
        c = len(b)
        d = np.zeros(a.shape, dtype=np.uint8)
        if c == 2:
            d[a == b[0]] = b[1]
            d[a == b[1]] = b[0]
        else:
            d[a != 0] = 0
            d[a == 0] = 1
        return d


class DslBlankCell:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        return np.asmatrix([1])


class DslTranspose:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        return a.transpose()


class DslUpscaleWrtColoredCnt:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        c = np.count_nonzero(np.bincount(np.array(a).flatten() , minlength=10)[1:])
        if a.shape[0] * c > 30 or a.shape[1] * c > 30:
            return None
        return np.repeat(np.repeat(a, c, axis=0), c, axis=1)


class DslTileWrtColoredCnt:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        c = np.count_nonzero(np.bincount(np.array(a).flatten(), minlength=10)[1:])
        if a.shape[0] * c > 30 or a.shape[1] * c > 30:
            return None
        return np.repeat(np.repeat(a, c, axis=0), c, axis=1)


class DslTileWrtSize:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        try:
            c = np.count_nonzero(np.bincount(np.array(a).flatten(), minlength=10)[1:])
            if a.shape[0] * c > 30 or a.shape[1] * c > 30:
                return None
            return np.tile(a, (c, c))
        except:
            return None


class DslUpscaleWrtSize:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        if a.shape[0] * a.shape[0] > 30 or a.shape[1] * a.shape[1] > 30:
            return None
        return a.repeat(a.shape[0], axis=0).repeat(a.shape[1], axis=1)


class DslCropMin:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        try:
            b = np.bincount(np.array(a).flatten(), minlength=10)
            c = int(np.where(b == np.min(b[np.nonzero(b)]))[0])
            coords = np.argwhere(a == c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            return a[x_min:x_max + 1, y_min:y_max + 1]
        except Exception as e:
            # print(e)
            return None


class DslCropMaxCC:
    def learn(self, a_mats, x_mats, y_mats=None):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        try:
            b = helper_get_shapes_cc_colors_ignored(a, 8)
            # print('b',b)
            max_s = 0
            max_o = None
            for o in b:
                d = o.size
                if d > max_s:
                    max_s = d
                    max_o = o
            # print('max_o', max_o)
            return max_o
        except Exception as e:
            print(e)
            return None


class DslCropMinCC:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        try:
            b = helper_get_shapes_cc_colors_ignored(a, 8)
            m_s = 100
            m_o = None
            for o in b:
                d = o.size
                if d < m_s:
                    m_s = d
                    m_o = o
            return m_o
        except Exception as e:
            # print(e)
            return None


class DslCropMax:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        try:
            b = np.bincount(np.array(a).flatten(), minlength=10)
            bgc = helper_get_bg_color_single(a)
            b[bgc] = 255
            c = np.argsort(b)[-2]
            coords = np.argwhere(a == c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            e = a.copy()
            e[e != c] = bgc
            o = e[x_min:x_max + 1, y_min:y_max + 1]
            # au.plot_matrix(o)
            return o
        except Exception as e:
            # print(e)
            return None


class DslSelectMax:
    def learn(self, a_mats, x_mats, y_mats=None):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        try:
            b = np.bincount(np.array(a).flatten(), minlength=10)
            bgc = helper_get_bg_color_single(a)
            b[bgc] = 255
            c = np.argsort(b)[-2]
            coords = np.argwhere(a == c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            d = np.zeros(a.shape, dtype=int)
            d[x_min:x_max + 1, y_min:y_max + 1] = a[x_min:x_max + 1, y_min:y_max + 1]
            return d
        except:
            return None


class DslSelectMin:
    def learn(self, a_mats, x_mats, y_mats=None):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        try:
            b = np.bincount(np.array(a).flatten(), minlength=10)
            c = int(np.where(b == np.min(b[np.nonzero(b)]))[0])
            coords = np.argwhere(a == c)
            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            d = np.zeros(a.shape, dtype=int)
            d[x_min:x_max + 1, y_min:y_max + 1] = a[x_min:x_max + 1, y_min:y_max + 1]
            return d
        except:
            return None


class DslGravity:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):

        ''' find gravity direction '''
        for p, y in zip(p_mats, y_mats):
            h = y.shape[0]
            h2 = h // 2
            s_top = np.sum(y[0:h2, :])
            s_btm = np.sum(y[h2:h, :])
            btm_or_top = s_top
            direction_tb = 'top'
            if s_btm > s_top:
                btm_or_top = s_btm
                direction_tb = 'bottom'
            w = y.shape[1]
            w2 = w // 2
            s_left = np.sum(y[:, 0:w2])
            s_right = np.sum(y[:, w2:w])
            left_or_right = s_left
            direction_lr = 'left'
            if s_right > s_left:
                left_or_right = s_right
                direction_lr = 'right'

            direction = direction_tb
            if left_or_right > btm_or_top:
                direction = direction_lr

            self.m["direction"] = direction

        preds = [self.pred(p, x) for p, x in zip(p_mats, x_mats)]
        return preds

    def pred(self, a, x=None):
        p = None
        try:
            shapes_x = helper_get_shapes_cc_colors_considered(a, coords_req=True)

            # genearate background image
            bgc = helper_get_bg_color_single(a)

            b = np.ones(a.shape, dtype=np.uint8) * bgc

            gravity_direction = self.m['direction']
            if gravity_direction == 'bottom': 
                m_top, m_left = b.shape[0], 0
                m_btm, m_right = b.shape[0], b.shape[1]
            elif gravity_direction == 'top': 
                m_top, m_left = -1, 0
                m_btm, m_right = -1, b.shape[1]
            elif gravity_direction == 'left': 
                m_top, m_left = 0, -1
                m_btm, m_right = b.shape[0], -1
            elif gravity_direction == 'right': 
                m_top, m_left = 0, b.shape[1]
                m_btm, m_right = b.shape[0], b.shape[1]


            # order objects wrt to distance
            ocnt = len(shapes_x)
            ordered_shapes_x = []
            for i in range(ocnt):

                if all(sx is None for sx in shapes_x):
                    break

                min_dist = 99
                min_ind = -1
                j = -1
                for sx in shapes_x:
                    # au.plot_matrix(sx[0])
                    j = j + 1
                    if sx is None:
                        continue
                    c = sx[1]
                    o_top, o_left = c.min(axis=0)
                    o_btm, o_right = c.max(axis=0)
                    distance = -1

                    magnet_below = o_btm < m_top
                    magnet_above = m_btm < o_top
                    magnet_right = o_right < m_left
                    magnet_left = m_right < o_left

                    dist_below = m_top - o_btm
                    dist_above = o_top - m_btm
                    dist_right = m_left - o_right
                    dist_left = o_left - m_right

                    if magnet_below and magnet_right and dist_below == dist_right:
                        distance = math.sqrt(dist_below ** 2 + dist_right ** 2)
                    elif magnet_below and magnet_left and dist_below == dist_left:
                        distance = math.sqrt(dist_below ** 2 + dist_left ** 2)
                    elif magnet_above and magnet_right and dist_above == dist_right:
                        distance = math.sqrt(dist_above ** 2 + dist_right ** 2)
                    elif magnet_above and magnet_left and dist_above == dist_left:
                        distance = math.sqrt(dist_above ** 2 + dist_left ** 2)
                    elif magnet_below:
                        distance = dist_below
                    elif magnet_above:
                        distance = dist_above
                    elif magnet_right:
                        distance = dist_right
                    elif magnet_left:
                        distance = dist_left
                    if distance == -1:
                        # print('return none distance -1')
                        return None
                    if distance < min_dist:
                        min_dist = distance
                        min_ind = j
                        # print(min_ind)

                if min_ind == -1:
                    # print('return none min_ind -1')
                    return None
                ordered_shapes_x.append(shapes_x[min_ind])
                shapes_x[min_ind] = None

            # move objects towards self.m['magnet']
            for sx in ordered_shapes_x:
                # au.plot_matrix(sx[0])
                c = sx[1]
                o_top, o_left = c.min(axis=0)
                o_btm, o_right = c.max(axis=0)
                direction = None

                magnet_below = o_btm < m_top
                magnet_above = m_btm < o_top
                magnet_right = o_right < m_left
                magnet_left = m_right < o_left

                dist_below = m_top - o_btm
                dist_above = o_top - m_btm
                dist_right = m_left - o_right
                dist_left = o_left - m_right

                if magnet_below:
                    direction = [+1, 0]
                    edge = a.shape[0]
                elif magnet_above:
                    direction = [-1, 0]
                    edge = -1
                elif magnet_right:
                    direction = [0, +1]
                    edge = a.shape[1]
                elif magnet_left:
                    direction = [0, -1]
                    edge = -1

                if direction is None:
                    return None

                
                d = b.copy()
                sc2 = c.copy()
                bg = bgc
                Touched = False
                while not Touched:

                    # move shape to one down
                    sc2 = sc2 + direction      

                                           
                    d = b.copy()
                    
                    # check if there is something in the new place
                    if not Touched:
                        for c2 in sc2:
                            if c2[0] == edge or c2[1] == edge:
                                Touched = True
                                break
                            if d[tuple(c2)] != bg:
                                Touched = True
                                break
    

                    if Touched:
                        sc2 = sc2 - direction
                        for c1, c2 in zip(c, sc2):
                            d[tuple(c2)] = a[tuple(c1)]
                    else:
                        for c1, c2 in zip(c, sc2):
                            d[tuple(c2)] = a[tuple(c1)]

                    # au.plot_pair(x, d)
                    p = d.copy()

                b = p.copy()
                
        except Exception as e:
            print(e)
            return None

        return p

class DslMagnetObject:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):

        for p, y in zip(p_mats, y_mats):
            shapes_x = helper_get_shapes_cc_colors_considered(p, coords_req=True)
            shapes_y = helper_get_shapes_cc_colors_considered(y, coords_req=True)

#            if len(shapes_x) != len(shapes_y):
#                return None

            '''check if there is a magnet shape'''
            for sx in shapes_x:
                for sy in shapes_y:
                    if helper_mats_are_equal_single(sx[0], sy[0]):
                        # print("same shape found")
                        if helper_mats_are_equal_single(sx[1], sy[1]):
                            # print('magnet shape found')
                            self.m['magnet'] = sx[0]
                            # au.plot_matrix(sx[0])
                            break

        if 'magnet' not in self.m:
            return None

        preds = [self.pred(p, x) for p, x in zip(p_mats, x_mats)]
        return preds

    def pred(self, a, x=None):
        p = None
        try:
            shapes_x = helper_get_shapes_cc_colors_considered(a, coords_req=True)

            # find magnet object
            mc = None
            for i, sx in enumerate(shapes_x):
                if helper_shapes_are_in_same_color_single(sx[0], self.m['magnet']):
                    mo = sx.copy()
                    mc = mo[1]
                    shapes_x[i] = None
                    break

            if mc is None:
                return None


            # genearate background image
            bgc = helper_get_bg_color_single(a)

            b = np.ones(a.shape, dtype=np.uint8)*bgc

            # copy fixed object
            for c in mc:
                b[tuple(c)] = a[tuple(c)]

            # fixed objects limits
            m_top, m_left = mc.min(axis=0)
            m_btm, m_right = mc.max(axis=0)

            # order objects wrt to distance
            ocnt = len(shapes_x)
            ordered_shapes_x = []
            for i in range(ocnt):
                
                if all(sx is None for sx in shapes_x):
                    break
                
                min_dist = 99
                min_ind = -1
                j = -1
                for sx in shapes_x:
                    # au.plot_matrix(sx[0])
                    j = j + 1
                    if sx is None:
                        continue
                    c = sx[1]
                    o_top, o_left = c.min(axis=0)
                    o_btm, o_right = c.max(axis=0)
                    distance = -1
                    
                    magnet_below = o_btm < m_top
                    magnet_above = m_btm < o_top
                    magnet_right = o_right < m_left
                    magnet_left = m_right < o_left
                    
                    dist_below = m_top - o_btm
                    dist_above = o_top - m_btm
                    dist_right = m_left - o_right
                    dist_left = o_left - m_right
                    
                    if magnet_below and magnet_right and dist_below == dist_right:
                        distance = math.sqrt(dist_below**2+dist_right**2)
                    elif magnet_below and magnet_left and dist_below == dist_left:
                        distance = math.sqrt(dist_below**2+dist_left**2)
                    elif magnet_above and magnet_right and dist_above == dist_right:
                        distance = math.sqrt(dist_above**2+dist_right**2)
                    elif magnet_above and magnet_left  and dist_above == dist_left:
                        distance = math.sqrt(dist_above**2+dist_left**2)
                    elif magnet_below:
                        distance = dist_below
                    elif magnet_above:
                        distance = dist_above
                    elif magnet_right:  
                        distance = dist_right
                    elif magnet_left:
                        distance = dist_left
                    if distance == -1:
                        # print('return none distance -1')
                        return None
                    if distance < min_dist:
                        min_dist = distance
                        min_ind = j
                        # print(min_ind)

                if min_ind == -1:
                    # print('return none min_ind -1')
                    return None
                ordered_shapes_x.append(shapes_x[min_ind])
                shapes_x[min_ind] = None

            # move objects towards self.m['magnet']
            for sx in ordered_shapes_x:
                # au.plot_matrix(sx[0])
                c = sx[1]
                o_top, o_left = c.min(axis=0)
                o_btm, o_right = c.max(axis=0)
                direction = None
               
                magnet_below = o_btm < m_top
                magnet_above = m_btm < o_top
                magnet_right = o_right < m_left
                magnet_left = m_right < o_left
                
                dist_below = m_top - o_btm
                dist_above = o_top - m_btm
                dist_right = m_left - o_right
                dist_left = o_left - m_right
                
                if magnet_below and magnet_right and dist_below == dist_right:
                    direction = [+1, +1]
                    edge = a.shape[0]
                    side = m_top
                elif magnet_below and magnet_left and dist_below == dist_left: 
                    direction = [+1, -1]
                    edge = a.shape[0]
                    side = m_top
                elif magnet_above and magnet_right and dist_above == dist_right:
                    direction = [-1, +1]
                    side = m_btm
                    edge = -1
                elif magnet_above and magnet_left and dist_above == dist_left: 
                    direction = [-1, -1]
                    side = m_btm
                    edge = -1
                elif magnet_below: 
                    direction = [+1, 0]
                    edge = a.shape[0]
                    side = m_top
                elif magnet_above: 
                    direction = [-1, 0]
                    side = m_btm
                    edge = -1
                elif magnet_right: 
                    direction = [0, +1]
                    side = m_left
                    edge = a.shape[1]
                elif magnet_left:
                    direction = [0, -1]
                    side = m_right
                    edge = -1

                if direction is None:
                    return None
                
                # print(direction)

                d = b.copy()
                sc2 = c.copy()
                bg = bgc
                Touched = False
                while not Touched:

                    # move shape to one down
                    sc2 = sc2 + direction

                    # check it if it touches to border edge
                    if side == edge:
                        Touched = True
                        break

                    d = b.copy()
                    # check if there is something in the new place
                    for c2 in sc2:
                        if d[tuple(c2)] != bg:
                            Touched = True
                            break

                    if Touched:
                        sc2 = sc2 - direction
                        for c1, c2 in zip(c, sc2):
                            d[tuple(c2)] = a[tuple(c1)]
                    else:
                        for c1, c2 in zip(c, sc2):
                            d[tuple(c2)] = a[tuple(c1)]

                    # au.plot_pair(x, d)
                    p = d.copy()

                    # update background image
                b = p.copy()
        except Exception as e:
            # print(e)
            return None

        return p


class DslSetColoredBoard:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):
        for p, y in zip(p_mats, y_mats):
            xc = np.unique(np.array(p))
            yc = np.unique(np.array(y))

            if len(xc) - len(yc) != 1:
                return None

            bc = np.setdiff1d(xc, yc)

            self.m["bc"] = bc
            self.m["y_shape"] = y.shape

        preds = [self.pred(p, x) for p, x in zip(p_mats, x_mats)]
        return preds

    def pred(self, a, x=None):
        xc = np.unique(np.array(a))
        bc = self.m["bc"]
        xc = np.setdiff1d(xc, bc)
        cc = len(xc)
        if cc == 0:
            return None
        colors = [None] * cc
        bgc = helper_get_bg_color_single(a)
        H, W = x.shape
        for yy in range(H):
            for xx in range(W):
                color = x[yy, xx]
                if color != bgc:
                    colors[(yy + xx) % cc] = color

        if any(c is None for c in colors):
            return None
        
        p = np.ones(self.m["y_shape"], dtype=np.uint8) * bgc
        H, W = p.shape
        for yy in range(H):
            for xx in range(W):
                p[yy, xx] = colors[(yy + xx) % cc]
        return p


class DslClrToSame:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x=None):
        return a


class DslClrWrtSize:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):
        preds = []
        
        if any(p.shape != y.shape for p, y in zip(p_mats, y_mats)):
            return None
        
        for p, x, y in zip(p_mats, x_mats, y_mats):
            p = np.array(p)
            y = np.array(y)
            unique, counts = np.unique(p, return_counts=True)
            c = counts[1:]
            u = unique[1:]
            e = np.zeros(p.shape)
            for i in range(len(c)):
                g = np.unique(y[p == u[i]])
                if len(g) > 1:
                    return None
                self.m[c[i]] = g
            preds.append(self.pred(p, x))
        return preds

    def pred(self, p, x):
        try:
            p = np.array(p)
            unique, counts = np.unique(p, return_counts=True)
            c = counts[1:]
            u = unique[1:]
            e = np.zeros(p.shape)
            for i in range(len(c)):
                e[p == u[i]] = self.m[c[i]]
            return e
        except:
            return None
        
        
class DslClrWrtSizeOrder:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):
        preds = []
        
        if any(p.shape != y.shape for p, y in zip(p_mats, y_mats)):
            return None
        
        for p, x, y in zip(p_mats, x_mats, y_mats):
            p = np.array(p)
            y = np.array(y)
            unique, counts = np.unique(p, return_counts=True)
            c = counts[1:]
            d = np.argsort(c)
            d = d + 1
            bgc = helper_get_bg_color_single(p)
            e = np.zeros(p.shape, dtype=np.uint8) * bgc
            for i in range(len(d)):
                g = np.unique(y[p == d[i]])
                if len(g) > 1:
                    # au.plot_matrix(p)
                    return None
                self.m[i] = g
                # print(self.m[i])
                e[p == d[i]] = self.m[i]
            preds.append(self.pred(p, x))
        return preds

    def pred(self, p, x):
        try:
            p = np.array(p)
            unique, counts = np.unique(p, return_counts=True)
            c = counts[1:]
            d = np.argsort(c)
            d = d + 1
            bgc = helper_get_bg_color_single(p)
            e = np.zeros(p.shape, dtype=np.uint8) * bgc
            for i in range(len(d)):
                e[p == d[i]] = self.m[i]
            return e
        except:
            return None


class DslClrToMinColor:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        b = DslCropMin().pred(x)
        if b is None:
            return None
        try: 
            c = set(np.array(b).flatten())
            bgc = helper_get_bg_color_single(x)
            c.discard(bgc)
            d = c.pop()
            e = a.copy()
            e[e != bgc] = d
            return e
        except:
            return None


class DslClrToMaxColor:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        b = DslCropMax().pred(x)
        if b is None:
            return None
        try:
            c = set(np.array(b).flatten())
            bgc = helper_get_bg_color_single(x)
            c.discard(bgc)
            d = c.pop()
            e = a.copy()
            e[e != bgc] = d
            return e
        except Exception as e:
            # print("error", e)
            return None

class DslClrToMostFreqColor:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        try:
            bgc = helper_get_bg_color_single(x)
            b = np.asarray(x).flatten().tolist()
            c = max(set(b), key=b.count)
            # d = np.ones(a.shape, dtype=np.uint8) * c
            d = a.copy()
            d[d!=bgc] = c
            return d
        except Exception as e:
            # print("error", e)
            return None

class DslClrToWidestColor:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        b = DslGetWidestShape().pred(x)
        if b is None:
            return None
        try:
            c = set(np.array(b).flatten())
            bgc = helper_get_bg_color_single(x)
            c.discard(bgc)
            d = c.pop()
            e = a.copy()
            e[e != bgc] = d
            return e
        except:
            return None


class DslClrToLongestColor:
    def learn(self, a_mats, x_mats, y_mats):
        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, a, x):
        b = DslGetLongestShape().pred(x)
        if b is None:
            return None
        try:
            c = set(np.array(b).flatten())
            bgc = helper_get_bg_color_single(x)
            c.discard(bgc)
            d = c.pop()
            e = a.copy()
            e[e != bgc] = d
            return e
        except:
            return None


class DslClrToMinShape:
    dsl_memory = {}

    def __init__(self):
        self.dsl_memory = {}

    def learn(self, a_mats, x_mats, y_mats):
        preds = []
        for a, x, y in zip(a_mats, x_mats, y_mats):
            b = DslCropMin().pred(x)
            if b is None:
                return None
            try:
                # c = np.count_nonzero(b)
                c = tuple(np.array(b).flatten())
                d = set(np.array(y).flatten())
                bgc = helper_get_bg_color_single(x)
                d.discard(bgc)
                e = d.pop()
                f = a.copy()
                f[f != bgc] = e
                if c not in self.dsl_memory:  # keep in memory
                    self.dsl_memory[c] = e
                elif self.dsl_memory[c] != e:  # memory conflict
                    return None
                preds.append(f)
            except:
                return None
        return preds

    def pred(self, a, x):
        # print(self.dsl_memory)
        b = DslCropMin().pred(x)
        if b is None:
            return None
        try:
            # c = np.count_nonzero(b)
            c = tuple(np.array(b).flatten())
            d = a.copy()
            bgc = helper_get_bg_color_single(x)
            d[d != bgc] = self.dsl_memory[c]
            return d
        except:
            return None


class DslClrToMaxShape:
    dsl_memory = {}

    def __init__(self):
        self.dsl_memory = {}

    def learn(self, a_mats, x_mats, y_mats):
        preds = []
        for a, x, y in zip(a_mats, x_mats, y_mats):
            b = DslCropMax().pred(x)
            if b is None:
                return None
            try:
                # c = np.count_nonzero(b)
                bgc = helper_get_bg_color_single(x)
                c = tuple(np.array(b).flatten())
                d = set(np.array(y).flatten())
                d.discard(bgc)
                e = d.pop()
                f = a.copy()
                f[f != bgc] = e
                if c not in self.dsl_memory:  # keep in memory
                    self.dsl_memory[c] = e
                elif self.dsl_memory[c] != e:  # memory conflict
                    return None
                preds.append(f)
            except:
                return None
        return preds

    def pred(self, a, x):
        # print(self.dsl_memory)
        b = DslCropMax().pred(x)
        if b is None:
            return None
        try:
            # c = np.count_nonzero(b)
            c = tuple(np.array(b).flatten())
            d = a.copy()
            bgc = helper_get_bg_color_single(x)
            d[d != bgc] = self.dsl_memory[c]
            return d
        except:
            return None
        
        
class DslClrChange:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):

        # if any(y.shape != p.shape for p, y in zip(p_mats, y_mats)):
        #     return None
        #
        # if any(len(np.unique(np.array(p)))!=len(np.unique(np.array(y)))
        #                                        for p, y in zip(p_mats, y_mats)):
        #     return None
    
        for p, y in zip(p_mats, y_mats):
            if y.shape != p.shape:
                continue
            if len(np.unique(np.array(p)))!=len(np.unique(np.array(y))):
                continue
            for i in range(p.shape[0]):
                for j in range(p.shape[1]):
                    xc = p[i,j]
                    yc = y[i,j]
                    if xc not in self.m:
                        self.m[xc] = yc
                    elif self.m[xc] != yc:
                        return None
            
        preds = [self.pred(p, x) for p, x in zip(p_mats, x_mats)]
        return preds

    def pred(self, a, x=None):
        y = np.zeros(a.shape, dtype=np.uint8)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i,j] in self.m:
                    y[i,j] = self.m[a[i,j]]
                # else:
                #     return None
        return y


class DslTileRotate:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):

        if any(y.shape[0] < p.shape[0] for p, y in zip(p_mats, y_mats)):
            return None

        def find_rule(a_mats, b_mats):
            for rule in rules_rotation:
                rule_found = True
                r = rule()
                for a, b in zip(a_mats, b_mats):
                    # if r.pred(a).tolist() != b.tolist():
                    if not helper_patterns_are_equal_single(r.pred(a), b):
                        rule_found = False
                        break
                if rule_found:
                    return r
            return None

        def divide_to_parts(a, hr, wr):
            parts = []

            h = a.shape[0] // hr
            w = a.shape[1] // wr
            for i in range(hr):
                for j in range(wr):
                    part = a[i * h:(i + 1) * h, j * w:(j + 1) * w].copy()
                    parts.append(part)
            return parts

        h_rs = [y.shape[0] // p.shape[0] for p, y in zip(p_mats, y_mats)]
        w_rs = [y.shape[1] // p.shape[1] for p, y in zip(p_mats, y_mats)]

        hr = np.unique(h_rs)
        wr = np.unique(w_rs)

        # print('hr', hr)
        # print('wr', wr)

        if len(hr) == 1 and len(wr) == 1 and wr[0] != 0 and hr[0] != 0:
            self.m['hr'] = hr[0]
            self.m['wr'] = wr[0]
        else:
            return None

        y_mats_parts = [divide_to_parts(y, self.m['hr'], self.m['wr']) for y in y_mats]

        rules = []
        for p in range(self.m['hr'] * self.m['wr']):
            target_parts = [y_mats_parts[i][p] for i in range(len(y_mats))]
            rule = find_rule(p_mats, target_parts)
            if rule is None:
                return None
            rules.append(rule)
        self.m["rules"] = rules

        preds = [self.pred(p, x) for p, x in zip(p_mats, x_mats)]
        return preds

    def pred(self, a, x=None):
        h = a.shape[0]
        w = a.shape[1]
        ans = np.zeros((h * self.m['hr'], w * self.m['wr']), dtype=int)
        counter = 0
        for i in range(self.m['hr']):
            for j in range(self.m['wr']):
                part = self.m["rules"][counter].pred(a)
                ans[i * h:(i + 1) * h, j * w:(j + 1) * w] = part.copy()
                counter = counter + 1
        return ans


class DslSelectPanel:
    m = {}
    selector_rules = [
        helper_select_least_colorful,
        helper_select_most_colorful,
        helper_select_most_colored,
        helper_select_least_colored,
        helper_overlap_panels_in_order,
        helper_generate_color_cnt_mat
    ]

    panel_detection_rules = [
        helper_get_panels,
        helper_get_inner_panels
    ]

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):
        for x, y in zip(p_mats, y_mats):

            for r in self.panel_detection_rules:
                res = r(x)
                if res is not None:
                    self.m["rule_get_panels"] = r
                    break

            if res is None:
                return None

            for r in self.selector_rules:
                p = r(res[0], res[1], res[2])
                # au.plot_pair(p, y)
                if helper_patterns_are_equal_single(p, y):              
                    if 'rule_select' not in self.m:
                        self.m["rule_select"] = r
                    elif self.m['rule_select'] != r:
                        return None

            if not 'rule_select' in self.m:
                return None

        preds = [self.pred(p, x) for p, x in zip(p_mats, x_mats)]
        return preds

    def pred(self, a, x=None):
        res = self.m["rule_get_panels"](a)
        if res is None:
            return None

        p = self.m['rule_select'](res[0], res[1], res[2])
        #  au.plot_matrix(p)

        return p


class DslSelectMostColoredPanel:
    m = {}
    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):
        preds = [self.pred(p, x) for p, x in zip(p_mats, x_mats)]
        return preds

    def pred(self, a, x=None):
        res = helper_get_panels(a)
        if res is None:
            return None

        p = helper_select_most_colored(res[0])

        return p


class DslContextFree:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, a_mats, x_mats, y_mats):

        a_mats_tmp = a_mats.copy()
        x_mats_tmp = x_mats.copy()
        y_mats_tmp = y_mats.copy()

        n = a_mats[0].shape[0]
        k = a_mats[0].shape[1]
        a = y_mats[0].shape[0]
        b = y_mats[0].shape[1]

        if any([x.shape[0] != n or x.shape[1] != k for x in a_mats]):
            return None

        if any([y.shape[0] != a or y.shape[1] != b for y in y_mats]):
            return None

        N = len(a_mats)
        # augment data
        for i in range(N):
            a_mats_tmp.append(np.fliplr(a_mats_tmp[i]))
            y_mats_tmp.append(np.fliplr(y_mats_tmp[i]))
            a_mats_tmp.append(np.flipud(a_mats_tmp[i]))
            y_mats_tmp.append(np.flipud(y_mats_tmp[i]))
            if a == b and n == k:
                a_mats_tmp.append(np.rot90(a_mats_tmp[i]))
                y_mats_tmp.append(np.rot90(y_mats_tmp[i]))
                a_mats_tmp.append(np.rot90(a_mats_tmp[i], 2))
                y_mats_tmp.append(np.rot90(y_mats_tmp[i], 2))
                a_mats_tmp.append(np.rot90(a_mats_tmp[i], 3))
                y_mats_tmp.append(np.rot90(y_mats_tmp[i], 3))

        List1 = {}
        candidates = {}

        for i in range(n):
            for j in range(k):
                seq = []
                for x in a_mats_tmp:
                    seq.append(x[i, j])
                List1[(i, j)] = seq

        for p in range(a):
            for q in range(b):
                seq1 = []
                for y in y_mats_tmp:
                    seq1.append(y[p, q])

                places = []
                for key in List1:
                    if List1[key] == seq1:
                        places.append(key)
#                        if key == (p, q):
#                            print("increase weight of same location")
#                            places.append(key)

                candidates[(p, q)] = places
                if len(places) == 0:
                    return None

        self.m["candidates"] = candidates
        self.m["out_size"] = (a, b)

        return [self.pred(a, x) for a, x in zip(a_mats, x_mats)]

    def pred(self, t, x):
        try:
            answer = np.zeros(self.m["out_size"], dtype=int)
            for p in range(self.m["out_size"][0]):
                for q in range(self.m["out_size"][1]):
                    palette = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    for i, j in self.m['candidates'][(p, q)]:
                        color = t[i, j]
                        palette[color] += 1
                    answer[p, q] = np.argmax(palette)
            return answer
        except:
            return None


class DslTileAndUpscale:
    m = {}

    def __init__(self):
        self.m = {}

    def learn(self, p_mats, x_mats, y_mats):

        r1 = DslTile()
        p1 = r1.learn(p_mats, x_mats, y_mats)

        if p1 is None or any(p is None for p in p1):
            return None

        r2 = DslUpscale()
        p2 = r2.learn(p_mats, x_mats, y_mats)

        if p2 is None or any(p is None for p in p2):
            return None

        for R3 in rules_match:
            r3 = R3()
            p3 = r3.learn(p1, p2, x_mats, y_mats)

            if p3 is None or any(p is None for p in p3):
                return None

            if helper_mats_are_equal(y_mats, p3):
                self.m['rules'] = [r1, r2, r3]
                break

        if 'rules' not in self.m:
            return None

        preds = [self.pred(p, x) for p, x in zip(p_mats, x_mats)]
        return preds

    def pred(self, a, x=None):

        if 'rules' not in self.m:
            return None

        r1, r2, r3 = self.m['rules']

        p1 = r1.pred(a, x)
        if p1 is None:
            return None

        p2 = r2.pred(a, x)
        if p2 is None:
            return None

        p3 = r3.pred(p1, p2, x)

        return p3


''' https://www.kaggle.com/szabo7zoltan/colorandcountingmoduloq '''
class Recolor:

    def predict(self, task, task_id=0):

        def defensive_copy(A):
            n = len(A)
            k = len(A[0])
            L = np.zeros((n, k), dtype=int)
            for i in range(n):
                for j in range(k):
                    L[i, j] = 0 + A[i][j]
            return L.tolist()

        def prepare_task_data(task, test_no=0):
            n = len(task['train'])
            Input = [defensive_copy(task['train'][i]['input']) for i in range(n)]
            Output = [defensive_copy(task['train'][i]['output']) for i in range(n)]
            Input.append(defensive_copy(task['test'][test_no]['input']))
            return Input, Output

        task_data = prepare_task_data(task, task_id)

        Input = task_data[0]
        Output = task_data[1]
        Test_Picture = Input[-1]
        Input = Input[:-1]
        N = len(Input)

        for x, y in zip(Input, Output):
            if len(x) != len(y) or len(x[0]) != len(y[0]):
                return None

        Best_Dict = -1
        Best_Q1 = -1
        Best_Q2 = -1
        Best_v = -1
        Pairs = []

        for Q1 in range(1, 9):
            for Q2 in range(1, 9):
                Pairs.append((Q1, Q2))

        for Q1, Q2 in Pairs:
            for v in range(4):
                if Best_Dict != -1:
                    continue
                possible = True
                Dict = {}

                for x, y in zip(Input, Output):
                    n = len(x)
                    k = len(x[0])
                    for i in range(n):
                        for j in range(k):
                            if v == 0 or v == 2:
                                p1 = i % Q1
                            else:
                                p1 = (n - 1 - i) % Q1
                            if v == 0 or v == 3:
                                p2 = j % Q2
                            else:
                                p2 = (k - 1 - j) % Q2
                            color1 = x[i][j]
                            color2 = y[i][j]
                            if color1 != color2:
                                rule = (p1, p2, color1)
                                if rule not in Dict:
                                    Dict[rule] = color2
                                elif Dict[rule] != color2:
                                    possible = False
                if possible:

                    # Let's see if we actually solve the problem
                    for x, y in zip(Input, Output):
                        n = len(x)
                        k = len(x[0])
                        for i in range(n):
                            for j in range(k):
                                if v == 0 or v == 2:
                                    p1 = i % Q1
                                else:
                                    p1 = (n - 1 - i) % Q1
                                if v == 0 or v == 3:
                                    p2 = j % Q2
                                else:
                                    p2 = (k - 1 - j) % Q2

                                color1 = x[i][j]
                                rule = (p1, p2, color1)

                                if rule in Dict:
                                    color2 = 0 + Dict[rule]
                                else:
                                    color2 = 0 + y[i][j]
                                if color2 != y[i][j]:
                                    possible = False
                    if possible:
                        Best_Dict = Dict
                        Best_Q1 = Q1
                        Best_Q2 = Q2
                        Best_v = v

        if Best_Dict == -1:
            return None  # meaning that we didn't find a rule that works for the traning cases

        # Otherwise there is a rule: so let's use it:
        n = len(Test_Picture)
        k = len(Test_Picture[0])
        answer = np.zeros((n, k), dtype=int)

        for i in range(n):
            for j in range(k):
                if Best_v == 0 or Best_v == 2:
                    p1 = i % Best_Q1
                else:
                    p1 = (n - 1 - i) % Best_Q1
                if Best_v == 0 or Best_v == 3:
                    p2 = j % Best_Q2
                else:
                    p2 = (k - 1 - j) % Best_Q2

                color1 = Test_Picture[i][j]
                rule = (p1, p2, color1)
                if (p1, p2, color1) in Best_Dict:
                    answer[i][j] = 0 + Best_Dict[rule]
                else:
                    answer[i][j] = 0 + color1

        return answer.tolist()


''' https://www.kaggle.com/meaninglesslives/stacking-models-and-new-features-for-arc '''

from xgboost import XGBClassifier
from sklearn.ensemble import (ExtraTreesClassifier,
                              BaggingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)


class MLStack:

    def learn(self, task, test_id=0):
        feats, targets = self.get_features(task)
        if feats is None:
            return None
        estimators = [
            ('xgb', XGBClassifier(n_estimators=25, n_jobs=-1, random_state=0)),
            ('extra_trees', ExtraTreesClassifier(random_state=0)),
            ('bagging', BaggingClassifier(random_state=0)),
            ('LogisticRegression', LogisticRegression(random_state=0))
        ]
        self.clf = StackingClassifier(
            estimators=estimators, final_estimator=XGBClassifier(n_estimators=15, n_jobs=-1, random_state=0)
        )
        self.clf.fit(feats, targets)

        n = len(task['train'])
        x_mats = [np.array(task['train'][i]['input']).astype(np.uint8) for i in range(n)]
        return [self.pred(x) for x in x_mats]

    def pred(self, x):
        return self.predict_output_matrix(np.array(x), self.clf)


    def get_moore_neighbours(self, color, cur_row, cur_col, nrows, ncols):
        # pdb.set_trace()

        if (cur_row <= 0) or (cur_col > ncols - 1):
            top = -1
        else:
            top = color[cur_row - 1][cur_col]

        if (cur_row >= nrows - 1) or (cur_col > ncols - 1):
            bottom = -1
        else:
            bottom = color[cur_row + 1][cur_col]

        if (cur_col <= 0) or (cur_row > nrows - 1):
            left = -1
        else:
            left = color[cur_row][cur_col - 1]

        if (cur_col >= ncols - 1) or (cur_row > nrows - 1):
            right = -1
        else:
            right = color[cur_row][cur_col + 1]

        return top, bottom, left, right

    def get_tl_tr(self, color, cur_row, cur_col, nrows, ncols):
        if cur_row == 0:
            top_left = -1
            top_right = -1
        else:
            if cur_col == 0:
                top_left = -1
            else:
                top_left = color[cur_row - 1][cur_col - 1]
            if cur_col == ncols - 1:
                top_right = -1
            else:
                top_right = color[cur_row - 1][cur_col + 1]
        return top_left, top_right

    def get_vonN_neighbours(self, color, cur_row, cur_col, nrows, ncols):
        if cur_row == 0:
            top_left = -1
            top_right = -1
        else:
            if cur_col == 0:
                top_left = -1
            else:
                top_left = color[cur_row - 1][cur_col - 1]
            if cur_col == ncols - 1:
                top_right = -1
            else:
                top_right = color[cur_row - 1][cur_col + 1]
        if cur_row == nrows - 1:
            bottom_left = -1
            bottom_right = -1
        else:
            if cur_col == 0:
                bottom_left = -1
            else:
                bottom_left = color[cur_row + 1][cur_col - 1]
            if cur_col == ncols - 1:
                bottom_right = -1
            else:
                bottom_right = color[cur_row + 1][cur_col + 1]
        return top_left, top_right, bottom_left, bottom_right


    def inp2img(self, inp):
        img = np.full((11, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
        for i in range(10):
            img[i] = (inp == i)
        img[10] = inp
        return img


    def make_features(self, input_color):
        nfeat = 34
        local_neighb = 5

        # inputs = self.inp2img(input_color)
        inputs = [input_color]

        input_cnt = len(inputs)
        nrows, ncols = input_color.shape
        feat = np.zeros((nrows * ncols, nfeat * input_cnt))

        cur_idx = 0
        for i in range(nrows):
            for j in range(ncols):
                cur_inp = 0
                for input_color in inputs:
                    cur_pos = cur_inp * nfeat
                    feat[cur_idx, cur_pos + 0] = i
                    feat[cur_idx, cur_pos + 1] = j
                    feat[cur_idx, cur_pos + 2] = input_color[i][j]
                    feat[cur_idx, cur_pos + 3:cur_pos + 7] = self.get_moore_neighbours(input_color, i, j, nrows, ncols)
                    try:
                        feat[cur_idx, cur_pos + 7] = len(np.unique(input_color[i - 1, :]))
                        feat[cur_idx, cur_pos + 8] = len(np.unique(input_color[:, j - 1]))
                    except IndexError:
                        pass

                    feat[cur_idx, cur_pos + 9] = len(np.unique(input_color[i, :]))
                    feat[cur_idx, cur_pos + 10] = len(np.unique(input_color[:, j]))
                    feat[cur_idx, cur_pos + 11] = len(np.unique(input_color[i - local_neighb:i + local_neighb,
                                                                j - local_neighb:j + local_neighb]))

                    feat[cur_idx, cur_pos + 12:cur_pos + 16] = self.get_moore_neighbours(input_color, i + 1, j, nrows,
                                                                                         ncols)
                    feat[cur_idx, cur_pos + 16:cur_pos + 20] = self.get_moore_neighbours(input_color, i - 1, j, nrows,
                                                                                         ncols)
                    feat[cur_idx, cur_pos + 20:cur_pos + 24] = self.get_moore_neighbours(input_color, i, j + 1, nrows,
                                                                                         ncols)
                    feat[cur_idx, cur_pos + 24:cur_pos + 28] = self.get_moore_neighbours(input_color, i, j - 1, nrows,
                                                                                         ncols)

                    feat[cur_idx, cur_pos + 28] = len(np.unique(feat[cur_idx, cur_pos + 3:cur_pos + 7]))
                    try:
                        feat[cur_idx, cur_pos + 29] = len(np.unique(input_color[i + 1, :]))
                        feat[cur_idx, cur_pos + 30] = len(np.unique(input_color[:, j + 1]))
                    except IndexError:
                        pass
                    feat[cur_idx, cur_pos + 31] = nrows - i
                    feat[cur_idx, cur_pos + 32] = ncols - j
                    feat[cur_idx, cur_pos + 33] = input_color.size
                    cur_inp += 1
                cur_idx += 1

        return feat

    def get_features(self, task):

        n = len(task['train'])
        print("num of train tasks", n)
        x_mats = [np.array(task['train'][i]['input']) for i in range(n)]
        y_mats = [np.array(task['train'][i]['output']) for i in range(n)]

        if all((x.shape[0] < y.shape[0] or x.shape[1] < y.shape[1]) for x, y in zip(x_mats, y_mats)):
            p1 = [DslTile().learn(x, x, y) for x, y in zip(x_mats, y_mats)]
            x_mats = p1
            if any(x is None for x in x_mats):
                return None, None

        if any(x.shape != y.shape for x, y in zip(x_mats, y_mats)):
            return None, None

        # augment data
        for i in range(n):
            x_mats.append(np.fliplr(x_mats[i]))
            y_mats.append(np.fliplr(y_mats[i]))
            x_mats.append(np.flipud(x_mats[i]))
            y_mats.append(np.flipud(y_mats[i]))
            # x_mats.append(np.rot90(x_mats[i]))
            # y_mats.append(np.rot90(y_mats[i]))
            # x_mats.append(np.rot90(x_mats[i], 2))
            # y_mats.append(np.rot90(y_mats[i], 2))
            # x_mats.append(np.rot90(x_mats[i], 3))
            # y_mats.append(np.rot90(y_mats[i], 3))

        feats, targets = [], []
        cur_idx = 0
        for x, y in zip(x_mats, y_mats):
            ftrs = self.make_features(x)
            feats.extend(ftrs)
            targets.extend(y.reshape(-1, ))
            cur_idx += 1

        return np.array(feats), np.array(targets)

    def predict_output_matrix(self, input_matrix, learner):
        feat = self.make_features(input_matrix)
        output_matrix = learner.predict(feat).reshape(input_matrix.shape)
        return output_matrix


rules_basic = [
     DslClone,
     DslNot,
     DslBlankCell,
     DslRotate90,
     DslRotate180,
     DslRotate270,
     DslFlipV,
     DslFlipH,
     DslTranspose,
     DslCropMinCC,
     DslCropMin,
     DslCropMaxCC,
     DslCropMax,
     DslSelectMax,
     DslSelectMin,
     DslLabel,
     DslLabelAfterRecoloring,
     DslGetLongestShape,
     DslGetLongestShapeCC,
     DslGetWidestShape,
     DslGetBiggestShape,
     DslDivide,
     DslGetMostFreqShape,
     DslGetLeastFreqShape,
     DslTileRotate,
     DslSetColoredBoard,
     DslMagnetObject,
     DslSelectPanel,
     DslContextFree,
     DslDownscale,
     DslSelectMostColoredPanel,
     DslGetMostColoredObject,
     DslRemoveNoise,
     DslRemoveNoiseMostFreq,
     DslRemoveNoiseBW,
     DslGravity
]

rules_upscale = [
    DslUpscale,
    DslUpscaleWrtColoredCnt,
    DslUpscaleWrtSize,
    DslTile,
    DslTileWrtColoredCnt,
    DslTileWrtSize,
    DslTileAndUpscale
]

rules_rotation = [
    DslClone,
    DslFlipH,
    DslFlipV,
    DslRotate90,
    DslRotate180,
    DslRotate270
]

rules_match = [
    DslFirst,
    DslSecond,
    DslAnd,
    DslOr,
    DslXor,
    DslConcatH,
    DslConcatV,
    DslNotaAndb,
    DslaAndNotb,
    DslNotaAndNotb,
    DslNotaOrb,
    DslNotaOrNotb
]

rules_color = [
    DslClrToSame,   
    DslClrToMinColor,
    DslClrToWidestColor,
    DslClrToLongestColor,
    DslClrToMaxColor,
    DslClrToMinShape,
    DslClrToMaxShape,
    DslClrWrtSizeOrder,
    DslClrWrtSize,
    DslClrChange,
    DslClrToMostFreqColor
]


def dsl_search(task):
    debug = False

    n = len(task['train'])
    x_mats = [np.asmatrix(task['train'][i]['input']).astype(np.uint8) for i in range(n)]
    y_mats = [np.asmatrix(task['train'][i]['output']).astype(np.uint8) for i in range(n)]

    if all(x.shape == y.shape for x, y in zip(x_mats, y_mats)):
        rules_0 = rules_basic 
        rules_1 = rules_basic
        rules_2 = rules_basic
    else:
        rules_0 = rules_basic + rules_upscale
        rules_1 = rules_basic
        rules_2 = rules_basic + rules_upscale

    # rules_0 = [DslLabelAfterRecoloring]
    # rules_1 = [DslClone]
    # rules_2 = [DslClone]
    # debug = True

    target_rules = []

    for R0 in rules_0:
        r0 = R0()
        p0 = r0.learn(x_mats, x_mats, y_mats)
        # print(r0.__class__)
        if p0 is None or any(p is None for p in p0):
            continue
        if debug:
            print(r0.__class__)
            plot_pair(p0[0], p0[1])

        for R1 in rules_1:
            r1 = R1()
            p1 = r1.learn(p0, x_mats, y_mats)
            # print('-->', r1.__class__)
            if p1 is None or any(p is None for p in p1):
                # print(r0.__class__, ' ', r1.__class__)
                continue
            if debug:
                print('-->', r1.__class__)
                plot_pair(p1[0], p1[1])

            for R2 in rules_2:
                r2 = R2()
                p2 = r2.learn(p1, x_mats, y_mats)
                # print('---->', r2.__class__)
                if p2 is None or any(p is None for p in p2):
                    continue

                if debug:
                    print('---->', r2.__class__)
                    plot_pair(p2[0], p2[1])

                if helper_mats_are_equal(y_mats, p2):
                    print('rule found')
                    target_rule = (r0, r1, r2)
                    target_rules.append(target_rule)
                    if len(target_rules) == TARGET_RULE_LIMIT:
                        return target_rules
                    else:
                        continue

                if debug:
                    print('mats are not equal, checking for patterns')

                # if any(p.shape != y.shape for p, y in zip(p2, y_mats)):
                #     continue

                # if not helper_patterns_are_equal(p2, y_mats):
                #     continue

                if debug:
                    print('patterns are equal, searching a rule for recoloring')

                '''find a rule for recoloring'''
                for R3 in rules_color:
                    r3 = R3()
                    p3 = r3.learn(p2, x_mats, y_mats)
                    # plot_pair(p3[0], p3[1])
                    if p3 is None or any(p is None for p in p3):
                        continue

                    if helper_mats_are_equal(y_mats, p3):
                        if debug:
                            for y, p in zip(y_mats, p3):
                                plot_pair(y, p)
                        target_rule = (r0, r1, r2, r3)
                        target_rules.append(target_rule)
                        if len(target_rules) == TARGET_RULE_LIMIT:
                             return target_rules
    return target_rules


def dsl_pred(task, test_id=0):

    start_time = time.time()
    found_rules = dsl_search(task)
    print("dsl search duration", time.time() - start_time)

    if len(found_rules) == 0:
        return []

    print("Num of found rules", len(found_rules))

    preds = []
    for i, rules in enumerate(found_rules):

        print('rules')
        print(rules[0].__class__)
        print(rules[1].__class__)
        print(rules[2].__class__)

        ''' test part '''
        t = np.asmatrix(task['test'][test_id]['input'])
        p0 = rules[0].pred(t, t)
        if p0 is None:
            continue
        p1 = rules[1].pred(p0, t)
        if p1 is None:
            continue
        start_time = time.time()
        pred = rules[2].pred(p1, t)
        if pred is None:
            continue

        ''' if color rule exists '''
        if len(rules) == 4:
            print(rules[3].__class__)
            pred = rules[3].pred(pred, t)

        if pred is not None:
            preds.append(pred.astype(int).tolist())

        # if i == 2:
        #     break

    return preds


def dsl_search_task112(task):
    debug = False

    n = len(task['train'])
    x_mats = [np.asmatrix(task['train'][i]['input']).astype(np.uint8) for i in range(n)]
    y_mats = [np.asmatrix(task['train'][i]['output']).astype(np.uint8) for i in range(n)]

    rules_0 = [DslClone]
    rules_1 = [DslFlipV, DslFlipH]
    rules_2 = rules_match

    # rules_0 = [DslLabel]
    # rules_1 = [DslClrWrtSize]
    # rules_2 = [DslClone]
    # debug = True

    target_rules = []

    for R0 in rules_0:
        r0 = R0()
        p0 = r0.learn(x_mats, x_mats, y_mats)
        # print(r0.__class__)
        if p0 is None or any(p is None for p in p0):
            continue
        if debug:
            plot_pair(p0[0], p0[1])

        for R1 in rules_1:
            r1 = R1()
            p1 = r1.learn(x_mats, x_mats, y_mats)
            # print('-->', r1.__class__)
            if p1 is None or any(p is None for p in p1):
                # print(r0.__class__, ' ', r1.__class__)
                continue
            if debug:
                plot_pair(p1[0], p1[1])

            for R2 in rules_2:
                r2 = R2()
                p2 = r2.learn(p0, p1, x_mats, y_mats)
                # print('---->', r2.__class__)
                if p2 is None or any(p is None for p in p2):
                    continue

                if debug:
                    plot_pair(p2[0], p2[1])

                if helper_mats_are_equal(y_mats, p2):
                    target_rule = (r0, r1, r2)
                    target_rules.append(target_rule)
                    if len(target_rules) == 3:
                        return target_rules
                    else:
                        continue

    return target_rules


def  dsl_pred_task112(task, test_id=0):

    start_time = time.time()
    found_rules = dsl_search_task112(task)
    print("dsl search duration", time.time() - start_time)

    if len(found_rules) == 0:
        return []

    print("Num of found rules", len(found_rules))

    preds = []
    for i, rules in enumerate(found_rules):

        print('rules')
        print(rules[0].__class__)
        print(rules[1].__class__)
        print(rules[2].__class__)

        ''' test part '''
        t = np.asmatrix(task['test'][test_id]['input'])
        p0 = rules[0].pred(t, t)
        if p0 is None:
            continue
        p1 = rules[1].pred(t, t)
        if p1 is None:
            continue
        pred = rules[2].pred(p0, p1, t)
        if pred is None:
            continue

        if pred is not None:
            preds.append(pred.astype(int).tolist())

        if i == 2:
            break

    return preds


def dsl_search_task169(task):
    debug = False

    n = len(task['train'])
    x_mats = [np.asmatrix(task['train'][i]['input']).astype(np.uint8) for i in range(n)]
    y_mats = [np.asmatrix(task['train'][i]['output']).astype(np.uint8) for i in range(n)]

    # rules_0 = [DslClone]
    # rules_1 = [DslFlipV, DslFlipH]
    # rules_2 = rules_match

    rules_0 = [DslCropMaxCC]
    rules_1 = [DslCropMinCC]
    rules_2 = [DslDownscaleToObjectSize]
    rules_3 = [DslCopyObjectsColors]
    #debug = True

    target_rules = []

    for R0 in rules_0:
        r0 = R0()
        p0 = r0.learn(x_mats, x_mats, y_mats)
        # print(r0.__class__)
        if p0 is None or any(p is None for p in p0):
            continue
        if debug:
            plot_pair(p0[0], p0[1])

        for R1 in rules_1:
            r1 = R1()
            p1 = r1.learn(x_mats, x_mats, y_mats)
            # print('-->', r1.__class__)
            if p1 is None or any(p is None for p in p1):
                # print(r0.__class__, ' ', r1.__class__)
                continue
            if debug:
                plot_pair(p1[0], p1[1])

            for R2 in rules_2:
                r2 = R2()
                p2 = r2.learn(p0, p1, x_mats)
                # print('-->', r1.__class__)
                if p2 is None or any(p is None for p in p2):
                    # print(r0.__class__, ' ', r1.__class__)
                    continue
                if debug:
                    plot_pair(p2[0], p2[1])

                for R3 in rules_3:
                    r3 = R3()
                    p3 = r3.learn(p2, p1, x_mats)
                    # print('-->', r1.__class__)
                    if p3 is None or any(p is None for p in p3):
                        # print(r0.__class__, ' ', r1.__class__)
                        continue
                    if debug:
                        plot_pair(p3[0], p3[1])
     
                    if helper_mats_are_equal(y_mats, p3):
                        target_rule = (r0, r1, r2, r3)
                        target_rules.append(target_rule)
                        if len(target_rules) == 3:
                            return target_rules
                        else:
                            continue

    return target_rules


def  dsl_pred_task169(task, test_id=0):

    start_time = time.time()
    found_rules = dsl_search_task169(task)
    print("dsl search duration", time.time() - start_time)

    if len(found_rules) == 0:
        return []

    print("Num of found rules", len(found_rules))

    preds = []
    for i, rules in enumerate(found_rules):

        print('rules')
        print(rules[0].__class__)
        print(rules[1].__class__)
        print(rules[2].__class__)
        print(rules[3].__class__)

        ''' test part '''
        t = np.asmatrix(task['test'][test_id]['input'])
        p0 = rules[0].pred(t, t)
        if p0 is None:
            continue
        p1 = rules[1].pred(t, t)
        if p1 is None:
            continue
        p2 = rules[2].pred(p0, p1, t)
        if p2 is None:
            continue
        pred = rules[3].pred(p2, p1, t)
        if pred is None:
            continue

        # ''' if color rule exists '''
        # if len(rules) == 4:
        #     print(rules[3].__class__)
        #     pred = rules[3].pred(pred, t)

        if pred is not None:
            preds.append(pred.astype(int).tolist())

        if i == 2:
            break

    return preds


def ml_pred(task, test_id=0):

    start_time = time.time()

    ml = MLStack()

    p0 = ml.learn(task, test_id)
    if p0 is None or any(p is None for p in p0):
        return None

    '''find a rule for recoloring'''
    n = len(task['train'])
    x_mats = [np.asmatrix(task['train'][i]['input']).astype(np.uint8) for i in range(n)]
    y_mats = [np.asmatrix(task['train'][i]['output']).astype(np.uint8) for i in range(n)]

    color_rule = None
    for R3 in rules_color:
        r3 = R3()
        p3 = r3.learn(p0, x_mats, y_mats)
        if p3 is None or any(p is None for p in p3):
            continue

        if helper_mats_are_equal(y_mats, p3):
            color_rule = r3
            break

    print("dsl search duration", time.time() - start_time)

    ''' test part '''
    t = np.asmatrix(task['test'][test_id]['input'])
    pred = ml.pred(t)
    if pred is None:
        return None

    ''' if color rule exists '''
    if color_rule is not None:
        print(color_rule.__class__)
        pred = color_rule.pred(pred, t)
        if pred is None:
            return None

    return pred.astype(int).tolist()


def solve_task(task, test_id=0):
    preds = []
    pred_methods = []

    def add_to_preds(p, m):
        if len(preds) == 3:
            print('No room for new prediction')
            return
        for a in preds:
            if p == a:
                print('This prediction is already made')
                return
        preds.append(p)
        pred_methods.append(m)
        return len(preds)

    ''' DSL '''
    ps = dsl_pred(task, test_id)
    for i in range(len(ps)):
        if 3 == add_to_preds(ps[i], 'dsl'):
            return preds, pred_methods

    ps = dsl_pred_task112(task, test_id)
    for i in range(len(ps)):
        if 3 == add_to_preds(ps[i], 'dsl_task_112'):
            return preds, pred_methods

    ps = dsl_pred_task169(task, test_id)
    for i in range(len(ps)):
        if 3 == add_to_preds(ps[i], 'dsl_task_169'):
            return preds, pred_methods

    ''' HC '''
    for func in hc_func_list:
        print('trying ', func.__qualname__)
        pred = func(task, test_id)
        print('  result: ', pred is not None)
        if pred is not None:
            if 3 == add_to_preds(pred, func.__qualname__):
                return preds, pred_methods

    ''' ML '''
    pred = ml_pred(task, test_id)
    if pred is not None:
        if 3 == add_to_preds(pred, "ML"):
            return preds, pred_methods

    return preds, pred_methods


''' SUBMISSION PART '''
import pandas as pd
import os
import json
from pathlib import Path

mode = 'TEST'
TARGET_RULE_LIMIT = 9

for dirname, _, filenames in os.walk('/kaggle/input'):
    print(dirname)

data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'
training_tasks = sorted(os.listdir(training_path))
eval_tasks = sorted(os.listdir(evaluation_path))


hc_func_list = [
    Recolor().predict
]

from multiprocessing import Pool
if mode == "TRAIN" or mode == "EVAL":
    print('\n\n ' + mode + ' EXAMPLES \n')

    Tasks = []
    TaskIds = []

    if mode == "TRAIN":
        liste = [0, 6, 7, 9, 13, 15, 30, 35, 38, 48, 51, 52, 56, 64, 66, 72, 78, 82, 86, 94, 99, 105,
                 108, 110, 115, 128, 129, 139, 141, 146, 149, 151, 154, 163, 168, 171, 173, 176,
                 178, 187, 193, 194, 206, 209, 210, 219, 222, 229, 240, 248, 249, 256, 257, 262,
                 266, 268, 271, 275, 276, 281, 282, 288, 289, 293, 299, 304, 306, 308, 309, 310,
                 316, 330, 373, 379, 383, 384]
    elif mode == "EVAL":
        liste = [0, 1, 6, 12, 16, 138, 143, 148, 156, 167, 176, 184, 196, 212, 235, 251, 264, 285,
                 288, 294, 296, 312, 325, 351, 375]
    # liste = range(400)
    # liste = [188]

    for i in liste:
        if mode == "TRAIN":
            task_file = str(training_path / training_tasks[i])
        else:
            task_file = str(evaluation_path / eval_tasks[i])
        task = json.load(open(task_file, 'r'))
        # au.plot_task(task)
        Tasks.append(task)
        TaskIds.append((i, training_tasks[i]))

    dataset = Tasks
    Ids = TaskIds
    def solve_dataset(start):
        solved_tasks = []
        wrong_preds = []
        for i in range(start, start+task_per_process, 1):
            task = dataset[i]

            print('\ntask no ', i, ' task name ', Ids[i])
            
            preds, pred_methods = solve_task(task, 0)

            # print('num of total preds made for this task:', len(preds))
            p = 0
            for pred in preds:
                test_out = task['test'][0]['output']
                plot_pair(test_out, pred)
                print('pred_methods:',pred_methods[p])
                if test_out == pred:
                    print('CORRECT PREDICTION')
                    solved_tasks.append((i, Ids[i], pred_methods[p]))
                    break
                else:
                    wrong_preds.append((i, Ids[i], pred_methods[p]))
                    print('WRONG PREDICTION')
                p = p + 1

        return solved_tasks, wrong_preds


    ''' caller code '''
    start_time = datetime.datetime.now().replace(microsecond=0)

    if True: # len(liste) == 1:
        task_cnt = len(dataset)
        task_per_process = task_cnt
        solved_tasks, wrong_preds = solve_dataset(0)
    else:
        ''' multiprocess '''
        process_cnt = 20
        task_cnt = len(dataset)
        task_per_process = task_cnt // process_cnt
        p = Pool(processes=process_cnt)
        res = p.map(solve_dataset, [i for i in range(0, task_cnt, task_per_process)])
        p.close()
        solved_tasks = []
        wrong_preds = []
        for r in res:
            solved_tasks = solved_tasks + r[0]
            wrong_preds = wrong_preds + r[1]


    finish_time = datetime.datetime.now().replace(microsecond=0)
    print('duration:', finish_time - start_time)

    print('\nsolved ', mode, ' example count ', len(solved_tasks))
    for s in range(len(solved_tasks)):
        print('',s,'-',solved_tasks[s])

    print('\nNum of wrong predictions:', len(wrong_preds))
    for w in range(len(wrong_preds)):
        print('', w, '-', wrong_preds[w])


elif mode == 'TEST':

    print('\n\n TESTING MODE \n')

    submission = pd.read_csv(data_path / 'sample_submission.csv')
    submission.head()

    def flattener(pred):
        str_pred = str([row for row in pred])
        str_pred = str_pred.replace(', ', '')
        str_pred = str_pred.replace('[[', '|')
        str_pred = str_pred.replace('][', '|')
        str_pred = str_pred.replace(']]', '|')
        return str_pred

    example_grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(flattener(example_grid))

    solved_test_examples = []
    Problems = submission['output_id'].values
    Proposed_Answers = []

    def make_real_submission():
        for i in range(len(Problems)):
            output_id = Problems[i]
            task_id = output_id.split('_')[0]
            test_id = int(output_id.split('_')[1])
            f = str(test_path / str(task_id + '.json'))

            with open(f, 'r') as read_file:
                task = json.load(read_file)

            print('\ntask no ', i, ' task_id ', task_id, ' - ', test_id)
            # au.plot_task(task)

            preds, preds_methods = solve_task(task, test_id)

            solution = ''
            pc = min(3, len(preds))
            for j in range(pc):
                plot_matrix(preds[j])
                solution = solution + flattener(preds[j]) + ' '

            if solution == '':
                solution = flattener(example_grid)

            print('solution', solution)

            Proposed_Answers.append(solution)

        submission['output'] = Proposed_Answers
        submission.to_csv('submission.csv', index=False)

        for solved_task in solved_test_examples:
            print(solved_task)

    def make_fake_submission():
        for i in range(len(Problems)):
            solution = flattener(example_grid)
            Proposed_Answers.append(solution)
        submission['output'] = Proposed_Answers
        submission.to_csv('submission.csv', index=False)

    print('id of first problem:', Problems[0])
    if Problems[0] == '00576224_0':
        print('fake submission')
        make_fake_submission()
    else:
        print('real submission')
        make_real_submission()
