#coding:utf-8
import pandas as pd
import copy
import gc

#======================
# Read CSV File
#======================
def read_csv(filename):
    datas = pd.read_csv(filename, header=None, index_col=0)
    return datas

#======================
# Calculate Happiness Values
#======================
def generate_happiness_list(like_list):
    happiness_list = {}
    for i in like_list.index.values: happiness_list[i] = {}
    for i, v in like_list.iterrows():
        happiness = float(len(v.values))
        for j in v.values:
            happiness_list[i][j] = happiness*2
            happiness -= 1.0
    return happiness_list

#======================
# Constant Definition
#======================
CHILD_FILE_PATH = '../input/child_wishlist_v2.csv'
GIFT_FILE_PATH = '../input/gift_goodkids_v2.csv'
RESULT_FILE_PATH = './result.csv'
CHILD_HAPPINESS_LIST = generate_happiness_list(read_csv(CHILD_FILE_PATH))
GIFT_HAPPINESS_LIST = generate_happiness_list(read_csv(GIFT_FILE_PATH))
CHILD_SIZE = len(CHILD_HAPPINESS_LIST)
GIFT_SIZE = len(GIFT_HAPPINESS_LIST)
WISH_SIZE = len(CHILD_HAPPINESS_LIST[0])
GOODKIDS_SIZE = len(GIFT_HAPPINESS_LIST[0])
MAX_TRIPLE_INDEX = 5000
MAX_TWINS_INDEX = 45000
MAX_FEATURE_SIZE = 20

#======================
# TODO: Calculate Feature
#======================
def calculate_feature(child, gift):
    b = brother_num(child)
    feature = 0
    candidate = []
    if b==3: candidate = [child-2, child-1, child]
    elif b==2: candidate = [child-1, child]
    elif b==1: candidate = [child]
    for c in candidate:
        ch = CHILD_HAPPINESS_LIST[c][gift] if g in CHILD_HAPPINESS_LIST[c] else -1.0
        gh = GIFT_HAPPINESS_LIST[gift][c] if c in GIFT_HAPPINESS_LIST[gift] else -1.0
        feature += ch/(WISH_SIZE*2)+gh/(GOODKIDS_SIZE*2)
    return feature/b

#======================
# Calculate Feature
#======================
def generate_mix_happiness_list():
    happiness_list = {}
    child_g_list = {}
    for c, happiness in CHILD_HAPPINESS_LIST.items():
        child_g_list[c] = list(happiness.keys())
    for g, happiness in GIFT_HAPPINESS_LIST.items():
        for c in happiness.keys():
            if g not in child_g_list[c]: child_g_list[c].append(g)
    for c in list(CHILD_HAPPINESS_LIST.keys()):
        happiness = {}
        for g in set(child_g_list[c]):
            ch =  CHILD_HAPPINESS_LIST[c][g] if g in CHILD_HAPPINESS_LIST[c] else -1.0
            gh =  GIFT_HAPPINESS_LIST[g][c] if c in GIFT_HAPPINESS_LIST[g] else -1.0
            happiness[g] = (ch/(WISH_SIZE*2))**3+(gh/(GOODKIDS_SIZE*2))**3
        b = brother_num(c)
        if b==3 and c%3==2 :
            h1 = happiness_list[c-2]
            h2 = happiness_list[c-1]
            for g in set(list(happiness.keys())+list(h1.keys())+list(h2.keys())):
                x0 = happiness[g] if g in happiness else (-1.0/(WISH_SIZE*2))**3+(-1.0/(GOODKIDS_SIZE*2))**3
                x1 = h1[g] if g in h1 else (-1.0/(WISH_SIZE*2))**3+(-1.0/(GOODKIDS_SIZE*2))**3
                x2 = h2[g] if g in h2 else (-1.0/(WISH_SIZE*2))**3+(-1.0/(GOODKIDS_SIZE*2))**3
                happiness[g] = (x0+x1+x2)/3
            happiness_list[c] = sorted(happiness.items(), key=lambda x: -x[1])[0:MAX_FEATURE_SIZE]
            happiness_list[c].append([GIFT_SIZE,-1])
            happiness_list[c-1] = happiness_list[c-2] = happiness_list[c]
        elif b==2 and c%2==0:
            h1 = happiness_list[c-1]
            for g in set(list(happiness.keys())+list(h1.keys())):
                x0 = happiness[g] if g in happiness else (-1.0/(WISH_SIZE*2))**3+(-1.0/(GOODKIDS_SIZE*2))**3
                x1 = h1[g] if g in h1 else (-1.0/(WISH_SIZE*2))**3+(-1.0/(GOODKIDS_SIZE*2))**3
                happiness[g] = (x0+x1)/2
            happiness_list[c] = sorted(happiness.items(), key=lambda x: -x[1])[0:MAX_FEATURE_SIZE]
            happiness_list[c].append([GIFT_SIZE,-1])
            happiness[c-1] = happiness_list[c]
        else:
            happiness_list[c] = happiness
            if b==1:
                happiness_list[c] = sorted(happiness.items(), key=lambda x: -x[1])[0:MAX_FEATURE_SIZE]
                happiness_list[c].append([GIFT_SIZE,-1])
        del(child_g_list[c])
        if(c%10000==0): gc.collect()
    return happiness_list

#======================
# Match Algolithm
#======================
def match():
    result = {}
    gifts = {}
    for i in range(0, GIFT_SIZE): gifts[i] = GIFT_SIZE
    triple_children = {}
    twin_children = {}
    single_children = {}

    # split children
    for c, g_list in list(mix_happiness_list.items()):
        b = brother_num(c)
        if b==3 and c%3==2: triple_children[c] = g_list
        if b==2 and c%2==0: twin_children[c] = g_list
        if b==1: single_children[c] = g_list
        del mix_happiness_list[c]
    triple_children = sorted(triple_children.items(), key=lambda x: -x[1][0][1])
    twin_children = sorted(twin_children.items(), key=lambda x: -x[1][0][1])
    single_children = sorted(single_children.items(), key=lambda x: -x[1][0][1])
    best_gift_id = -1

    print("match triple!")
    while True:
        print(len(triple_children))
        if len(triple_children)>0: print(triple_children[0][0], triple_children[0][1][0])
        while True:
            if len(triple_children)==0: break
            c = triple_children[0][0]
            g_list = triple_children[0][1]
            gift_id = -1
            best_gift_id = g_list[0][0]
            if best_gift_id == GIFT_SIZE:
                for g in range(0, GIFT_SIZE):
                    gift_id = check_gift(g, gifts, 3)
                    if gift_id >= 0 : break
            else:
                gift_id = check_gift(best_gift_id, gifts, 3)
            if gift_id >= 0:
                result[c] = [gift_id]
                result[c-1] = [gift_id]
                result[c-2] = [gift_id]
                del triple_children[0]
            else:
                break
        if len(triple_children)==0: break
        triple_children = remove_gift(triple_children, best_gift_id)
        twin_children = remove_gift(twin_children, best_gift_id)
        single_children = remove_gift(single_children, best_gift_id)
    best_gift_id = -1

    print("match twins!")
    while True:
        print(len(twin_children))
        if len(twin_children)>0: print(twin_children[0][0], twin_children[0][1][0])
        while True:
            if len(twin_children)==0: break
            c = twin_children[0][0]
            g_list = twin_children[0][1]
            gift_id = -1
            best_gift_id = g_list[0][0]
            if best_gift_id == GIFT_SIZE:
                for g in range(0, GIFT_SIZE):
                    gift_id = check_gift(g, gifts, 2)
                    if gift_id >= 0 : break
            else:
                gift_id = check_gift(best_gift_id, gifts, 2)
            if gift_id >= 0:
                result[c] = [gift_id]
                result[c-1] = [gift_id]
                del twin_children[0]
            else:
                break
        if len(twin_children)==0: break
        twin_children = remove_gift(twin_children, best_gift_id)
        single_children = remove_gift(single_children, best_gift_id)
    best_gift_id = -1

    print("match single!")
    while True:
        print(len(single_children))
        if len(single_children)>0: print(single_children[0][0], single_children[0][1][0])
        while True:
            if len(single_children)==0: break
            c = single_children[0][0]
            g_list = single_children[0][1]
            gift_id = -1
            best_gift_id = g_list[0][0]
            if best_gift_id == GIFT_SIZE:
                for g in range(0, GIFT_SIZE):
                    gift_id = check_gift(g, gifts)
                    if gift_id >= 0 : break
            else:
                gift_id = check_gift(best_gift_id, gifts)
            if gift_id >= 0:
                result[c] = [gift_id]
                del single_children[0]
            else:
                break
        if len(single_children)==0: break
        single_children = remove_gift(single_children, best_gift_id)
    return result

def brother_num(index):
    if index <= MAX_TRIPLE_INDEX: return 3
    if index <= MAX_TWINS_INDEX: return 2
    return 1

def check_gift(id, gifts, left_num=1):
    if gifts[id] < left_num : return -1
    gifts[id]-=left_num
    return id

def remove_gift(child_list, gift_id):
    c_index = 0
    while c_index < len(child_list):
        g_list = child_list[c_index][1]
        g_index = 0
        while g_index < len(g_list):
            if g_list[g_index][0] == gift_id:
                del g_list[g_index]
                break
            g_index += 1
        c_index += 1
    return sorted(child_list, key=lambda x: -x[1][0][1])

#======================
# Evaluate
#======================
def evaluate(result):
    anch = 0
    ansh = 0
    for c, g in result.items():
        wish = CHILD_HAPPINESS_LIST[c][g[0]] if g[0] in CHILD_HAPPINESS_LIST[c] else -1.0
        gift = GIFT_HAPPINESS_LIST[g[0]][c] if c in GIFT_HAPPINESS_LIST[g[0]] else -1.0
        anch += wish/(WISH_SIZE*2)
        ansh += gift/(GOODKIDS_SIZE*2)
    print(anch)
    print(ansh)
    return (anch/CHILD_SIZE)**3+(ansh/CHILD_SIZE)**3

#======================
# Output Result File
#======================
def output(res):
    result = pd.DataFrame(res).T
    result = result.rename(columns={0:'GiftId'})
    result.to_csv(RESULT_FILE_PATH, index_label='ChildId', header=True)

#======================
# Main Process
#======================
print("generate mix happiness_list ...")
mix_happiness_list = generate_mix_happiness_list()
gc.collect()

print("create result...")
result = match()

print("evaluate result...")
print(evaluate(result))
output(result)
