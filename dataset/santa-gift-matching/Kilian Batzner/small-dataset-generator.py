from collections import Counter

import numpy as np
import pandas as pd

n_children = 10000
n_gift_type = 100
n_gift_quantity = 100
n_gift_pref = 5
n_child_pref = 100
twins = 40
ratio_gift_happiness = 2
ratio_child_happiness = 2


def generate_small_dataset():
    # Generate the children's wishlists
    child_wishlist = []
    for i_child in range(n_children):
        wishes = np.random.choice(n_gift_type, n_gift_pref, replace=False)
        wishes = np.insert(wishes, 0, i_child)
        child_wishlist.append(wishes)
    child_wishlist = np.array(child_wishlist)

    # Generate the gift's good kids
    gift_goodkids = []
    for i_gift in range(n_gift_type):
        goodkids = np.random.choice(n_children, n_child_pref, replace=False)
        goodkids = np.insert(goodkids, 0, i_gift)
        gift_goodkids.append(goodkids)
    gift_goodkids = np.array(gift_goodkids)

    np.savetxt("child_wishlist_small.csv", child_wishlist, fmt='%i', delimiter=",")
    np.savetxt("gift_goodkids_small.csv", gift_goodkids, fmt='%i', delimiter=",")

    return child_wishlist, gift_goodkids


def avg_normalized_happiness(pred, child_pref, gift_pref):
    # check if number of each gift exceeds n_gift_quantity
    gift_counts = Counter(elem[1] for elem in pred)
    for count in gift_counts.values():
        assert count <= n_gift_quantity

    # check if twins have the same gift
    for t1 in range(0, twins, 2):
        twin1 = pred[t1]
        twin2 = pred[t1 + 1]
        assert twin1[1] == twin2[1]

    max_child_happiness = n_gift_pref * ratio_child_happiness
    max_gift_happiness = n_child_pref * ratio_gift_happiness
    total_child_happiness = 0
    total_gift_happiness = np.zeros(n_gift_type)

    for row in pred:
        child_id = row[0]
        gift_id = row[1]

        # check if child_id and gift_id exist
        assert child_id < n_children
        assert gift_id < n_gift_type
        assert child_id >= 0
        assert gift_id >= 0
        child_happiness = (n_gift_pref - np.where(gift_pref[child_id] == gift_id)[0]) * ratio_child_happiness
        if not child_happiness:
            child_happiness = -1

        gift_happiness = (n_child_pref - np.where(child_pref[gift_id] == child_id)[0]) * ratio_gift_happiness
        if not gift_happiness:
            gift_happiness = -1

        total_child_happiness += child_happiness
        total_gift_happiness[gift_id] += gift_happiness

    # print(max_child_happiness, max_gift_happiness
    print('normalized child happiness=',
          float(total_child_happiness) / (float(n_children) * float(max_child_happiness)), \
          ', normalized gift happiness', np.mean(total_gift_happiness) / float(max_gift_happiness * n_gift_quantity))
    return float(total_child_happiness) / (float(n_children) * float(max_child_happiness)) + np.mean(
        total_gift_happiness) / float(max_gift_happiness * n_gift_quantity)


def generate_random_solution():
    gift_counts = dict((g, n_gift_quantity) for g in range(n_gift_type))
    random_sub = []

    for i_child in range(n_children):
        # Give twins the same present
        if i_child < 4000:
            if i_child % 2 == 0:
                # Select a gift that has a count of at least two
                candidate_gifts = [g for g in gift_counts if gift_counts[g] >= 2]
                choice = np.random.choice(candidate_gifts, 1)[0]
            else:
                # Select the same present as the first twin
                choice = random_sub[-1][1]
        else:
            # Select a gift that has a count of at least one
            candidate_gifts = [g for g in gift_counts if gift_counts[g] >= 1]
            choice = np.random.choice(candidate_gifts, 1)[0]
        # Assign the gift
        random_sub.append([i_child, choice])
        gift_counts[choice] -= 1
    return random_sub


if __name__ == '__main__':
    generate_small_dataset()

    # Read the generated dataset
    gift_pref = pd.read_csv('child_wishlist_small.csv', header=None).drop(0, 1).values
    child_pref = pd.read_csv('gift_goodkids_small.csv', header=None).drop(0, 1).values

    # Generate a random solution
    random_sub = generate_random_solution()

    print(avg_normalized_happiness(random_sub, child_pref, gift_pref))