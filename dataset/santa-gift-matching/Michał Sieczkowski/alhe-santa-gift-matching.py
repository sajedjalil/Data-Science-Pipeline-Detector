import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from ortools.graph import pywrapgraph
import os

DATA_FILES_DIR = '../input/'

def evaluate_score(solution, child_prefs, gift_prefs, last_triplet, last_twin, gift_supply):

    # czy wszystkie trojaczki mają ten sam prezent
    for c in range(0, last_triplet + 1, 3):
        assert solution[c] == solution[c + 1] and solution[c + 1] == solution[c + 2]

    # czy wszystkie bliżniaki mają ten sam prezent
    for c in range(last_triplet + 1, last_twin + 1, 2):
        assert solution[c] == solution[c + 1]

    num_children = child_prefs.shape[0]
    num_gifts = gift_prefs.shape[0]
    num_child_prefs = child_prefs.shape[1]
    num_gift_prefs = gift_prefs.shape[1]

    # czy rozwiązanie obejmuje wszystkie dzieci
    assert len(solution) == num_children

    # ilość użytych prezentów danego typu
    gift_count = np.zeros(1000, dtype=np.int32)

    child_happiness_sum = 0
    gift_happiness_sum = 0

    for c in range(len(solution)):
        g = solution[c]  # id prezentu
        # czy prezent został przydzielony i jego numer mieści się w zakresie rodzajów prezentów
        assert g >= 0 and g < num_gifts

        gift_count[g] += 1
        child_happiness = 2 * (num_child_prefs - np.where(child_prefs[c] == g)[0])
        if not child_happiness:
            child_happiness = -1

        child_happiness_sum += child_happiness

        gift_happiness = 2 * (num_gift_prefs - np.where(gift_prefs[g] == c)[0])
        if not gift_happiness:
            gift_happiness = -1

        gift_happiness_sum += gift_happiness

    # czy prezent nie został przydzielony zbyt wiele razy
    assert gift_count.max() <= gift_supply

    max_child_happiness = num_child_prefs * 2
    max_gift_happiness = num_gift_prefs * 2

    anch = child_happiness_sum[0] / (max_child_happiness * num_children)
    ansh = gift_happiness_sum[0] / (max_gift_happiness * num_gifts * gift_supply)

    score = anch ** 3 + ansh ** 3

    return score

def include_child_prefs(cost, child_prefs, last_triplet, last_twin, prefs_limit=100):
    for c in range(0, last_triplet + 1):
        triplet = c - (c % 3)
        for g in range(prefs_limit):
            if (triplet, child_prefs[c][g]) in cost:
                cost[(triplet, child_prefs[c][g])] += ((child_prefs.shape[1] - g) * 2) ** 3
            else:
                cost[(triplet, child_prefs[c][g])] = ((child_prefs.shape[1] - g) * 2) ** 3


    for c in range(last_triplet + 1, last_twin + 1):
        twin = c + (c % 2)
        for g in range(prefs_limit):
            if (twin, child_prefs[c][g]) in cost:
                cost[(twin, child_prefs[c][g])] += ((child_prefs.shape[1] - g) * 2) ** 3
            else:
                cost[(twin, child_prefs[c][g])] = ((child_prefs.shape[1] - g) * 2) ** 3

    for c in range(last_twin + 1, child_prefs.shape[0]):
        single = c
        for g in range(prefs_limit):
            cost[(single, child_prefs[c][g])] = ((child_prefs.shape[1] - g) * 2) ** 3


def include_santa_prefs(cost, santa_prefs, last_triplet, last_twin, prefs_limit=1000):
    for g in range(santa_prefs.shape[0]):
        for j in range(santa_prefs.shape[1]):
            child_node = santa_prefs[g][j]
            if child_node <= last_triplet:
                child_node -= child_node % 3
            elif child_node <= last_twin:
                child_node += child_node % 2

            if (child_node, g) in cost:
                cost[(child_node, g)] += (santa_prefs.shape[1] - j) * 2
            else:
                cost[(child_node, g)] = (santa_prefs.shape[1] - j) * 2

def find_best_gift(child_id, child_prefs, gift_prefs, gift_count):
    happiness = []

    for i in range(0, len(child_prefs[child_id])):
        h = ((child_prefs.shape[1] - i) * 2) ** 3
        g = child_prefs[child_id][i]
        gift_happiness = ((gift_prefs.shape[1] - np.where(gift_prefs[g] == child_id)[0]) * 2)
        if gift_happiness:
            h += gift_happiness
        happiness.append((g, h))

    happiness.sort(reverse=True, key=lambda v: v[1])

    for h in happiness:
        gift = h[0]
        if gift_count[gift] < 1000:
            return gift

    return np.argmin(gift_count)

def get_score(child_id, gift_id, child_prefs, gift_prefs):
    h = 0

    child_happiness = ((child_prefs.shape[1] - np.where(child_prefs[child_id] == gift_id)[0]) * 2) ** 3
    if child_happiness:
        h += child_happiness
    gift_happiness = ((gift_prefs.shape[1] - np.where(gift_prefs[gift_id] == child_id)[0]) * 2)
    if gift_happiness:
        h += gift_happiness
    return h

def main():
    last_triplet = 5000
    last_twin = 45000
    child_prefs_limit = 40
    gift_prefs_limit = 1000
    gift_supply = 1000

    # child_prefs[i] -> lista prezentów preferowanych przez dziecko o numerze i
    # gift_prefs[i] -> lista dzieci preferowanych przez św. Mikołaja dla i-tego prezentu
    child_prefs = pd.read_csv(DATA_FILES_DIR + 'child_wishlist_v2.csv', header=None).values[:, 1:]
    gift_prefs = pd.read_csv(DATA_FILES_DIR + 'gift_goodkids_v2.csv', header=None).values[:, 1:]

    # mapowanie pary (child, gift) na koszt przepływu dla tej krawędzi
    child_gift_cost = dict()

    # dodaj krawędzie na podstawie preferencji dziecka
    include_child_prefs(child_gift_cost, child_prefs, last_triplet, last_twin, child_prefs_limit)

    # dodaj krawędzie na podstawie preferencji św. Mikołaja
    include_santa_prefs(child_gift_cost, gift_prefs, last_triplet, last_twin, gift_prefs_limit)

    print('Liczba utworzonych krawędzi (child, gift): {}'.format(len(child_gift_cost.keys())))

    # utwórz graf sieci przepływów
    start_nodes = []
    end_nodes = []
    capacities = []
    unit_costs = []
    supplies = []

    num_children = child_prefs.shape[0]
    num_gifts = gift_prefs.shape[0]

    # definicja pojemności dla każdej krawędzi (child, gift), pojemność zależna od typu dziecka
    for pair in child_gift_cost:
        child, gift = pair

        start_nodes.append(int(child))
        end_nodes.append(int(gift + num_children))
        if child <= last_triplet:
            capacities.append(3)
        elif child <= last_twin:
            capacities.append(2)
        else:
            capacities.append(1)

        unit_costs.append(-child_gift_cost[pair])

    # trojaczki - wezel z zapasem 3, blizniaki - wezel z zapasem 2, pojedyncze dziecko - wezel z zapasem 1
    for c in range(num_children):
        if c <= last_triplet:
            supplies.append(3)
        elif c <= last_twin:
            supplies.append(2)
        else:
            supplies.append(1)

    # zapotrzebowanie dla prezentow (ujemny zapas) = - ilosc prezentow danego typu
    for t in range(num_children, num_children + num_gifts):
        supplies.append(-gift_supply)

        # https://developers.google.com/optimization/flow/mincostflow
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()

    for i in range(0, len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], unit_costs[i])

    for i in range(0, len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    print('Uruchamiamy solver')
    min_cost_flow.SolveMaxFlowWithMinCost()

    solution = np.full((num_children), -1, dtype=np.int32)

    children_with_gift = 0
    for i in range(min_cost_flow.NumArcs()):
        if min_cost_flow.Flow(i) != 0:
            # tail = id dziecka, head - child_count = id prezentu
            solution[min_cost_flow.Tail(i)] = min_cost_flow.Head(i) - num_children
            children_with_gift += 1
    print('Dzieci z przydzielonymi prezentami: {}'.format(children_with_gift))

    # policz ile razy dany prezent został przydzielony do dziecka
    # gift_count[numer prezentu] = liczba dzieci, które go dostaną
    gift_count = np.zeros(1000, dtype=np.int32)
    for i in range(len(solution)):
        if solution[i] != -1:
            gift_count[solution[i]] += 1

    print('Przydzielenie trojaczkom i bliźniakom tych samych prezentów')

    # przydzielenie prezentów trojaczkom i bliźniakom bez prezentów
    # (prezenty mogły nie zostać przydzielone ze względu na ograniczenie liczby preferencji)
    for i in range(0, last_twin + 1):
        if solution[i] == -1:
            least_frequent_gift = np.argmin(gift_count)
            solution[i] = least_frequent_gift
            gift_count[least_frequent_gift] += 1

    # przydzielenie trojaczkom i bliźniakom tych samych prezentów (wg heurystyki - opisana w dokumentacji)
    for i in range(0, last_triplet + 1, 3):

        q = get_score(i, solution[i], child_prefs, gift_prefs) + get_score(i + 1, solution[i], child_prefs, gift_prefs) + get_score(i + 2, solution[i], child_prefs, gift_prefs)
        w = get_score(i, solution[i+1], child_prefs, gift_prefs) + get_score(i + 1, solution[i+1], child_prefs, gift_prefs) + get_score(i + 2, solution[i+1], child_prefs, gift_prefs)
        e = get_score(i, solution[i+2], child_prefs, gift_prefs) + get_score(i + 1, solution[i+2], child_prefs, gift_prefs) + get_score(i + 2, solution[i+2], child_prefs, gift_prefs)

        optimal = max(q, w, e)
        optimal_gift = -1

        if optimal == q:
            optimal_gift = solution[i]
        elif optimal == w:
            optimal_gift = solution[i + 1]
        else:
            optimal_gift = solution[i + 2]

        # oddajemy prezenty trojaczkow
        gift_count[solution[i]] -= 1
        gift_count[solution[i + 1]] -= 1
        gift_count[solution[i + 2]] -= 1
        # zabieramy 3 sztuki tego ktory przydzielimy całej trójce
        gift_count[optimal_gift] += 3

        solution[i] = optimal_gift
        solution[i + 1] = optimal_gift
        solution[i + 2] = optimal_gift

    for i in range(last_triplet + 1, last_twin + 1, 2):
        q = get_score(i, solution[i], child_prefs, gift_prefs) + get_score(i + 1, solution[i], child_prefs, gift_prefs)
        w = get_score(i, solution[i+1], child_prefs, gift_prefs) + get_score(i + 1, solution[i+1], child_prefs, gift_prefs)

        optimal = max(q, w)
        optimal_gift = -1

        if optimal == q:
            optimal_gift = solution[i]
        else:
            optimal_gift = solution[i + 1]

        gift_count[solution[i]] -= 1
        gift_count[solution[i + 1]] -= 1

        gift_count[optimal_gift] += 2

        solution[i] = optimal_gift
        solution[i + 1] = optimal_gift

    print('Przydzielanie brakujących prezentów')

    # zabieramy jedynakom nadmiarowe prezenty i przypisujemy użyty najmniej razy prezent
    # przypisujemy prezenty jedynkom bez prezentów
    for i in range(last_twin + 1, len(solution)):
        if solution[i] == -1:
            best_gift = find_best_gift(i, child_prefs, gift_prefs, gift_count)
            solution[i] = best_gift
            gift_count[best_gift] += 1
        elif gift_count[solution[i]] > num_gifts:
            former_gift = solution[i]
            best_gift = find_best_gift(i, child_prefs, gift_prefs, gift_count)
            solution[i] = best_gift
            gift_count[best_gift] += 1
            gift_count[former_gift] -= 1

    gift_count_max = gift_count.max()
    if gift_count_max > 1000:
        print('Błąd: istnieją prezenty użyte zbyt wiele razy, liczba: {}'.format(gift_count_max))
        exit()
    else:
        print('Sukces')

    print('Wyliczanie wyniku...')
    score  = evaluate_score(solution, child_prefs, gift_prefs, last_triplet, last_twin, gift_supply)
    print('Average Normalized Happiness wynosi: {:.10f}'.format(score))

    out = open('wynik_{:.10f}.csv'.format(score), 'w')
    out.write('ChildId,GiftId\n')
    for i in range(len(solution)):
        out.write(str(i) + ',' + str(solution[i]) + '\n')
    out.close()

if __name__ == '__main__':
    main()