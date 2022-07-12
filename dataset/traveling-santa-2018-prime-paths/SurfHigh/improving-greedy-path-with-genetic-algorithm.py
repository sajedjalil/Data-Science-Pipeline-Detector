import numpy as np, random, operator, pandas as pd,  time, os
import multiprocessing as mp

### mark prime numbers
def is_prime(n):
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n < 2: return False
    for i in range(3, int(n ** 0.5) + 1, 2):  # only odd numbers
        if n % i == 0:
            return False

    return True


### calculate distance of given path
def calc_path_dist(path):

    len_path = len(path)
    np_xy_path = np.array(cities.iloc[path][['X','Y']].values)
    np_non_prime = np.array(~cities.iloc[path][['is_prime']].values)[:,0]
    np_is_10th_step = np.array(([0] * 9 + [1]) * (len_path // 10) + [0] * (len_path % 10))
    np_net_dist_path = np.sum((np_xy_path[0:-1] - np_xy_path[1:]) ** 2, axis=1) ** 0.5
    np_dist_path = np.sum(np_net_dist_path * (1.0 + 0.1 * np_non_prime[:-1] * np_is_10th_step[:-1]))

    return np_dist_path


##### Genetic algorithm functions ##########
### rank routes, fitness is 1 divided by distance of route
def rankRoutes(population):

    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = 1/calc_path_dist(population[i]) ### fitness is defined as 1 divided by distance

    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)

### selected routes for mating pool
def selection(popRanked, eliteSize):

    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break

    return selectionResults

### create mating pool
def matingPool(population, selectionResults):

    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

### orederd crossover breeding
def breed_ox(parent1, parent2):

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    while geneA == geneB: ## to make sure it's not the same index
        geneA = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    childP1 = parent1[startGene: endGene]
    childP2 = [item for item in parent2 if item not in childP1]
    child = np.concatenate((childP1, childP2))

    return child

### modified crossover breeding
def breed_mx(parent1, parent2):


    geneA = int(random.random() * len(parent1))
    while geneA == 0:
        geneA = int(random.random() * len(parent1))

    childP1 = parent1[: geneA]
    childP2 = [item for item in parent2 if item not in childP1]
    child = np.concatenate((childP1, childP2))

    return child


### breed population
def breedPopulation(matingpool, eliteSize):

    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        # child = breed_ox(pool[i], pool[len(matingpool) - i - 1])
        child = breed_mx(pool[i], pool[len(matingpool) - i - 1])
        # child = breed(matingpool[0], pool[i]) ## Gil's line
        children.append(child)

    return children

### insert mutations to individual route
def mutate(individual, mutationRate):

    length = len(individual)
    for swapped in range(length):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1

    return individual

### insert mutations to population
def mutatePopulation(population, mutationRate, eliteSize):

    mutatedPop = []

    for ind in range(0, len(population)): ### create mutation just if it is not the elite
        if ind >= eliteSize:
            mutatedInd = mutate(population[ind], mutationRate)
            mutatedPop.append(mutatedInd)
        else:
            mutatedPop.append(population[ind])

    return mutatedPop

### create next generation's population
def nextGeneration(currentGen, eliteSize, mutationRate):

    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate, eliteSize)
    return nextGeneration


### run the genetic algorithm
def geneticAlgorithmMP(work_queue, result_queue): #population, popSize, eliteSize, mutationRate, generations):

    while not work_queue.empty():
        ### get data and present it
        current_process = mp.current_process().pid
        data = work_queue.get()
        path__ = data['path']
        dist = calc_path_dist(path__)
        print('current_process {} - Slicing [{}:{}]. Initial distance {}'.format(current_process, data['start_idx'], data['end_idx'], dist))

        pop = [path__] * popSize ## initialize generation 0 population
        progress = []
        best_path = []
        for i in range(0, generations):
            pop = nextGeneration(pop, eliteSize, mutationRate)
            rr = np.array(rankRoutes(pop))
            best_path = pop[int(rr[0,0])]
            # print('top ranked {} {}'.format(rr[0][0], 1/rr[0][1]))
            # print('Distances {}'.format( 1/rr[:5,1]))
            progress.append(1 / rr[0][1])
            # if i % 100 == 0:  ## print progress
            #     print('{} Generation {}, top ranked {}, distnace {}, mean {}'.format(
            #         time.strftime('%x %X'), i, rr[0][0], progress[-1], np.mean(1/rr[eliteSize:,1]))
            #     )

        print('current_process {} - Slicing [{}:{}]. Final distance {}'.format(current_process, data['start_idx'], data['end_idx'], progress[-1]))
        result_queue.put({'start_idx':data['start_idx'],
                          'end_idx':data['end_idx'],
                          'best_path':best_path})

### globals
popSize=50
eliteSize=5
mutationRate=0.01
generations=1500
### intialize cityList dataframe
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv')
cities['is_prime'] = cities.CityId.apply(is_prime)



if __name__ == '__main__':

    startedd = time.time()

    ### load greedy solution path
    greedy = pd.read_csv('../input/greedy-path/greedy_submission.csv')

    #### trying to improve greedy result algorithm ################
    ### using multiprocessing
    manager = mp.Manager()
    # now using the manager, create our shared data structures
    result_queue = manager.Queue()
    work_queue = manager.Queue()

    ### intialize work queue
    step_size = 100
    length = greedy.shape[0]
    i = 1
    while i < 400: ### change 400 to length for full run
        start_idx = i
        end_idx = min(i+step_size, length-1)
        work_queue.put({'start_idx':start_idx,
                        'end_idx': end_idx,
                        'path': greedy[start_idx:end_idx]['Path'].values})
        i += step_size

    pool = []
    # Create new processes
    for i in range(6): ### number of processes running in parallel
        p = mp.Process(target=geneticAlgorithmMP, args=(work_queue, result_queue,))
        p.start()
        pool.append(p)

    # Wait for all processes to complete
    for p in pool:
        p.join()

    ### collect data from result queue
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    new_path = [0] * length
    for r in results:
        new_path[r['start_idx']:r['end_idx']] = r['best_path']

    old_dist = calc_path_dist(greedy['Path'].values)
    new_dist = calc_path_dist(new_path)
    print('old {}, new {}'.format(old_dist, new_dist))

    ### save to file
    df = pd.DataFrame(new_path, columns=['Path'])
    df.to_csv('GA_submission.csv', index=False)

    finishedd = time.time()
    print('Total time: ' + str(finishedd - startedd))

