from concorde.tsp import TSPSolver
import numpy as np
import pandas as pd
cities = pd.read_csv('../input/cities.csv')
solver = TSPSolver.from_data(cities.X, cities.Y, norm="EUC_2D")
tour_data = solver.solve(time_bound = 77.0, verbose = True, random_seed = 2018)
pd.DataFrame({'Path': np.append(tour_data.tour,[0])}).to_csv('submission.csv', index=False)
