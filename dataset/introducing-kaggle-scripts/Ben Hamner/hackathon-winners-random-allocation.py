import numpy as np

teams = {"baseball":    {"mortality": 100.0},
         "warming\t":   {"baseball":  100.0},
         "mortality":   {"baseball":  100.0},
         "education":   {"baseball":  100.0},
         "clusterduck": {"mortality": 45.0,
                         "baseball":  30.0,
                         "education": 24.0,
                         "warming\t": 1.0}}

num_draws = 100000

successes = dict((key, 0.0) for key in teams)

for i in range(num_draws):
    draw =  dict((key, np.random.randint(100)) for key in teams)
    score = dict((team,
                  draw[team]*1.00000001 + # this simulates the tiebreaker 
                  np.sum([teams[team][investment]/100.0*draw[investment] for investment in teams[team]]))
                 for team in teams)
    winner = [team for team in sorted(score, key=score.get, reverse=True)][0]
    successes[winner] += 1.0

print("Team\t\tWinPercentage")
for team in sorted(successes, key=successes.get, reverse=True):
    print("%s\t%0.2f%%" % (team, successes[team]/num_draws*100.0))
