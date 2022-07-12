import pandas as pd


df_cities = pd.read_csv('../input/cities.csv')
df_submit = pd.read_csv('../input/sample_submission.csv')

xy = df_cities[['X', 'Y']].values
li = df_submit['Path'].values.tolist()


def primes(N):
    f    = [True] * (N+1)
    f[0] = False
    f[1] = False
    for p in range(2, int(N**0.5)+1):
        if f[p] == True:
            for P in range(2*p, N+1, p):
                f[P] = False
    return [p for p in range(N+1) if f[p] == True]


lp = primes(len(df_cities))

score = 0

for i in range(len(li)-1):
    step = sum((xy[li[i]] - xy[li[i+1]]) ** 2) ** 0.5
    if (i+1) % 10 == 0:
        if (li[i] in lp) == False:
            score += step * 1.1
        else:
            score += step
    else:
        score += step

print(score)
