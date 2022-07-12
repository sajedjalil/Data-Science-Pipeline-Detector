
# Takes 48 seconds
import kagglegym

env = kagglegym.make()
observation = env.reset()

products = observation.train['id'].unique()
avg = observation.train[['id', 'y']].groupby(['id']).mean()
while True:
    target = observation.target
    index = target['id']
    avg.loc[index, 'y']  # <-- Submit fails on this line
    observation, reward, done, info = env.step(target)
    if done:
        break


print(info)
