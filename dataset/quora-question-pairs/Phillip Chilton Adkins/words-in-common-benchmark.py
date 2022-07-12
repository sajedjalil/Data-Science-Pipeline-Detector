# Two questions:
# 1. Can you improve on this benchmark?
# 2. Can you beat the score obtained by this example kernel?


import pandas as pd

test = pd.DataFrame.from_csv("../input/test.csv")
test["is_duplicate"] = [0.165] * len(test)
test["is_duplicate"].to_csv("count_words_benchmark.csv", header=True)