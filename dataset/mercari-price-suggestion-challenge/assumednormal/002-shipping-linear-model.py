import pandas as pd
import statsmodels.formula.api as smf

train_df = pd.read_csv("../input/train.tsv", sep="\t")
test_df = pd.read_csv("../input/test.tsv", sep="\t")

model = smf.ols("price ~ shipping", data=train_df).fit()

submission_df = test_df[["test_id"]].copy()
submission_df["price"] = model.predict(test_df)

submission_df.to_csv("002_shipping_linear_model_submission.csv", index=False)