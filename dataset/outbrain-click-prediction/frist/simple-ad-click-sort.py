import gzip
import pandas as pd


def intjoin(iterable, sep=" "):
    """Join list of non-str."""
    return sep.join((str(i) for i in iterable))


def main():
    """Main program entry point."""
    csv = pd.read_csv("../input/clicks_train.csv")
    clicks = csv.groupby(["ad_id"]).clicked.sum()

    csv = pd.merge(
        pd.read_csv("../input/clicks_test.csv"),
        pd.DataFrame(clicks),
        left_on="ad_id",
        right_index=True,
        how="left",
        sort=False)
    csv.clicked = csv.clicked.fillna(0)

    result = csv.sort_values("clicked", ascending=False).groupby("display_id").ad_id.agg(intjoin)
    result.name = "ad_id"

    with open("submission.csv", "wt") as fo:
        result.to_csv(fo, header=True)

if __name__ == "__main__":
    main()
