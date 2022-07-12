import pandas as pd
import glob
print ("\t%16s\t%0s\t%s\t%s\t%s" % ( "col", "# uniques", "% of nulls", "ex value", "type" ))
for name in glob.glob("../input/*.csv"):
    df = pd.read_csv(name, sep=",")
    print (name, df.shape)
    for col in df.columns:
        u = pd.Series(df[col].unique())
        npp = df[col].isnull().mean()
        u = u[~pd.Series.isnull(u)].values
        u0 = -1 if not len(u) else u[0]
        print ("\t%16s\t%i\t%2.2f\t%s\t%s" % ( col, len(u), npp, u0, type(u0).__name__ ))
