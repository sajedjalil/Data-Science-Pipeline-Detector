
import pandas as pd 

train = pd.read_csv("../input/train.csv").fillna("")
test  = pd.read_csv("../input/test.csv").fillna("")

submission_top_rows = pd.DataFrame({"id":         [3, 6, 9, 11, 12, 14, 15, 16, 18, 19],
                                    "prediction": [4, 4, 2,  3,  1,  3,  2,  1,  2,  1]})

with open("output.html", "wb") as f:
    f.write("<body>".encode("utf-8"))
    f.write("<style>table, th, td {border-collapse: collapse;}</style>".encode("utf-8"))
    f.write(("<p><b>train.csv</b> top 10 rows (out of %d total rows)</p>" % len(train)).encode("utf-8"))
    f.write(train[0:10].to_html(index=False).encode("utf-8"))
    f.write(("<p><b>test.csv</b> top 10 rows (out of %d total rows)</p>" % len(test)).encode("utf-8"))
    f.write(test[0:10].to_html(index=False).encode("utf-8"))
    f.write("<p><b>submission.csv sample</b></p>".encode("utf-8"))
    f.write(submission_top_rows.to_html(index=False).encode("utf-8"))
    f.write("</body>".encode("utf-8"))
