# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



from sklearn import datasets, linear_model
import statsmodels.formula.api as smf
import time


def predict_last(N, inputfilepath):

    print("runnig last number prediction")

    data = pd.read_csv(inputfilepath)
    clf = linear_model.LinearRegression()
    results = []
    Ids = []


    for i in range(len(data)):
        Id =  data["Id"][i]
        seq = np.fromstring(data['Sequence'][i], dtype=int, sep=",")
        number_of_points = N

        if 8 < len(seq) < number_of_points + 4:
            # issues += 1
            # continue
            number_of_points = len(seq) - 5

        if len(seq) < 9:
            # issues_5 += 1
            # # print(train)
            # # print("index number = ", i )
            # # input("Press Enter to continue...")
            number_of_points = len(seq)-1

        if len(seq) < 2:
            Ids.append(Id)
            results.append(seq[-1])
            continue

        clf.fit([seq[i:i + number_of_points]
                 for i in range(len(seq) - number_of_points)], seq[number_of_points:])

        result = clf.predict([seq[-number_of_points:]])[0]


        results.append(int(round(result)))
        Ids.append(Id)
        
        if i % 10000 ==0:
                print("Done with {:2.4} % ".format((i*100)/len(data)) )



    # print(results[0:20])

    print("saving the csv file")

    import csv
    df = pd.DataFrame({'"Id"':Ids,"\"Last\"":results})
    df.to_csv('out.csv', index=False, quoting=csv.QUOTE_NONE)


def test_prediction(inputfilepath):

    clf = linear_model.LinearRegression()

    data = pd.read_csv(inputfilepath)
    # print(len(data))


    for N in range(6, 16):

        corrects = 0
        issues = 0
        issues_4 = 0
        issues_2 = 0
        print("number_of_points = ", N)


        for i in range(len(data)):
            train = np.fromstring(data['Sequence'][i], dtype=int, sep=",")

            number_of_points = N

            last_integer = train[-1]
            seq = train[:-1]
            

        #train = [1,1,2,3,5,8,13,21,34,55,89,144]

        #train = [1,3,6,10,15,21,28,36]

            if 3 < len(seq) < number_of_points + 4:
                issues += 1
                # continue
                number_of_points = len(seq) - 3

            if 1 < len(seq) < 4:
                issues_4 += 1
                # # print(train)
                # # print("index number = ", i )
                # # input("Press Enter to continue...")
                # Ids.append(Id)
                # results.append(seq[-1])
                number_of_points = len(seq)-1
                # print(len(seq))

            if len(seq) < 2:
                issues_2 += 1
                if  len(seq)==1 and seq[-1] == last_integer:
                    corrects += 1
                continue




            X = [seq[i:i + number_of_points] 
                     for i in range(len(seq) - number_of_points)] 


            Y = seq[number_of_points:]

            # print(X)
            # print(Y)

            if i % 10000 ==0:
                print("Done with {:2.4} % ".format((i*100)/len(data)) )

            clf.fit(X, Y)

            results = clf.predict( [seq[-number_of_points:] ] )[0]
            score = clf.score([seq[i:i + number_of_points]
                               for i in range(len(seq) - number_of_points)], seq[number_of_points:])
        #
            if int(round(results)) == last_integer:
                corrects += 1

        print("Number of issues = ", issues)
        print("length < 4 = ", issues_4)
        print("length < 2 = ", issues_2)
        print("Percentage = ", corrects / len(data), "\n")


def main():

    start = time.time()
    
    predict_last(12,"../input/test.csv")
    # test_prediction("test.csv")


    end = time.time()
    print("time = {} s".format(end - start) )


if __name__ == "__main__":
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
