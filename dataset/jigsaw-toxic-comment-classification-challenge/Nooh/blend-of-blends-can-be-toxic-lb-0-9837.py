import pandas as pd

sub1 = '../input/toxic-avenger/submission.csv'
sub2 = '../input/minimal-lstm-nb-svm-baseline-ensemble/submission.csv'
sub3 = '../input/toxic-hight-of-blending/hight_of_blending.csv'

# SUB1 = https://www.kaggle.com/the1owl/toxic-avenger (Ver 25)
# SUB2 = https://www.kaggle.com/jhoward/minimal-lstm-nb-svm-baseline-ensemble (Ver 10)
# SUB3 = https://www.kaggle.com/prashantkikani/toxic-hight-of-blending (Ver 7)

###### PLEASE NOTE ######
# Before just forking on downloading and submitting anything, please be aware that this is NOT AT ALL a standard procedure that I'm following here.
# I ususally blend submission based on weights that I get from OOF prediction with sort of a random sweep. Its more of a Greedy approach.
# But in this example its just a random blend of blends' of blends. Lol. Weights are a rough estimations of what I've done it my own ensemble.
# Blended_out_2 = 0.9837

blend1="Blended_out_1.csv"

ff=open(sub1, "r")
ff_pred=open(sub2, "r")
fs=open(blend1,"w")
fs.write(ff.readline())
ff_pred.readline()

s=0
for line in ff:
    splits=line.replace("\n","").split(",")
    ids=splits[0]
    preds=[]
    for j in range (1,7):
        preds.append(float(splits[j]))
        
        
    pre_line_splits=ff_pred.readline().replace("\n","").split(",")
    for j in range (1,7):
        preds[j-1]=(preds[j-1]*0.60+ float(pre_line_splits[j])*0.40)
        
    fs.write(ids)
    for j in range(6):
        fs.write( "," +str(preds[j] ))
    fs.write("\n")
    s+=1
    
ff.close() 
ff_pred.close()
fs.close()    
   
print ("Parking 1st Blend")


blend2="Blended_out_2.csv"

ff=open(blend1, "r")
ff_pred=open(sub3, "r")
fs=open(blend2,"w")
fs.write(ff.readline())
ff_pred.readline()

s=0
for line in ff:
    splits=line.replace("\n","").split(",")
    ids=splits[0]
    preds=[]
    for j in range (1,7):
        preds.append(float(splits[j]))
        
        
    pre_line_splits=ff_pred.readline().replace("\n","").split(",")
    for j in range (1,7):
        preds[j-1]=(preds[j-1]*0.385+ float(pre_line_splits[j])*0.615)
        
    fs.write(ids)
    for j in range(6):
        fs.write( "," +str(preds[j] ))
    fs.write("\n")
    s+=1
    
ff.close() 
ff_pred.close()
fs.close()    
   
print ("Parking 2nd Blend")
