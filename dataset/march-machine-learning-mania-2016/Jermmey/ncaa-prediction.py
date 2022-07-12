#First time coder here. Please take a look and provide me with some things I can improve on. Thank you.

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

rscr = pd.read_csv("../input/RegularSeasonCompactResults.csv")
tcr = pd.read_csv("../input/TourneyCompactResults.csv")
teams = pd.read_csv("../input/Teams.csv")
submission = pd.read_csv("../input/SampleSubmission.csv")
seeds = pd.read_csv("../input/TourneySeeds.csv")

submit_to_test = pd.DataFrame(submission.Id.str.split('_').tolist(), columns=["Season","Team1","Team2"]).astype("int64")


teams['ELO']=0

def listtodf(list):
    dflist=pd.DataFrame(list)
    dflist=dflist.transpose()
    return dflist

def FindELO(yearfd, teams, year):
    teamslist=teams.values.T.tolist()
    yearfd=yearfd.drop(['Wscore', 'Lscore', 'Wloc', 'Numot'], axis=1)
    yeartest=yearfd[yearfd['Season']==year]
    yeartest=yeartest.values.T.tolist()
    c=0
    SeasonList=[yeartest[0][0]]*len(teamslist[0])
    teamslist.append(SeasonList)
    for n in yeartest[2]:
        wteamindex=teamslist[0].index(n)
        lteamindex=teamslist[0].index(yeartest[3][c])
        wteamelo=teamslist[2][wteamindex]
        lteamelo=teamslist[2][lteamindex]
        awinscore=1
        alosescore=0
        ewinscore=1/(1+10**((lteamelo-wteamelo)/400))
        elosescore=1-ewinscore
        nwteamelo=wteamelo+32*(awinscore-ewinscore)
        nlteamelo=lteamelo+32*(alosescore-elosescore)
        teamslist[2][wteamindex]=nwteamelo
        teamslist[2][lteamindex]=nlteamelo
        c+=1
    return teamslist

def prediction(submittotestlist, submission, elo):
    submissionlist=submission.values.T.tolist()
    submittotestlist=submittotestlist.values.T.tolist()
    c=0
    for n in submittotestlist[1]:
        if submittotestlist[0][c]==elo[3][0]:
            team1index=elo[0].index(n)
            team2index=elo[0].index(submittotestlist[2][c])
            team1ELO=elo[2][team1index]
            team2ELO=elo[2][team2index]
            submissionlist[1][c]=1/(1+10**((team2ELO-team1ELO)/400))
        c+=1
    dfsubmission=listtodf(submissionlist)
    return dfsubmission

ELO2012 = FindELO(rscr, teams, 2012)
ELO2013 = FindELO(rscr, teams, 2013)
ELO2014 = FindELO(rscr, teams, 2014)
ELO2015 = FindELO(rscr, teams, 2015)

submission=prediction(submit_to_test, submission, ELO2012)
submission=prediction(submit_to_test, submission, ELO2013)
submission=prediction(submit_to_test, submission, ELO2014)
submission=prediction(submit_to_test, submission, ELO2015)

submission.columns=['Id','Pred']
submission.to_csv('submission.csv', index=False)