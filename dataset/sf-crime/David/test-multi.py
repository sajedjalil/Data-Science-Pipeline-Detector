import numpy as np
import pandas as pd

test_df = pd.read_csv('../input/test.csv')

test_rows = test_df.shape[0]
test_columns = test_df.shape[1]
	
print('Rows: %d, Columns: %d' % (test_rows,test_columns))

unqiue_crimes = 40;
out = open('sample_submission_1.csv', "w")
out.write("Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS\n")
for num in range(0,test_rows):
    out.write('%d'% num)
    for i in range(0,39):
        out.write(',%6f'% (1))
    out.write('\n')
    if (num%100000==0):
        print("Row %d/%d\n" % (num, test_rows))
print("Row %d/%d" % (test_rows, test_rows))
out.close()