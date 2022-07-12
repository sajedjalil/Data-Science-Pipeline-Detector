__author__ = 'prasanna23'
import pandas as pd

def encode_text(data):
	# This function converts the data into type 'str'.
    try:
        text = data
    except:
        text = data.encode("UTF-8")
    return text

if __name__ == "__main__":
	
	path = '../input/' # Enter the path to home folder containing the input files.
	df = pd.read_excel(path + 'documentation/CAPSULE_TEXT_Translation.xlsx',skiprows=5)
	
	# dictionary for the column 'CAPSULE_TEXT'.
	k = [ encode_text(x) for x in df['CAPSULE_TEXT'] ] 
	v = [ encode_text(x) for x in df['English Translation'] ]
	capsuleText = dict(zip(k,v))
	
	# dictionary for the column 'GENRE_NAME'.
	k = df['CAPSULE_TEXT.1'].dropna()
	v = df['English Translation.1'].dropna()
	k = [ encode_text(x) for x in k ]
	v = [ encode_text(x) for x in v ]
	genreName = dict(zip(k,v))
	
	#translating the columns from japanese to english.
	files = ['coupon_list_train','coupon_list_test']
	
	for f in files:
		df = pd.read_csv(path + f + '.csv')
		df['CAPSULE_TEXT'] = [ capsuleText[encode_text(x)] for x in df['CAPSULE_TEXT'] ]
		df['GENRE_NAME'] = [ genreName[encode_text(x)] for x in df['GENRE_NAME'] ]
		df.to_csv(f + '_translated.csv',index = False)