__author__ = 'prasanna23'
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
		break
	
	#
	coupon_list_train = df.set_index('COUPON_ID_hash')
	plt.figure()
	sns.boxplot(y='GENRE_NAME', x='PRICE_RATE', data=coupon_list_train)
	plt.savefig("PRICE_RATE_by_GENRE_NAME.png")
	
	#
	visit_train = pd.read_csv(path + 'coupon_visit_train.csv', parse_dates=[1])
	user_list = pd.read_csv(path + 'user_list.csv', index_col=5)
	visit_train = visit_train.join(user_list, on='USER_ID_hash')
	
	colnames = visit_train.columns.values
	colnames[4] = 'COUPON_ID_hash'
	visit_train.columns = colnames
	visit_train = visit_train.join(coupon_list_train, on='COUPON_ID_hash', rsuffix='_coupon')
	
	plt.figure()
	sns.factorplot(x='AGE', y='GENRE_NAME', hue='SEX_ID', kind='violin', data=visit_train[visit_train.PURCHASE_FLG==1], orient='h', size=8, scale='count', split=True, cut=0)
	plt.savefig("AGE_dist_by_GENRE_NAME_and_SEX_ID.png")

    #
	visit_train.set_index('I_DATE', inplace=True)
	
	plt.figure()
	visit_train.PURCHASE_FLG.resample('W', how='count').plot()
	plt.savefig("Weekly_view_counts.png")
