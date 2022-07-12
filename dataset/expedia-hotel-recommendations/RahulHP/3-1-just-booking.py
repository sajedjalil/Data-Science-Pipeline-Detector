from collections import defaultdict
from datetime import datetime
user_srchdistance_dict = defaultdict(int)
user_srchid_dict = defaultdict(int)

f=open("../input/train.csv", "r")
f.readline()


linecount=0
while 1:
	line = f.readline().strip()
	linecount+=1
	if line=='':
		break
	if linecount % 100000 == 0:
		print('Write {} lines...'.format(linecount))
	arr = line.split(",")
	is_booking = float(arr[18])
	if is_booking==0:
		continue
	user_id = arr[7]
	orig_destination_distance = arr[6]
	srch_destination_id = arr[16]
	hotel_cluster = arr[23]
	if user_srchdistance_dict[(user_id,orig_destination_distance)]==0:
		user_srchdistance_dict[(user_id,orig_destination_distance)]=hotel_cluster
	if user_srchid_dict[(user_id,srch_destination_id)]==0:
		user_srchid_dict[(user_id,srch_destination_id)]=hotel_cluster
f.close()




now = datetime.now()
path = 'submission_3_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
out = open(path, "w")
f = open("../input/test.csv", "r")
f.readline()
out.write("id,hotel_cluster\n")
while 1:
	line = f.readline().strip()
	total += 1

	if total % 100000 == 0:
		print('Write {} lines...'.format(total))

	if line == '':
		break

	arr = line.split(",")
	id = arr[0]
	user_location_city = arr[6]
	orig_destination_distance = arr[7]
	user_id = arr[8]
	srch_destination_id = arr[17]
	srch_destination_type_id = arr[18]
	hotel_country = arr[20]
	hotel_market = arr[21]
	out.write(str(id) + ',')
	filled = []

	if (user_id,orig_destination_distance) in user_srchdistance_dict:
		top = user_srchdistance_dict[(user_id,orig_destination_distance)]
		filled.append(top)
		out.write(' '+str(top))

	if (user_id,srch_destination_id) in user_srchid_dict:
		top = user_srchid_dict[(user_id,srch_destination_id)]
		if top not in filled:
			filled.append(top)
			out.write(' '+str(top))
	out.write('\n')

f.close()

