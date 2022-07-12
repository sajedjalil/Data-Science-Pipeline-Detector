fr = open("../input/sample_submission.csv", 'r')
fw = open("sub.csv",'w')
fw.writelines("ParcelId,201610,201611,201612,201710,201711,201712"+"\n")
fr.readline() #Skip the header
lines=fr.readlines()
for line in lines:
	id=line.split(',')[0]
	datas=[id]
	for i in range(6):
		datas.append("0.010816")# 0.010816 is the mean of training logerror
	out=','.join(datas)
	fw.writelines(out+"\n")
fw.close()
fr.close()