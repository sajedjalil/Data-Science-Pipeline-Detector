import os
cmd = """tail -n +2 ../input/train.csv | awk -F "," ' { gsub("s1600/", "", $0); gsub(/"/, "", $0); print "curl", "-s", "-C", "-", "--create-dirs", $2 "/s224", "-o", "train/"$3 "/" $1".jpg"; if(FNR % 10 == 0 ) {print "echo downloaded", FNR, "files"} ; }'  | bash"""
os.system(cmd)