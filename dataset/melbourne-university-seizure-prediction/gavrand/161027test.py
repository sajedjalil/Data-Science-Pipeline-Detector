import os

# Any results you write to the current directory are saved as output.

filenames=["1_97.mat","1_110.mat","1_132.mat","1_155.mat","1_211.mat","1_234.mat","1_252.mat","1_258.mat","1_294.mat","1_523.mat","1_532.mat","1_538.mat","1_539.mat","1_636.mat","1_640.mat","1_642.mat","1_659.mat","1_672.mat","1_770.mat","1_783.mat","1_802.mat","1_825.mat","1_874.mat","1_896.mat","1_905.mat","1_933.mat","1_958.mat","1_969.mat","1_981.mat","1_1027.mat","1_1120.mat","1_1139.mat","1_1147.mat","1_1174.mat","1_1199.mat","1_1205.mat","1_1210.mat","1_1254.mat","1_1273.mat","1_1340.mat","1_1347.mat","1_1365.mat","1_1483.mat","1_1491.mat","1_1499.mat","1_1518.mat","1_1545.mat","2_9.mat","2_101.mat","2_131.mat","2_153.mat","2_193.mat","2_204.mat","2_307.mat","2_330.mat","2_349.mat","2_367.mat","2_383.mat","2_632.mat","2_732.mat",
"2_849.mat","2_894.mat","2_935.mat","2_946.mat","2_1013.mat","2_1542.mat","2_1624.mat","2_1669.mat","2_1788.mat","2_1841.mat","2_1901.mat","2_1940.mat","2_2010.mat","2_2036.mat","2_2037.mat","2_2059.mat","3_1158.mat","3_1404.mat","3_1635.mat","3_1821.mat"]


for fl in filenames:
    arr = fl.split("_")
    #patient = int(f1[0])
    print(str(os.path.getsize("../input/test_" + str(arr[0]) + "/"+str(fl))) +"   " +str(fl))
   