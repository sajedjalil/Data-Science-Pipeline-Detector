# Note: Kaggle only runs Python 3, not Python 2

# We'll use the pandas library to read CSV files into dataframes
import pandas as pd
import numpy as np

types = {'ID':np.uint32, 'target':np.uint8, 'VAR_0002':np.uint16, 'VAR_0003':np.uint16, 'VAR_0532':np.uint8, 'VAR_0533':np.uint8, 'VAR_0534':np.uint8,
         'VAR_0535':np.uint8, 'VAR_0536':np.uint8, 'VAR_0537':np.uint8,'VAR_0538':np.uint8, 'VAR_0539':np.uint8, 'VAR_0540':np.uint8, 'VAR_0545':np.uint16,
         'VAR_0546':np.uint16, 'VAR_0547':np.uint16, 'VAR_0548':np.uint16, 'VAR_0549':np.uint16, 'VAR_0550':np.uint16, 'VAR_0551':np.uint16,
         'VAR_0552':np.uint8, 'VAR_0553':np.uint8, 'VAR_0554':np.uint16, 'VAR_0555':np.uint16, 'VAR_0556':np.uint16, 'VAR_0557':np.uint16,
         'VAR_0558':np.uint16, 'VAR_0559':np.uint8, 'VAR_0560':np.uint8, 'VAR_0561':np.uint16, 'VAR_0562':np.uint8, 'VAR_0563':np.uint8,
         'VAR_0564':np.uint8, 'VAR_0565':np.uint8, 'VAR_0566':np.uint8, 'VAR_0567':np.uint8, 'VAR_0568':np.uint8, 'VAR_0569':np.uint8,
         'VAR_0570':np.uint16, 'VAR_0572':np.uint8, 'VAR_0580':np.uint8, 'VAR_0581':np.uint8, 'VAR_0582':np.uint8, 'VAR_0604':np.uint8,
         'VAR_0605':np.uint8, 'VAR_0606':np.uint8, 'VAR_0617':np.uint8, 'VAR_0618':np.uint8, 'VAR_0619':np.uint8, 'VAR_0620':np.uint8,
         'VAR_0621':np.uint8, 'VAR_0622':np.uint8, 'VAR_0623':np.uint8, 'VAR_0624':np.uint8, 'VAR_0625':np.uint8, 'VAR_0626':np.uint8,
         'VAR_0627':np.uint8, 'VAR_0637':np.uint8, 'VAR_0638':np.uint8, 'VAR_0639':np.uint8, 'VAR_0640':np.uint8, 'VAR_0646':np.uint8,
         'VAR_0647':np.uint8, 'VAR_0657':np.uint8, 'VAR_0658':np.uint8, 'VAR_0662':np.uint8, 'VAR_0663':np.uint8, 'VAR_0664':np.uint8, 'VAR_0665':np.uint8,
         'VAR_0666':np.uint8,'VAR_0667':np.uint8, 'VAR_0668':np.uint8, 'VAR_0685':np.uint8, 'VAR_0686':np.uint8, 'VAR_0689':np.uint8, 'VAR_0690':np.uint8,
         'VAR_0696':np.uint8, 'VAR_0697':np.uint8, 'VAR_0703':np.uint8, 'VAR_0708':np.uint8, 'VAR_0709':np.uint8, 'VAR_0710':np.uint8, 'VAR_0711':np.uint8,
         'VAR_0712':np.uint8, 'VAR_0713':np.uint8, 'VAR_0714':np.uint8, 'VAR_0715':np.uint8, 'VAR_0716':np.uint8, 'VAR_0717':np.uint8, 'VAR_0718':np.uint8,
         'VAR_0719':np.uint8, 'VAR_0720':np.uint8, 'VAR_0721':np.uint8, 'VAR_0722':np.uint8, 'VAR_0723':np.uint8, 'VAR_0724':np.uint8, 'VAR_0725':np.uint8,
         'VAR_0726':np.uint8, 'VAR_0727':np.uint8, 'VAR_0728':np.uint8, 'VAR_0729':np.uint8, 'VAR_0730':np.uint8, 'VAR_0731':np.uint8, 'VAR_0732':np.uint8,
         'VAR_0733':np.uint8, 'VAR_0734':np.uint8, 'VAR_0735':np.uint8, 'VAR_0736':np.uint8, 'VAR_0737':np.uint8, 'VAR_0738':np.uint8, 'VAR_0739':np.uint8,
         'VAR_0740':np.uint8, 'VAR_0741':np.uint8, 'VAR_0742':np.uint8, 'VAR_0743':np.uint8, 'VAR_0744':np.uint8, 'VAR_0745':np.uint8, 'VAR_0746':np.uint8,
         'VAR_0747':np.uint8, 'VAR_0748':np.uint8, 'VAR_0749':np.uint8, 'VAR_0750':np.uint8, 'VAR_0751':np.uint8, 'VAR_0752':np.uint8, 'VAR_0753':np.uint8,
         'VAR_0754':np.uint8, 'VAR_0755':np.uint8, 'VAR_0756':np.uint8, 'VAR_0758':np.uint8, 'VAR_0759':np.uint8, 'VAR_0760':np.uint8, 'VAR_0761':np.uint8,
         'VAR_0762':np.uint8, 'VAR_0763':np.uint8, 'VAR_0764':np.uint8, 'VAR_0765':np.uint8, 'VAR_0766':np.uint8, 'VAR_0767':np.uint8, 'VAR_0768':np.uint8,
         'VAR_0769':np.uint8, 'VAR_0770':np.uint8, 'VAR_0771':np.uint8, 'VAR_0772':np.uint8, 'VAR_0773':np.uint8, 'VAR_0774':np.uint8, 'VAR_0775':np.uint8,
         'VAR_0776':np.uint8, 'VAR_0777':np.uint8, 'VAR_0778':np.uint8, 'VAR_0779':np.uint8, 'VAR_0780':np.uint8, 'VAR_0781':np.uint8, 'VAR_0782':np.uint8,
         'VAR_0783':np.uint8, 'VAR_0784':np.uint8, 'VAR_0785':np.uint8, 'VAR_0786':np.uint8, 'VAR_0787':np.uint8, 'VAR_0788':np.uint8, 'VAR_0789':np.uint8,
         'VAR_0790':np.uint8, 'VAR_0791':np.uint8, 'VAR_0792':np.uint8, 'VAR_0793':np.uint8, 'VAR_0794':np.uint8, 'VAR_0795':np.uint8, 'VAR_0796':np.uint8,
         'VAR_0797':np.uint8, 'VAR_0798':np.uint8, 'VAR_0799':np.uint8, 'VAR_0800':np.uint8, 'VAR_0801':np.uint8, 'VAR_0802':np.uint8, 'VAR_0803':np.uint8,
         'VAR_0804':np.uint8, 'VAR_0805':np.uint8, 'VAR_0806':np.uint8, 'VAR_0807':np.uint8, 'VAR_0808':np.uint8, 'VAR_0809':np.uint8, 'VAR_0810':np.uint8,
         'VAR_0812':np.uint8, 'VAR_0813':np.uint8, 'VAR_0814':np.uint8, 'VAR_0815':np.uint8, 'VAR_0816':np.uint8, 'VAR_0817':np.uint8, 'VAR_0818':np.uint8,
         'VAR_0819':np.uint8, 'VAR_0820':np.uint8, 'VAR_0821':np.uint8, 'VAR_0822':np.uint8, 'VAR_0823':np.uint8, 'VAR_0824':np.uint8, 'VAR_0825':np.uint8,
         'VAR_0826':np.uint8, 'VAR_0827':np.uint8, 'VAR_0828':np.uint8, 'VAR_0829':np.uint8, 'VAR_0830':np.uint8, 'VAR_0831':np.uint8, 'VAR_0832':np.uint8,
         'VAR_0833':np.uint8, 'VAR_0834':np.uint8, 'VAR_0835':np.uint8, 'VAR_0836':np.uint8, 'VAR_0837':np.uint8, 'VAR_0838':np.uint8, 'VAR_0839':np.uint8,
         'VAR_0841':np.uint8, 'VAR_0842':np.uint8, 'VAR_0843':np.uint8, 'VAR_0844':np.uint8, 'VAR_0845':np.uint8, 'VAR_0846':np.uint8, 'VAR_0847':np.uint8,
         'VAR_0848':np.uint8, 'VAR_0849':np.uint8, 'VAR_0850':np.uint8, 'VAR_0851':np.uint8, 'VAR_0852':np.uint8, 'VAR_0853':np.uint8, 'VAR_0854':np.uint8,
         'VAR_0855':np.uint8, 'VAR_0856':np.uint8, 'VAR_0857':np.uint8, 'VAR_0859':np.uint8, 'VAR_0877':np.uint8, 'VAR_0878':np.uint8, 'VAR_0879':np.uint8,
         'VAR_0885':np.uint8, 'VAR_0886':np.uint8, 'VAR_0911':np.uint8, 'VAR_0914':np.uint8, 'VAR_0915':np.uint8, 'VAR_0916':np.uint8, 'VAR_0923':np.uint8,
         'VAR_0924':np.uint8, 'VAR_0925':np.uint8, 'VAR_0926':np.uint8, 'VAR_0927':np.uint8, 'VAR_0940':np.uint8, 'VAR_0945':np.uint8, 'VAR_0947':np.uint8,
         'VAR_0952':np.uint8, 'VAR_0954':np.uint8, 'VAR_0959':np.uint8, 'VAR_0962':np.uint8, 'VAR_0963':np.uint8, 'VAR_0969':np.uint8, 'VAR_0973':np.uint8,
         'VAR_0974':np.uint8, 'VAR_0975':np.uint8, 'VAR_0983':np.uint8, 'VAR_0984':np.uint8, 'VAR_0985':np.uint8, 'VAR_0986':np.uint8, 'VAR_0987':np.uint8,
         'VAR_0988':np.uint8, 'VAR_0989':np.uint8, 'VAR_0990':np.uint8, 'VAR_0991':np.uint8, 'VAR_0992':np.uint8, 'VAR_0993':np.uint8, 'VAR_0994':np.uint8,
         'VAR_0995':np.uint8, 'VAR_0996':np.uint8, 'VAR_0997':np.uint8, 'VAR_0998':np.uint8, 'VAR_0999':np.uint8, 'VAR_1000':np.uint8, 'VAR_1001':np.uint8,
         'VAR_1002':np.uint8, 'VAR_1003':np.uint8, 'VAR_1004':np.uint8, 'VAR_1005':np.uint8, 'VAR_1006':np.uint8, 'VAR_1007':np.uint8, 'VAR_1008':np.uint8,
         'VAR_1009':np.uint8, 'VAR_1010':np.uint8, 'VAR_1011':np.uint8, 'VAR_1012':np.uint8, 'VAR_1013':np.uint8, 'VAR_1014':np.uint8, 'VAR_1015':np.uint8,
         'VAR_1016':np.uint8, 'VAR_1017':np.uint8, 'VAR_1018':np.uint8, 'VAR_1019':np.uint8, 'VAR_1020':np.uint8, 'VAR_1021':np.uint8, 'VAR_1022':np.uint8,
         'VAR_1023':np.uint8, 'VAR_1024':np.uint8, 'VAR_1025':np.uint8, 'VAR_1026':np.uint8, 'VAR_1027':np.uint8, 'VAR_1028':np.uint8, 'VAR_1029':np.uint8,
         'VAR_1030':np.uint8, 'VAR_1031':np.uint8, 'VAR_1032':np.uint8, 'VAR_1033':np.uint8, 'VAR_1034':np.uint8, 'VAR_1035':np.uint8, 'VAR_1036':np.uint8,
         'VAR_1037':np.uint8, 'VAR_1038':np.uint8, 'VAR_1039':np.uint8, 'VAR_1040':np.uint8, 'VAR_1041':np.uint8, 'VAR_1042':np.uint8, 'VAR_1043':np.uint8,
         'VAR_1044':np.uint8, 'VAR_1045':np.uint8, 'VAR_1046':np.uint8, 'VAR_1047':np.uint8, 'VAR_1048':np.uint8, 'VAR_1049':np.uint8, 'VAR_1050':np.uint8,
         'VAR_1051':np.uint8, 'VAR_1052':np.uint8, 'VAR_1053':np.uint8, 'VAR_1054':np.uint8, 'VAR_1055':np.uint8, 'VAR_1056':np.uint8, 'VAR_1057':np.uint8,
         'VAR_1058':np.uint8, 'VAR_1059':np.uint8, 'VAR_1060':np.uint8, 'VAR_1061':np.uint8, 'VAR_1062':np.uint8, 'VAR_1063':np.uint8, 'VAR_1064':np.uint8,
         'VAR_1065':np.uint8, 'VAR_1066':np.uint8, 'VAR_1067':np.uint8, 'VAR_1068':np.uint8, 'VAR_1069':np.uint8, 'VAR_1070':np.uint8, 'VAR_1071':np.uint8,
         'VAR_1072':np.uint8, 'VAR_1073':np.uint8, 'VAR_1080':np.uint8, 'VAR_1108':np.uint8, 'VAR_1109':np.uint8, 'VAR_1161':np.uint8, 'VAR_1162':np.uint8,
         'VAR_1163':np.uint8, 'VAR_1164':np.uint8, 'VAR_1165':np.uint8, 'VAR_1166':np.uint8, 'VAR_1167':np.uint8, 'VAR_1168':np.uint8, 'VAR_1175':np.uint8,
         'VAR_1176':np.uint8, 'VAR_1177':np.uint8, 'VAR_1178':np.uint8, 'VAR_1185':np.uint8, 'VAR_1186':np.uint8, 'VAR_1187':np.uint8, 'VAR_1188':np.uint8,
         'VAR_1189':np.uint8, 'VAR_1190':np.uint8, 'VAR_1191':np.uint8, 'VAR_1192':np.uint8, 'VAR_1193':np.uint8, 'VAR_1194':np.uint8, 'VAR_1195':np.uint8,
         'VAR_1196':np.uint8, 'VAR_1197':np.uint8, 'VAR_1198':np.uint8, 'VAR_1212':np.uint8, 'VAR_1213':np.uint8, 'VAR_1217':np.uint8, 'VAR_1218':np.uint8,
         'VAR_1224':np.uint8, 'VAR_1225':np.uint8, 'VAR_1226':np.uint8, 'VAR_1229':np.uint8, 'VAR_1230':np.uint8, 'VAR_1231':np.uint8, 'VAR_1232':np.uint8,
         'VAR_1233':np.uint8, 'VAR_1234':np.uint8, 'VAR_1235':np.uint8, 'VAR_1236':np.uint8, 'VAR_1237':np.uint8, 'VAR_1238':np.uint8, 'VAR_1239':np.uint8,
         'VAR_1267':np.uint8, 'VAR_1268':np.uint8, 'VAR_1269':np.uint8, 'VAR_1270':np.uint8, 'VAR_1271':np.uint8, 'VAR_1272':np.uint8, 'VAR_1273':np.uint8,
         'VAR_1274':np.uint8, 'VAR_1275':np.uint8, 'VAR_1276':np.uint8, 'VAR_1277':np.uint8, 'VAR_1278':np.uint8, 'VAR_1279':np.uint8, 'VAR_1280':np.uint8,
         'VAR_1281':np.uint8, 'VAR_1282':np.uint8, 'VAR_1283':np.uint8, 'VAR_1284':np.uint8, 'VAR_1285':np.uint8, 'VAR_1286':np.uint8, 'VAR_1287':np.uint8,
         'VAR_1288':np.uint8, 'VAR_1289':np.uint8, 'VAR_1290':np.uint8, 'VAR_1291':np.uint8, 'VAR_1292':np.uint8, 'VAR_1293':np.uint8, 'VAR_1294':np.uint8,
         'VAR_1295':np.uint8, 'VAR_1296':np.uint8, 'VAR_1297':np.uint8, 'VAR_1298':np.uint8, 'VAR_1299':np.uint8, 'VAR_1300':np.uint8, 'VAR_1301':np.uint8,
         'VAR_1302':np.uint8, 'VAR_1303':np.uint8, 'VAR_1304':np.uint8, 'VAR_1305':np.uint8, 'VAR_1306':np.uint8, 'VAR_1307':np.uint8, 'VAR_1338':np.uint8,
         'VAR_1339':np.uint8, 'VAR_1340':np.uint8, 'VAR_1345':np.uint8, 'VAR_1346':np.uint8, 'VAR_1347':np.uint8, 'VAR_1348':np.uint8, 'VAR_1349':np.uint8,
         'VAR_1350':np.uint8, 'VAR_1351':np.uint8, 'VAR_1352':np.uint8, 'VAR_1359':np.uint8, 'VAR_1360':np.uint8, 'VAR_1361':np.uint8, 'VAR_1362':np.uint8,
         'VAR_1363':np.uint8, 'VAR_1364':np.uint8, 'VAR_1365':np.uint8, 'VAR_1366':np.uint8, 'VAR_1367':np.uint8, 'VAR_1368':np.uint8, 'VAR_1369':np.uint8,
         'VAR_1386':np.uint8, 'VAR_1387':np.uint8, 'VAR_1388':np.uint8, 'VAR_1389':np.uint8, 'VAR_1392':np.uint8, 'VAR_1393':np.uint8, 'VAR_1394':np.uint8,
         'VAR_1395':np.uint8, 'VAR_1396':np.uint8, 'VAR_1404':np.uint8, 'VAR_1405':np.uint8, 'VAR_1406':np.uint8, 'VAR_1407':np.uint8, 'VAR_1408':np.uint8,
         'VAR_1409':np.uint8, 'VAR_1410':np.uint8, 'VAR_1411':np.uint8, 'VAR_1412':np.uint8, 'VAR_1413':np.uint8, 'VAR_1414':np.uint8, 'VAR_1415':np.uint8,
         'VAR_1416':np.uint8, 'VAR_1417':np.uint8, 'VAR_1427':np.uint8, 'VAR_1428':np.uint8, 'VAR_1429':np.uint8, 'VAR_1430':np.uint8, 'VAR_1431':np.uint8,
         'VAR_1432':np.uint8, 'VAR_1433':np.uint8, 'VAR_1434':np.uint8, 'VAR_1435':np.uint8, 'VAR_1449':np.uint8, 'VAR_1450':np.uint8, 'VAR_1456':np.uint8,
         'VAR_1457':np.uint8, 'VAR_1458':np.uint8, 'VAR_1459':np.uint8, 'VAR_1460':np.uint8, 'VAR_1461':np.uint8, 'VAR_1462':np.uint8, 'VAR_1463':np.uint8,
         'VAR_1464':np.uint8, 'VAR_1465':np.uint8, 'VAR_1466':np.uint8, 'VAR_1467':np.uint8, 'VAR_1468':np.uint8, 'VAR_1469':np.uint8, 'VAR_1470':np.uint8,
         'VAR_1471':np.uint8, 'VAR_1472':np.uint8, 'VAR_1473':np.uint8, 'VAR_1474':np.uint8, 'VAR_1475':np.uint8, 'VAR_1476':np.uint8, 'VAR_1477':np.uint8,
         'VAR_1478':np.uint8, 'VAR_1479':np.uint8, 'VAR_1480':np.uint8, 'VAR_1481':np.uint8, 'VAR_1482':np.uint8, 'VAR_1532':np.uint8, 'VAR_1533':np.uint8,
         'VAR_1534':np.uint8, 'VAR_1535':np.uint8, 'VAR_1537':np.uint8, 'VAR_1538':np.uint8, 'VAR_1539':np.uint8, 'VAR_1540':np.uint8, 'VAR_1542':np.uint8,
         'VAR_1543':np.uint8, 'VAR_1544':np.uint8, 'VAR_1545':np.uint8, 'VAR_1546':np.uint8, 'VAR_1547':np.uint8, 'VAR_1548':np.uint8, 'VAR_1549':np.uint8,
         'VAR_1551':np.uint8, 'VAR_1552':np.uint8, 'VAR_1553':np.uint8, 'VAR_1554':np.uint8, 'VAR_1556':np.uint8, 'VAR_1557':np.uint8, 'VAR_1558':np.uint8,
         'VAR_1559':np.uint8, 'VAR_1561':np.uint8, 'VAR_1562':np.uint8, 'VAR_1563':np.uint8, 'VAR_1564':np.uint8, 'VAR_1565':np.uint8, 'VAR_1566':np.uint8,
         'VAR_1567':np.uint8, 'VAR_1568':np.uint8, 'VAR_1569':np.uint8, 'VAR_1570':np.uint8, 'VAR_1571':np.uint8, 'VAR_1572':np.uint8, 'VAR_1574':np.uint8,
         'VAR_1575':np.uint8, 'VAR_1576':np.uint8, 'VAR_1577':np.uint8, 'VAR_1578':np.uint8, 'VAR_1579':np.uint8, 'VAR_1583':np.uint8, 'VAR_1584':np.uint8,
         'VAR_1585':np.uint8, 'VAR_1586':np.uint8, 'VAR_1587':np.uint8, 'VAR_1588':np.uint8, 'VAR_1589':np.uint8, 'VAR_1590':np.uint8, 'VAR_1591':np.uint8,
         'VAR_1592':np.uint8, 'VAR_1593':np.uint8, 'VAR_1594':np.uint8, 'VAR_1595':np.uint8, 'VAR_1596':np.uint8, 'VAR_1597':np.uint8, 'VAR_1598':np.uint8,
         'VAR_1599':np.uint8, 'VAR_1600':np.uint8, 'VAR_1601':np.uint8, 'VAR_1602':np.uint8, 'VAR_1603':np.uint8, 'VAR_1604':np.uint8, 'VAR_1605':np.uint8,
         'VAR_1606':np.uint8, 'VAR_1607':np.uint8, 'VAR_1608':np.uint8, 'VAR_1609':np.uint8, 'VAR_1610':np.uint8, 'VAR_1656':np.uint8, 'VAR_1657':np.uint8,
         'VAR_1658':np.uint8, 'VAR_1659':np.uint8, 'VAR_1660':np.uint8, 'VAR_1661':np.uint8, 'VAR_1662':np.uint8, 'VAR_1663':np.uint8, 'VAR_1664':np.uint8,
         'VAR_1665':np.uint8, 'VAR_1666':np.uint8, 'VAR_1667':np.uint8, 'VAR_1668':np.uint8, 'VAR_1669':np.uint8, 'VAR_1670':np.uint8, 'VAR_1671':np.uint8,
         'VAR_1672':np.uint8, 'VAR_1673':np.uint8, 'VAR_1674':np.uint8, 'VAR_1675':np.uint8, 'VAR_1676':np.uint8, 'VAR_1677':np.uint8, 'VAR_1678':np.uint8,
         'VAR_1679':np.uint8, 'VAR_1680':np.uint8, 'VAR_1681':np.uint8, 'VAR_1682':np.uint8, 'VAR_1683':np.uint8, 'VAR_1713':np.uint8, 'VAR_1714':np.uint8,
         'VAR_1721':np.uint8, 'VAR_1722':np.uint8, 'VAR_1723':np.uint8, 'VAR_1724':np.uint8, 'VAR_1725':np.uint8, 'VAR_1726':np.uint8, 'VAR_1727':np.uint8,
         'VAR_1728':np.uint8, 'VAR_1740':np.uint8, 'VAR_1741':np.uint8, 'VAR_1742':np.uint8, 'VAR_1743':np.uint8, 'VAR_1744':np.uint8, 'VAR_1745':np.uint8,
         'VAR_1746':np.uint8, 'VAR_1752':np.uint8, 'VAR_1753':np.uint8, 'VAR_1760':np.uint8, 'VAR_1761':np.uint8, 'VAR_1762':np.uint8, 'VAR_1763':np.uint8,
         'VAR_1764':np.uint8, 'VAR_1765':np.uint8, 'VAR_1766':np.uint8, 'VAR_1767':np.uint8, 'VAR_1768':np.uint8, 'VAR_1769':np.uint8, 'VAR_1770':np.uint8,
         'VAR_1771':np.uint8, 'VAR_1772':np.uint8, 'VAR_1773':np.uint8, 'VAR_1774':np.uint8, 'VAR_1775':np.uint8, 'VAR_1776':np.uint8, 'VAR_1777':np.uint8,
         'VAR_1778':np.uint8, 'VAR_1779':np.uint8, 'VAR_1780':np.uint8, 'VAR_1781':np.uint8, 'VAR_1782':np.uint8, 'VAR_1783':np.uint8, 'VAR_1784':np.uint8,
         'VAR_1785':np.uint8, 'VAR_1786':np.uint8, 'VAR_1787':np.uint8, 'VAR_1788':np.uint8, 'VAR_1789':np.uint8, 'VAR_1790':np.uint8, 'VAR_1791':np.uint8,
         'VAR_1792':np.uint8, 'VAR_1793':np.uint8, 'VAR_1794':np.uint8, 'VAR_1843':np.uint8, 'VAR_1844':np.uint8, 'VAR_1853':np.uint8, 'VAR_1854':np.uint8,
         'VAR_1855':np.uint8, 'VAR_1856':np.uint8, 'VAR_1857':np.uint8, 'VAR_1866':np.uint8, 'VAR_1867':np.uint8, 'VAR_1872':np.uint8, 'VAR_1873':np.uint8,
         'VAR_1874':np.uint8, 'VAR_1875':np.uint8, 'VAR_1876':np.uint8, 'VAR_1877':np.uint8, 'VAR_1878':np.uint8, 'VAR_1879':np.uint8, 'VAR_1880':np.uint8,
         'VAR_1881':np.uint8, 'VAR_1882':np.uint8, 'VAR_1883':np.uint8, 'VAR_1884':np.uint8, 'VAR_1885':np.uint8, 'VAR_1886':np.uint8, 'VAR_1887':np.uint8,
         'VAR_1888':np.uint8, 'VAR_1903':np.uint8, 'VAR_1904':np.uint8, 'VAR_1905':np.uint8, 'VAR_1906':np.uint8, 'VAR_1907':np.uint8, 'VAR_1908':np.uint8,
         'VAR_1909':np.uint8, 'VAR_1910':np.uint8, 'VAR_1920':np.uint8, 'VAR_1921':np.uint8, 'VAR_1925':np.uint8, 'VAR_1926':np.uint8, 'VAR_1927':np.uint8,
         'VAR_1928':np.uint16, 'VAR_1930':np.uint16}

# Read train data file:
train = pd.read_csv("../input/train.csv", dtype=types)

# Write summaries of the train and test sets to the log
print('\nSummary and memory usage of train dataset:\n')
print(train.info(memory_usage = True))
