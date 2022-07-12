import csv
d={}
a=[]
x=[]
with open("../input/train.csv",encoding="ascii", errors="surrogateescape") as f:
    d= csv.DictReader(f)
    for line in d:
        #print (line)
        a.append([line["Latitude"],line["Longitude"],line["AddressNumberAndStreet"]])
f.close
with open("../input/spray.csv",encoding="ascii", errors="surrogateescape") as f:
    d= csv.DictReader(f)
    for line in d:
        #print (line)
        x.append([line["Latitude"],line["Longitude"],line["Date"]])
f.close
#FIRST 50 LOCATIONS ONLY FOR PERFORMANCE - YOU CAN CHANGE IN FORK
a=a[:50]
x=x[:50]
#HTML Comments Only
h=[]
lat=a[0][0]
lon=a[0][1]
h.append("<html><head></head><body><script type='text/javascript' src='https://dev.virtualearth.net/mapcontrol/mapcontrol.ashx?v=6.3&s=1'></script><script type='text/javascript'>var map = null;var Center = new VELatLong(" + lat + " ," + lon + ");var pinPoint = null;var pinPixel = null;function GetMap(){map = new VEMap('myMap');map.LoadMap(Center, 7, VEMapStyle.Road, false, VEMapMode.Mode2D, true, 1);AddPin();}function AddPin(){")
i=1
for b in a:
    lat=b[0]
    lon=b[1]
    c=b[2]
    h.append("var pin = new VEPushpin(" + str(i) + ", new VELatLong(" + lat + " ," + lon + "), null,'" + c + "');  map.AddPushpin(pin);")
    i+=1
for b in x:
    lat=b[0]
    lon=b[1]
    h.append("var pin = new VEPushpin(" + str(i) + ", new VELatLong(" + lat + " ," + lon + "), null,'Spray');  map.AddPushpin(pin);")
    i+=1
h.append("}</script><script>document.addEventListener('DOMContentLoaded', function(e){window.setTimeout(GetMap, 500);});</script><div id='myMap' style='position:relative; width:700px; height:650px;border:1px solid #000000;'></div></body></html>")
with open("output.html","w",encoding="ascii", errors="surrogateescape") as f:
    for i in range(len(h)):
        f.write(h[i]+"\n")
f.close()
#End Comments

