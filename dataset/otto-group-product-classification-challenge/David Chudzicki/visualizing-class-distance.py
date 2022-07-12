import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

x = np.linspace(0,1)

for i in range(10):
    y = i*x
    plt.scatter(x, y)
    plt.ylim([0,10])
    plt.savefig('image%d.png' % i)
    plt.clf()
 
# Doesn't work b/c we don't have imagemagick:   
#os.system("convert   -delay 1   -loop 0   *.png   animation.gif")

# Modified html from http://www.java2s.com/Code/JavaScript/HTML/ImageAnimation.htm
html = '''<html>
<head>
<script language="JavaScript">
<!--
 var alternate = 0;
 var timerId;
 var numImages = 10
 var i = 0
 function startAnimation() {
    document.images[0].src = "image" + i + ".png";     // Update image
    i = (i+1) % numImages
    timerId = setTimeout("startAnimation()", 1000); // 1 second update
 }
 window.onload = startAnimation
//-->
</script>
</head>
<body>
<img src="image0.png"></body>
</body>
</html>'''

with open("output.html", "w") as output_file:
    output_file.write(html)