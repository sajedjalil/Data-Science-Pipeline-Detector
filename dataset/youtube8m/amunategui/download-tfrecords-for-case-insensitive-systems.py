# This script works in Python 2.x and is meant to be run on your local system only. 
#
#It downloads tfrecords to your local system and will rename them so they don't
# get overwritten on case-insensitive storage systems.
#
# Currently set to download test records, you will need to change links to whichever
# records you need.

import json
import urllib2
from BeautifulSoup import BeautifulSoup 
html_page = urllib2.urlopen("http://us.data.yt8m.org/1/video_level/test/index.html")
index_data_web_page = BeautifulSoup(html_page)
 
# create list of all tfrecord addresses 
links = []
for link in index_data_web_page.findAll('a'):
    links.append(link.get('href'))

# download all records to your local system
# make each file unique by appendubg counter number to each tfrecord file names 
counter = 0
for link in links:
	print(len(links)-counter)
	full_link =  'http://us.data.yt8m.org/1/video_level/test/' + link
	response = urllib2.urlopen(full_link)
	html = response.read()
	link_split = link.split('.')
	f = open("%s%i.%s"%(link_split[0],counter,link_split[1]), "wb")
	f.write(html)
	f.close()
	counter += 1