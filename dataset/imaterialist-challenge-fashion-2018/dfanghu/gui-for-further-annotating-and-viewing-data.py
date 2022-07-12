#!/user/bin/env python
#
# > python ./gui.py
#
# Require folder prepared as
# |--gui.py
# |--validation.json
# |--validation/
#    |--1.jpg
#    |--2.jpg
#    |--3.jpg
# 

import os
import glob
import PIL.Image
import PIL.ImageTk
from tkinter import *
import json

window = Tk()
labelframes = []
labelframes.append(LabelFrame(window))
newlabelname = StringVar()
alllabelnames = []
imgfolderpath = StringVar()
idxImg = 0
logfile = os.path.abspath('log')
jsondata = None

def log(msg):
    global logfile
    with open(logfile,'a') as f:
        f.write(msg)

def newLabel(name:str):
    if name in alllabelnames: return
    if len(alllabelnames) % 10 == 0:
        labelframes.append(LabelFrame(window))
        labelframes[-1].pack()
    labelframe = labelframes[-1]
    lb = Label(labelframe, text=name, bg='black', fg='white', font=('KaiTi',16,'bold'))
    lb.bind('<Button-1>', labelLeftClick)
    lb.bind('<Button-3>', labelRightClick)
    lb.pack(padx=5, pady=10, side=LEFT)
    alllabelnames.append(name)

def newLabel_event(event):
    newLabel(newlabelname.get())

def crd2str(x,y):
    return str(x) + ',' + str(y)

def leftClick(event):
    out = 'c'+crd2str(event.x,event.y)
    log(out)

def leftRelease(event):
    out = 'r'+crd2str(event.x,event.y)
    log(out)

def labelLeftClick(event):
    event.widget.config(fg='red')
    log('\ny'+event.widget.cget('text'))

def labelRightClick(event):
    event.widget.config(fg='grey')
    log('\nw'+event.widget.cget('text'))

def now():
    import time
    return time.strftime('%Y-%m-%d_%H:%M:%S',time.localtime())

def begin(imgfolderpath):
    global log
    os.chdir(imgfolderpath)
    global imgs
    imgs = os.listdir()
    imgs.sort()
    log("Session begins at " + now() + ' with data in ' + os.path.abspath('.'))
    global jsondata
    with open('../{0}.json'.format(os.path.basename(os.path.abspath('.'))),'rb') as f:
        jsondata = json.load(f)

def cdImgFolder(event):
    os.chdir(imgfolderpath.get())
    global idxImg
    global imgs
    imgs = os.listdir()
    idxImg = 0
    setImage(imgs[0])
    begin('.')

imgfolder = Entry(window, textvariable=imgfolderpath)
imgfolder.bind('<Return>',cdImgFolder)
imgfolder.pack(padx=5,pady=10)
mainImgLabel = Label(window)
mainImgLabel.bind('<Button-1>', leftClick)
mainImgLabel.bind('<ButtonRelease-1>', leftRelease)
attributeLabel = Label(window)
imgIdLabel = Label(window)

def getImg(imgpath):
    return PIL.ImageTk.PhotoImage(PIL.Image.open(imgpath))

def setImage(imgpath):
    global imgIdLabel
    imgIdLabel.configure(text=imgpath)
    imgIdLabel.pack()
    global attributeLabel
    attributeLabel.configure(text=', '.join(getAttributes(imgpath)))
    attributeLabel.pack()
    global mainImgLabel
    photoimage = getImg(imgpath)
    mainImgLabel.image = photoimage
    mainImgLabel.configure(image=photoimage)
    mainImgLabel.pack()
    imgname = os.path.basename(imgpath)
    log(imgname)

def preImage(event=None):
    global idxImg
    if idxImg <= 0: return
    global mainImgLabel
    idxImg = idxImg - 1
    setImage(imgs[idxImg])

def nxtImage(event=None):
    global idxImg
    if idxImg >= len(imgs) - 1: return
    global mainImgLabel
    idxImg = idxImg + 1
    setImage(imgs[idxImg])

	
def getAttributes(imgpath):
    jImg = int(os.path.splitext(imgpath)[0]) - 1
    global jsondata
    attr = jsondata['annotations'][jImg]['labelId']
    return attr

pre = Button(window,text='Prev<PageUp>',command=preImage)
pre.pack()
begin('validation')
setImage(imgs[0])

nxt = Button(window,text='Next<PageDown>',command=nxtImage)
nxt.pack()

for lb in [
'上半身Upper-Body',
'下半身Lower-Body',
'头Head',
'额Forehead',
'眉Eyebrow(s)',
'眼Eye(s)',
'耳Ear',
'鼻Nose',
'脸颊Cheek(s)',
'嘴Mouse',
'下巴Chin',
'颈Neck',
'领Collar',
'胸Chest',
'乳Breast(s)',
'肩Shoulder(s)',
'腋Armpit ',
'肘Elbow',
'手Hand(s)',
'腕Wrist(s)',
'袖口Cuff(s)',
'侧腰Side waist',
'后腰Lower back',
'上背Upper back',
'腹Abdomen',
'骨盆Pelvis',
'臀Butt',
'裤腰/裙腰Waist line',
'裤门襟Pants fly opening',
'裤裆Crotch',
'裤管口Trouser legs opening(s)',
'裙拉链Skirt zipper)',
'裙边Skirt hem',
'大腿面Lap(s)',
'膝Knee(s)',
'小腿Lowerleg(s)',
'脚Foot(s)',
'踝Ankle(s)',
'脚跟Heel(s)',
'前脚掌Fore foot(s)',
'脚趾Toes']: newLabel(lb)

namenew = Entry(window, textvariable=newlabelname)
namenew.bind('<Return>', newLabel_event)
namenew.pack(padx=5,pady=10)

window.bind('<Prior>',preImage)
window.bind('<Next>', nxtImage)
window.mainloop()
