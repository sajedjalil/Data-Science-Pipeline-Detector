#This is the image padding application from the excellent kernel :
#https://www.kaggle.com/chewzy/eda-weird-images-with-new-updates
#Thanks for the author: Toukenize
#Notice that perhaps because of the Kaggle output size restriction, this kernel will not be successfully
#generate the output if you config a large img_size
#This is just a demo, I hate to say it but this probably will not impove your LB rank

import os
print(os.listdir("../input"))
os.system('mkdir imet-padded;mkdir imet-padded/test/;mkdir imet-padded/train/')


import PIL
from PIL import Image
from PIL import ImageOps
def pad_img(input_dir,output_dir,img_size):
    count = 0
    for name in sorted(os.listdir(input_dir)):
        print(name)
        img = PIL.Image.open(input_dir+name)
#print(img.size[0])
#print(img.size[1])
        if img.size[0] < img.size[1]:
            w_resized = int(img.size[0] * img_size / img.size[1])
            resized = img.resize((w_resized ,img_size))
            pad_width = img_size - w_resized
            padding = (pad_width // 2, 0, pad_width-(pad_width//2), 0)
        else:
            w_resized = int(img.size[1] * img_size / img.size[0])
            resized = img.resize((img_size,w_resized))
            pad_width = img_size - w_resized
            padding = (0,pad_width // 2, 0,pad_width-(pad_width//2))
        resized_w_pad = ImageOps.expand(resized, padding)
        count += 1

#resized.save('k.png')
#resized_wo_pad = img.resize(size=(512,512))
        resized_w_pad.save(output_dir+name)
        print('count: ',count)
        if count > 1000:
            break
        
print('Test File Padded')
pad_img('../input/test/','./imet-padded/test/',256)
os.system('zip -r padded-test.zip ./imet-padded/test/')
#print('Train File Padded')
#pad_img('../input/train/','./imet-padded/train/',300)
os.system('rm -rf ./imet-padded/')
print('Done')
#os.system('ls ./imet-padded/test/')