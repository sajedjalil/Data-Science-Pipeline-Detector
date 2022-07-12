import os
import gc
import numpy as np
import pandas as pd
import cv2

from hubmap_v2 import rle_decode
from submit_layer1 import submit as submit_layer1
from submit_layer2 import submit as submit_layer2

def rle_encode_batched(img):
    # the image should be transposed
    pixels = img.T.flatten()
    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]
    
    length = 100000
    output = []
    for i in range(runs.shape[0] // length + 1):
        i1 = i * length
        i2 = min((i+1) * length, runs.shape[0])
        output.append(' '.join(runs[i1:i2].astype(str).tolist()))
        
    return ' '.join(output)

################################################################
print(50*'=' + "\nStarts Layer_1 inference\n" + 50*'=')
submit_layer1(
    None,
    server='kaggle',
    iterations = [
       '../input/hubmap-checkpoints/checkpoint_26d1bfb56/checkpoint_26d1bfb56/00011750_0.058490_model.pth',
       '../input/hubmap-checkpoints/checkpoint_5f3cf9f23/checkpoint_5f3cf9f23/00017500_0.047052_model.pth',
    ],
    fold=[''],
    scale=0.25,
    flip_predict=True,
    checkpoint_sha=None,
    backbone='efficientnet-b0',
    proba_threshold=0.15,
    )

gc.collect()
print(30*'-')
print("list of files created:")
print(os.listdir('/kaggle/working/'))

################################################################
print(50*'=' + "\nStarts Layer_2 inference\n" + 50*'=')
scale = 2
full_sizes = submit_layer2(
    None,
    server          = 'kaggle',
    iterations      = [
                     '../input/hubmap-checkpoints/checkpoint_f765ca3ec/checkpoint_f765ca3ec/00013250_0.940921_model.pth',
#                    '../input/hubmap-checkpoints/checkpoint_f765ca3ec/checkpoint_f765ca3ec/00008750_0.940808_model.pth',
#                    '../input/hubmap-checkpoints/checkpoint_f765ca3ec/checkpoint_f765ca3ec/00012250_0.940725_model.pth',
                     '../input/hubmap-checkpoints/checkpoint_c4c4ab692/checkpoint_c4c4ab692/00010000_0.944602_model.pth',               # LB = 0.923
#                    '../input/hubmap-checkpoints/checkpoint_c4c4ab692/checkpoint_c4c4ab692/00006750_0.943827_model.pth',
#                    '../input/hubmap-checkpoints/checkpoint_c4c4ab692/checkpoint_c4c4ab692/00006250_0.943618_model.pth',
#                    '../input/hubmap-checkpoints/checkpoint_eea12ea38/checkpoint_eea12ea38/00007500_0.939149_model.pth'                 
#                    '../input/hubmap-checkpoints/checkpoint_eea12ea38/checkpoint_eea12ea38/00006750_0.939526_model.pth',                # LB = 0.916
#                    '../input/hubmap-checkpoints/checkpoint_2d5650f29/checkpoint_2d5650f29/00013000_0.940886_model.pth'
#                    '../input/hubmap-checkpoints/checkpoint_2d5650f29/checkpoint_2d5650f29/00006500_0.940636_model.pth'
#                    '../input/hubmap-checkpoints/checkpoint_2d5650f29/checkpoint_2d5650f29/00011750_0.940931_model.pth',               # LB = 0.910
                     '../input/hubmap-checkpoints/checkpoint_75b9744fa/checkpoint_75b9744fa/00011750_0.934729_model.pth',               # LB = 0.920
#                    '../input/hubmap-checkpoints/checkpoint_75b9744fa/checkpoint_75b9744fa/00011500_0.934239_model.pth',
#                    '../input/hubmap-checkpoints/checkpoint_75b9744fa/checkpoint_75b9744fa/00009750_0.934313_model.pth',
#                    '../input/hubmap-checkpoints/checkpoint_75b9744fa/checkpoint_75b9744fa/00009250_0.933873_model.pth',
#                    '../input/hubmap-checkpoints/checkpoint_75b9744fa/checkpoint_75b9744fa/00011250_0.933255_model.pth',
    ],
    fold            = [''],
    scale           = 1/scale,
    flip_predict    = True,
    checkpoint_sha  = None,
    layer1          = '/kaggle/working/',
    backbone        = 'efficientnet-b0',
    )

###############################################################
## Scales up the masks
sub_layer2 = pd.read_csv('submission_layer2.csv')

predicted = []
valid_image_id = []
for i in range(0, len(sub_layer2)):
    
    image_id, encoding = sub_layer2.iloc[i]
    valid_image_id.append(image_id)
    
    width, height, scaled_width, scaled_height = full_sizes[image_id]
    mask = rle_decode(encoding, scaled_height, scaled_width, 1)
    mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)    
    mask = mask.astype(np.uint8)
    
    print('predict mask shape:', mask.shape)

    p = rle_encode_batched(mask)
    predicted.append(p)

df = pd.DataFrame()
df['id'] = valid_image_id
df['predicted'] = predicted

df.to_csv('submission.csv', index=False)

    
print(df)
    
    

