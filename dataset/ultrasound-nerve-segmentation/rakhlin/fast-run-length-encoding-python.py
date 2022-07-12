'''
Fast inplementation of Run-Length Encoding algorithm
Takes only 200 seconds to process 5635 mask files
'''

import numpy as np
from PIL import Image
import os


def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


if __name__ == '__main__':
    input_path = '../input/train'
    masks = [f for f in os.listdir(input_path) if f.endswith('_mask.tif')]
    masks = sorted(masks, key=lambda s: int(s.split('_')[0])*1000 + int(s.split('_')[1]))
    
    encodings = []
    N = 100     # process first N masks
    for i,m in enumerate(masks[:N]):
        if i % 10 == 0: print('{}/{}'.format(i, len(masks)))
        img = Image.open(os.path.join(input_path, m))
        x = np.array(img.getdata(), dtype=np.uint8).reshape(img.size[::-1])
        x = x // 255
        encodings.append(rle_encoding(x))
        
    #check output
    conv = lambda l: ' '.join(map(str, l)) # list -> string
    subject, img = 1, 1
    print('\n{},{},{}'.format(subject, img, conv(encodings[0])))
    
    # train_masks.csv:
    print('1,1,168153 9 168570 15 168984 22 169401 26 169818 30 170236 34 170654 36 171072 39 171489 42 171907 44 172325 46 172742 50 173159 53 173578 54 173997 55 174416 56 174834 58 175252 60 175670 62 176088 64 176507 65 176926 66 177345 66 177764 67 178183 67 178601 69 179020 70 179438 71 179857 71 180276 71 180694 73 181113 73 181532 73 181945 2 181950 75 182365 79 182785 79 183205 78 183625 78 184045 77 184465 76 184885 75 185305 75 185725 74 186145 73 186565 72 186985 71 187405 71 187825 70 188245 69 188665 68 189085 68 189506 66 189926 65 190346 63 190766 63 191186 62 191606 62 192026 61 192446 60 192866 59 193286 59 193706 58 194126 57 194546 56 194966 55 195387 53 195807 53 196227 51 196647 50 197067 50 197487 48 197907 47 198328 45 198749 42 199169 40 199589 39 200010 35 200431 33 200853 29 201274 27 201697 20 202120 15 202544 6')