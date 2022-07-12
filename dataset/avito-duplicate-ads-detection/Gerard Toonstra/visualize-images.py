try:
    import cv2
except ImportError:
    print('Couldn\'t find opencv so trying to use the fallback')
    from _cv2_fallback import cv2

import pandas as pd
import numpy as np

IMG_WIDTH  = 208
IMG_HEIGHT = 156 
NUM_IMAGES = 9

def show_new_set(set1, set2, isdup, gen_method):

    combo_w = 0
    combo_image = np.zeros((2 * IMG_HEIGHT, NUM_IMAGES * IMG_WIDTH, 3), np.uint8)

    #if not isinstance(set1, unicode) and not isinstance(set1, str):
    if not isinstance(set1, str):
        print( "str1 has no images" )
        return

    #if not isinstance(set2, unicode) and not isinstance(set2, str):
    if not isinstance(set2, str):
        print( "str2 has no images" )
        return

    if len(set1) == 0 or len(set2) == 0:
        print( "one has no images" )
        return

    for s1 in set1.split(","):
        s1 = s1.strip()
        idx = s1[-2:]
        path = '../input/images/Images_%s/%s/%s.jpg' % ( idx[0], idx, s1 )
        if idx[0] == '0':
            path = '../input/images/Images_%s/%s/%s.jpg' % ( idx[0], idx[1], s1 )

        print("Attempt load %s"%(path))
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        h = img.shape[0]
        w = img.shape[1]
        print(w,h)
        if h<1 or w<1 or img is None:
            print("Corrupt file %s"%(path))
            continue

        combo_image[0:h, combo_w:combo_w+w] = img  # copy the obj into the combo image
        combo_w += IMG_WIDTH

    combo_w = 0
    for s2 in set2.split(","):
        s2 = s2.strip()
        idx = s2[-2:]
        path = '../input/images/Images_%s/%s/%s.jpg' % ( idx[0], idx, s2 )
        if idx[0] == '0':
            path = '../input/images/Images_%s/%s/%s.jpg' % ( idx[0], idx[1], s2 )
        print("Attempt load %s"%(path))
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        h = img.shape[0]
        w = img.shape[1]
        print(w,h)
        if h<1 or w<1 or img is None:
            print("Corrupt file %s"%(path))
            continue
        combo_image[IMG_HEIGHT:IMG_HEIGHT+h, combo_w:combo_w+w] = img  # copy the obj into the combo image
        combo_w += IMG_WIDTH

    dupped = 'NotDup'
    if isdup:
        dupped = 'Dup'

    cv2.imshow('%s:%d, %s vs %s' % (dupped, gen_method, set1, set2), combo_image)

    print("Showing images")
    cv2.waitKey()
    cv2.destroyAllWindows()

def show_datasets( infofilename, pairfilename ):
    info = pd.read_csv(infofilename, encoding="utf-8")
    df = pd.read_csv(pairfilename)
    info = info.drop(['title','description','attrsJSON'], axis = 1)
    df = pd.merge(pd.merge(df, info, how = 'inner', left_on = 'itemID_1', right_on = 'itemID'), info, how = 'inner', left_on = 'itemID_2', right_on = 'itemID')

    df[['images_array_x', 'images_array_y', 'isDuplicate', 'generationMethod']].apply(lambda x:show_new_set(x[0],x[1], x[2], x[3]), axis=1)

show_datasets("../input/ItemInfo_train.csv", "../input/ItemPairs_train.csv")
