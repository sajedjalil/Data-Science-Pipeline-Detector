import random
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import multiprocessing as mp
from tqdm import tqdm, trange

root = "../input/global-wheat-detection/"
train_csv = root + "train.csv"
train_img = root + "train/"
augFolder = root + "imgAug/"



def augworker(k):
    ia.seed(k+random.randint(0,100))
    train = pd.read_csv(train_csv)
    aug = pd.DataFrame(columns=train.columns)
    image_ids = train.image_id.unique()
    beta = image_ids[:10]
    for id in tqdm(image_ids):
        image = imageio.imread(train_img + id + '.jpg')
        bbox = []
        for box in list(train[train.image_id == id]['bbox']):
            (xmin, ymin, width, height) = list(map(int, map(float, box[:-1][1:].split(","))))
            bbox.append(BoundingBox(xmin, ymin, xmin + width, ymin + height))
        bbs = BoundingBoxesOnImage(bbox, shape=image.shape)

        seq = iaa.Sequential([
            iaa.Affine(rotate=(-10, 10)),
            iaa.Affine(shear=(-10, 10)),
            iaa.Affine(scale=(0.5, 1.5)),
            iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, mode=ia.ALL, cval=(0, 255)),

            iaa.WithColorspace(
                to_colorspace="HSV",
                from_colorspace="RGB",
                children=iaa.WithChannels(
                    0,
                    iaa.Add((0, 50))
                )
            )
        ], random_order=True)

        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

        bbs_after = []
        for cors in bbs_aug.remove_out_of_image():
            if cors.is_fully_within_image((1000, 1000, 3)):
                if image_aug[int(cors.x1), int(cors.y1)].all() and image_aug[int(cors.x1), int(cors.y2)].all() and \
                        image_aug[int(cors.x2), int(cors.y1)].all() and image_aug[int(cors.x2), int(cors.y2)].all():
                    bbs_after.append([int(cors.x1), int(cors.y1), int(cors.x2 - cors.x1), int(cors.y2 - cors.y1)])
        name = f"{id}_{k}"
        imageio.imwrite(augFolder + name + '.jpg', image_aug)
        for bbox in bbs_after:
            aug = aug.append([{'image_id': name, 'width': 1024, 'height': 1024, 'bbox': str(bbox),
                               'source': train[train.image_id == id].source.unique()[0]}],
                             ignore_index=True)
    return aug
if __name__ == '__main__':

    train = pd.read_csv(train_csv)
    aug = pd.DataFrame(columns=train.columns)


    num_cores = int(mp.cpu_count())
    #num_cores = 16
    
    pool = mp.Pool(num_cores)

    results = [pool.apply_async(augworker, args=(k,)) for k in range(num_cores)]

    aug = pd.concat([result.get() for result in results])

    aug.to_csv(root + 'aug.csv', index=False)

    trainaug = pd.concat([train,aug])
    trainaug.to_csv(root + 'trainaug.csv', index=False)