import argparse
import os
import sys
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image
import git
import PIL
import numpy as np
import cv2
import gc
import pandas as pd
import rasterio
import torch
from rasterio.windows import Window
from torch.nn.parallel.data_parallel import data_parallel
from datetime import datetime

# import warnings

# warnings.filterwarnings("ignore")

os.environ['PYTHONWARNINGS'] = 'ignore'

SERVER_RUN = 'kaggle'
DEBUG = False

if SERVER_RUN == 'local':
    from code.data_preprocessing.dataset_v2020_11_12 import make_image_id, \
        draw_contour_overlay, image_show_norm
    from code.hubmap_v2 import rle_encode, rle_encode_less_memory, get_data_path, read_mask, to_mask
    from code.lib.include import IDENTIFIER
    from code.lib.utility.file import Logger, time_to_str
    from code.unet_b_resnet34_aug_corrected.model import Net, \
        np_binary_cross_entropy_loss_optimized, np_dice_score_optimized, np_accuracy_optimized

elif SERVER_RUN == 'kaggle':
    IDENTIFIER = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    #     Let's see what we have in our imported directory
    os.system('pip install ../input/offline-packages-hubmap/imutils-0.5.4')

    from hubmap_v2 import rle_encode, rle_encode_less_memory, get_data_path, read_mask, to_mask, make_image_id
    from utility_hubmap import Logger, time_to_str
    from hubmap_model import Net, np_binary_cross_entropy_loss_optimized, np_dice_score_optimized, np_accuracy_optimized

    print(os.listdir())
    print(os.listdir('../input/'))

import imutils

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Image.MAX_IMAGE_PIXELS = None
is_mixed_precision = False


def get_probas(net, tile_image, flip_predict):
    """
    Itère sur les images et calcule les probas de chaque image.
    Les prédictions sont éventuellement moyennées / à des inversions suivants les axes x et y.
    """
    m = torch.from_numpy(tile_image[np.newaxis, ...]).cuda()
    p = []
    with torch.no_grad():
        logit = data_parallel(net, m)
        p.append(torch.sigmoid(logit))
        if flip_predict:  # inference sur les images inversées / axes x et y
            for _dim in [(2,), (3,), (2, 3)]:
                _logit = data_parallel(net, m.flip(dims=_dim))
                p.append(torch.sigmoid(_logit).flip(dims=_dim))

    p = torch.stack(p).mean(0)
    p = p.squeeze()
    return p.data.cpu().numpy()


class TileGenerator:
    """ Reads an image and creates a generator to load sub-images
    """

    def __init__(self, image_id, raw_data_dir, size=320, scale=1, server=None):
        self.size = int(size / scale)
        self.server = server
        self.scale = scale

        print(50 * '-')
        print(f"processing image: {image_id}")

        if server == 'local':
            self.image_loader = HuBMAPDataset(raw_data_dir + '/train/%s.tiff' % image_id,
                                              sz=self.size, scale=scale)
            mask_file = raw_data_dir + '/train/%s.mask.png' % image_id
            self.original_mask = read_mask(mask_file)
            # if self.scale < 1:
            #     self.original_mask = cv2.resize(self.original_mask, dsize=None,
            #                                     fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        else:
            self.image_loader = HuBMAPDataset(raw_data_dir + '/test/%s.tiff' % image_id,
                                              sz=self.size, scale=scale)

        self.height, self.width = self.image_loader.height, self.image_loader.width
        print(f"image size: width={self.width} x height={self.height}")

        # ----------------------------------------------------------------------
        # calcul des centroides des glomeruli / aux prédictions de la layer 1
        # ----------------------------------------------------------------------
        #####################
        # make coord
        #####################
        half = self.size // 2
        step = self.size * 0.8

        print(f"image size: {self.size}")
        print(f"nb width step: {np.floor((self.width - self.size) / step)}")
        print(f"nb height step: {np.floor((self.height - self.size) / step)}")

        xx = np.array_split(np.arange(half, self.width - half), np.floor((self.width - self.size) / step))
        yy = np.array_split(np.arange(half, self.height - half), np.floor((self.height - self.size) / step))
        xx = [int(x[0]) for x in xx] + [self.width - half]
        yy = [int(y[0]) for y in yy] + [self.height - half]
        self.centroid_list = [[cx, cy] for cx in xx for cy in yy]
        # for index, _pos in enumerate(self.centroid_list):
        #     if index % 100 != 0: continue
        #     print(index, _pos)

        print(f'nb of images: {len(self.centroid_list)}')

        if DEBUG:
            self.centroid_list = self.centroid_list[::8]

    def get_next(self):
        for cX, cY in self.centroid_list:

            sub_image, _ = self.image_loader[(cX - self.size // 2, cY - self.size // 2)]

            hsv = cv2.cvtColor(sub_image.astype(np.float32), cv2.COLOR_RGB2HSV)
            h, s, v = [c.mean() for c in cv2.split(hsv)]

            if self.server == 'local':
                sub_mask = self.original_mask[
                           cY - self.size // 2: cY + self.size // 2,
                           cX - self.size // 2: cX + self.size // 2]

                sub_mask = cv2.resize(sub_mask, dsize=None,
                                      fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

                tile_mask = np.copy(sub_mask)
            else:
                tile_mask = None

            coord = [cX, cY, 1]
            tile_image = np.copy(sub_image)

            tile_image = np.ascontiguousarray(tile_image.transpose(2, 0, 1))
            tile_image = tile_image.astype(np.float32) / 255  # Colors x tile_x x tile_y

            if self.server == 'local':
                tile_mask = np.ascontiguousarray(tile_mask)
                tile_mask = tile_mask.astype(np.float32) / 255  # tile_x x tile_y

            yield {
                'tile_image': tile_image,
                'tile_mask': tile_mask,
                'centroids': coord,
                'hsv': [h, s, v]
            }


class HuBMAPDataset:
    def __init__(self, image_path, sz=256, scale=1, saturation_threshold=40):

        self.scale = scale
        self.s_th = saturation_threshold  # saturation blancking threshold
        self.p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels
        scale_transform = rasterio.Affine(1, 0, 0, 0, 1, 0)

        self.data = rasterio.open(
            os.path.join(image_path),
            transform=scale_transform,
            num_threads='all_cpus'
        )
        # some images have issues with their format
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape

        self.width = self.shape[1]
        self.height = self.shape[0]
        self.sz = sz

        print(f"image loader for {image_path} created")
        print(f"original image size = {self.width} x {self.height}")

    # def __len__(self):
    #     return self.n0max * self.n1max

    def __getitem__(self, pos):
        x0, y0 = pos
        # make sure that the region to read is within the image

        p10, p11 = max(0, x0), min(x0 + self.sz, self.width)
        p00, p01 = max(0, y0), min(y0 + self.sz, self.height)
        img = np.zeros((self.sz, self.sz, 3), np.uint8)

        # print(self.data.count, pos, (p00 + p01) / 2, (p10 + p11) / 2)

        # mapping the load region to the tile
        if self.data.count == 3:
            img[(p00 - y0):(p01 - y0), (p10 - x0):(p11 - x0)] = np.moveaxis(
                self.data.read([1, 2, 3], window=Window.from_slices((p00, p01), (p10, p11))), 0, -1)
        else:
            for i in range(3):
                img[(p00 - y0):(p01 - y0), (p10 - x0):(p11 - x0), i] = \
                    self.layers[i].read(1, window=Window.from_slices((p00, p01), (p10, p11)))

        if self.scale < 1:
            img = cv2.resize(img, dsize=None,
                             fx=self.scale, fy=self.scale, interpolation=cv2.INTER_LINEAR)

            # img = cv2.resize(img, (self.sz // self.reduce, self.sz // self.reduce),
            #                  interpolation=cv2.INTER_AREA)

        # print('x, y = ', x0, y0)
        # image_show_norm('overlay2',
        #                 img / 255,
        #                 min=0, max=1, resize=0.5)
        # cv2.waitKey(0)

        # sys.exit()

        # check for empty images
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        norm = 1
        if (s > self.s_th).sum() <= self.p_th or img.sum() <= self.p_th:
            return img / norm, -1
        else:
            return img / norm, +1


def result_bookeeping(id, tile_probability, overall_probabilities,
                      tile_mask, image, tile_centroids, server, submit_dir, save_to_disk, resize_scale):
    probas = np.mean(overall_probabilities, axis=0)

    Path(submit_dir + f'/{id}').mkdir(parents=True, exist_ok=True)

    x0, y0 = tile_centroids[:2]
    if server == 'local':
        truth = tile_mask[:, :]
        overlay = draw_contour_overlay(
            np.ascontiguousarray(image.transpose(1, 2, 0)).astype(np.float32),
            tile_mask[:, :].astype(np.float32),
            color=(1, 0, 0),
            thickness=6
        )
    else:
        overlay = np.ascontiguousarray(image.transpose(1, 2, 0)).astype(np.float32)

    # print('overlay:', overlay.shape)
    # print('tile_probabilities', tile_probability.shape)

    if len(overall_probabilities) == 1:

        proba = tile_probability[:, :]

        overlay2 = draw_contour_overlay(
            overlay,
            tile_probability[:, :].astype(np.float32),
            color=(0, 1, 0),
            thickness=6
        )
        if save_to_disk:
            cv2.imwrite(submit_dir + f'/{id}/y{y0}_x{x0}.png', (overlay2 * 255).astype(np.uint8))

    else:

        proba = probas[:, :]

        overlay2 = draw_contour_overlay(
            overlay,
            probas[:, :].astype(np.float32),
            color=(0, 1, 0),
            thickness=6
        )
        if save_to_disk:
            cv2.imwrite(submit_dir + f'/{id}/y{y0}_x{x0}.png', (overlay2 * 255).astype(np.uint8))

        overlay2 = draw_contour_overlay(
            overlay2,
            tile_probability[:, :].astype(np.float32),
            color=(0, 0, 1),
            thickness=3
        )

    if server == 'local':
        dice = np_dice_score_optimized(proba, truth)
    else:
        dice = None

    #     image_show_norm('overlay2',
    #                     overlay2,
    #                     min=0, max=1, resize=resize_scale)
    #     cv2.waitKey(1)

    # df = pd.DataFrame(tile_scores, columns=['tile', 'image', 'x', 'y', 'dice'])
    # df.to_csv(submit_dir + f'/{id}_scores.csv')

    return f'y{y0}_x{x0}.png', x0, y0, dice


def submit(sha, server, iterations, fold, scale,
           flip_predict, checkpoint_sha, backbone, proba_threshold):
    project_repo, raw_data_dir, data_dir = get_data_path(SERVER_RUN)

    if SERVER_RUN == 'kaggle':
        out_dir = f'../input/hubmap-checkpoints/checkpoint_{checkpoint_sha}/'
        result_dir = '/kaggle/working/'
    else:
        out_dir = project_repo + f"/result/Layer_1/fold{'_'.join(map(str, fold))}"
        result_dir = out_dir

    # --------------------------------------------------------------
    # Verifie le sha1 du modèle à utiliser pour faire l'inférence
    # Le commit courant est utilisé si non spécifié
    # --------------------------------------------------------------
    if checkpoint_sha is not None or SERVER_RUN == 'kaggle':
        _sha = checkpoint_sha
    else:
        _sha = sha
    
    if _sha is not None:
        _checkpoint_dir = out_dir + f"/checkpoint_{_sha}/"
        print("Checkpoint for current inference:", _sha)
        print(os.listdir(_checkpoint_dir))

    # --------------------------------------------------------------
    # Verifie les checkpoints à utiliser pour l'inférence:
    # - 'all'
    # - 'topN' avec N entier
    # - INTEGER (= nb iterations)
    # --------------------------------------------------------------

    if isinstance(iterations, list):
        iter_tag = 'custom'
        initial_checkpoint = iterations

    elif iterations == 'all':
        iter_tag = 'all'
        model_checkpoints = [_file for _file in os.listdir(_checkpoint_dir)]
        initial_checkpoint = [out_dir + f'/checkpoint_{_sha}/{model_checkpoint}'
                              for model_checkpoint in model_checkpoints]

    elif 'top' in iterations:
        nbest = int(iterations.strip('top'))

        iter_tag = f'top{nbest}'
        model_checkpoints = [_file for _file in os.listdir(_checkpoint_dir)]
        scores = [float(_file.split('_')[1]) for _file in os.listdir(_checkpoint_dir)]

        ordered_models = list(zip(model_checkpoints, scores))
        ordered_models.sort(key=lambda x: x[1], reverse=True)
        model_checkpoints = np.array(ordered_models[:nbest])[:, 0]
        model_checkpoints = model_checkpoints.tolist()

        initial_checkpoint = [out_dir + f'/checkpoint_{_sha}/{model_checkpoint}'
                              for model_checkpoint in model_checkpoints]

    else:
        iter_tag = f"{int(iterations):08}"
        [model_checkpoint] = [_file for _file in os.listdir(_checkpoint_dir)
                              if iter_tag in _file.split('_')[0]]
        initial_checkpoint = [out_dir + f'/checkpoint_{_sha}/{model_checkpoint}']

    print("checkpoint(s):", initial_checkpoint)
    print(f"submit with server={server}")

    # ------------------------------------------------------
    # Get checkpoint of the model used to make predictions
    # ------------------------------------------------------
    if SERVER_RUN == 'kaggle':
        submit_dir = result_dir
    else:
        if checkpoint_sha is None:
            tag = ''
        else:
            tag = checkpoint_sha + '-'

        if iterations == 'all':
            submit_dir = result_dir + f'/predictions_{sha}/%s-%s-%smax' % (server, 'all', tag)
        elif flip_predict:
            submit_dir = result_dir + f'/predictions_{sha}/%s-%s-%smax' % (server, iter_tag, tag)
        else:
            submit_dir = result_dir + f'/predictions_{sha}/%s-%s-%snoflip' % (server, iter_tag, tag)
    os.makedirs(submit_dir, exist_ok=True)

    log = Logger()
    log.open(result_dir + f'/log.submit_{sha}.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))

    ##########################################################################################
    # Get the IDs of the images --------------------------------------------------------------
    ##########################################################################################
    if SERVER_RUN == 'kaggle':
        df_submit = pd.read_csv('../input/hubmap-kidney-segmentation/sample_submission.csv', index_col='id')
        valid_image_id = df_submit.index.tolist()
    elif server == 'local':
        valid_image_id = make_image_id('train-all')
    elif server == 'kaggle':
        valid_image_id = make_image_id('test-all')

    ##########################################################################################
    # Define prediction parameters -----------------------------------------------------------
    ##########################################################################################
    tile_size = 256 * 3  # taille dans le système scalé
    tile_average_step = 320
    tile_min_score = 0.25

    log.write('tile_size = %d \n' % tile_size)
    log.write('tile_average_step = %d \n' % tile_average_step)
    log.write('tile_scale = %f \n' % scale)
    log.write('tile_min_score = %f \n' % tile_min_score)
    log.write('\n')

    ##################################
    # Starts iterating over images
    ##################################
    predicted = []
    df = pd.DataFrame()
    start_timer = timer()

    for ind, id in enumerate(valid_image_id):

        log.write(50 * "=" + "\n")
        log.write(f"Inference for image: {id} \n")

        # if id != '3589adb90': continue

        ###############
        # Define tiles
        ###############
        tiles = TileGenerator(image_id=id, raw_data_dir=raw_data_dir, size=tile_size,
                              scale=scale, server=server)
        print(30 * '-')
        height = tiles.height
        width = tiles.width
        print(f"tile matrix shape (without scaling): {height} x {width}")

        tile_probability = []
        results = []
        ##############################################
        ### Iterate on sub-images with scaled sizes
        ##############################################

        for index, tile in enumerate(tiles.get_next()):

            if SERVER_RUN != 'kaggle':
                print('\r %s: n°%d %s' %
                      (ind, index, time_to_str(timer() - start_timer, 'sec')),
                      end='', flush=True)
            elif index % 100 == 0:
                print('\r %s: n°%d %s' %
                      (ind, index, time_to_str(timer() - start_timer, 'sec')),
                      end='', flush=True)

            h, s, v = tile['hsv']
            condition = s > 0.05
            if s < 0.05:
                # print(f"image removed, saturation: {s}")
                tile_probability.append(np.zeros((tile_size, tile_size)))
                continue

            #######################################
            # Iterates over models.
            # The predictions are then averaged.
            #######################################
            overall_probabilities = []
            for _num, _checkpoint in enumerate(initial_checkpoint):
                net = Net(backbone).cuda()
                state_dict = torch.load(_checkpoint, map_location=lambda storage, loc: storage)['state_dict']
                net.load_state_dict(state_dict, strict=True)
                net = net.eval()
                image_probability = get_probas(net, tile['tile_image'], flip_predict)
                overall_probabilities.append(image_probability)

                ################################################################
                # Sauvegarde + visualisation de l'image courante
                ################################################################
                last_iter = _num == len(initial_checkpoint) - 1

                if SERVER_RUN == 'local':

                    # print('\n', index, tiles.centroid_list[index], tile['hsv'], '\n')
                    # h, s, v = tile['hsv']
                    # condition = s > 0.05
                    # print(f"kept: {condition}")
                    image_name, x0, y0, dice = result_bookeeping(
                        id,
                        image_probability,
                        overall_probabilities,
                        tile['tile_mask'],
                        tile['tile_image'],
                        tile['centroids'],
                        server,
                        submit_dir,
                        save_to_disk=last_iter,
                        resize_scale=800 / tile_size
                    )
                    if last_iter:
                        results.append([id, image_name, x0, y0, dice])

            # print(np.max(overall_probabilities, axis=0)[:3])
            # print()
            # print(np.mean(overall_probabilities, axis=0)[:3])

            # sys.exit()

            _probas = np.max(overall_probabilities, axis=0)
            tile_probability.append(_probas.astype(np.float32))
            del overall_probabilities, _probas
            del net, state_dict, image_probability
            gc.collect()

        ###############################################################################
        # Concatène les sous images et recrée une image conforme à la taille initiale
        # Lors de la concaténation, les pixels sont pondérés / à la distance au centre
        # de l'image
        ###############################################################################

        scaled_centroid_list = (np.array(tiles.centroid_list) * scale).astype(np.int).tolist()

        probability = to_mask(tile_probability,  # N * scaled_height x scaled_width
                              scaled_centroid_list,
                              int(scale * height),
                              int(scale * width),
                              scale,
                              tile_size,
                              tile_average_step,
                              tile_min_score,
                              aggregate='max'
                              )

        # print(probability.min())
        # print(probability.max())
        # sys.exit()


        predict = (probability > proba_threshold).astype(np.uint8)
        cv2.imwrite(submit_dir + '/%s.probability.png' % id, (probability * 255).astype(np.uint8))
        cv2.imwrite(submit_dir + '/%s.predict.png' % id, (predict * 255))
        del predict, probability
        gc.collect()


########################################################################
# main #################################################################
########################################################################
if __name__ == '__main__':

    DEBUG = False

    if SERVER_RUN == 'local':
        # Initialize parser
        parser = argparse.ArgumentParser()

        # Adding optional argument
        parser.add_argument("-i", "--Iterations", nargs="+", help="number of iterations")
        parser.add_argument("-s", "--Server", help="run mode: server or kaggle")
        parser.add_argument("-f", "--fold", help="fold")
        parser.add_argument("-r", "--flip", help="flip image and merge", default=True)
        parser.add_argument("-c", "--CheckpointSha", help="checkpoint with weights", default=None)

        args = parser.parse_args()

        if not args.fold:
            print("fold missing")
            sys.exit()
        elif isinstance(args.fold, int):
            fold = [int(args.fold)]
        elif isinstance(args.fold, str):
            fold = [int(c) for c in args.fold.split()]
        else:
            print("unsupported format for fold")
            sys.exit()

        if not args.Iterations:
            print("iterations missing")
            sys.exit()

        if args.Server in ['kaggle', 'local']:
            print("Server: % s" % args.Server)
        else:
            print("Server missing")
            sys.exit()

        repo = git.Repo(search_parent_directories=True)
        model_sha = repo.head.object.hexsha[:9]
        print(f"current commit: {model_sha}")

        changedFiles = [item.a_path for item in repo.index.diff(None) if item.a_path.endswith(".py")]
        if len(changedFiles) > 0:
            print("ABORT submission -- There are unstaged files:")
            for _file in changedFiles:
                print(f" * {_file}")

        else:
            submit(model_sha,
                   server=args.Server,
                   iterations=args.Iterations,
                   fold=fold,
                   scale=0.25,
                   flip_predict=args.flip,
                   checkpoint_sha=args.CheckpointSha,
                   backbone='efficientnet-b0',
                   proba_threshold=0.15,
                   )

    elif SERVER_RUN == 'kaggle':
#                 pass
        submit(None,
               server='kaggle',
                iterations = [
                   '../input/hubmap-checkpoints/checkpoint_26d1bfb56/checkpoint_26d1bfb56/00011750_0.058490_model.pth',
               ],
               fold=[''],
               scale=0.25,
               flip_predict=False,
               checkpoint_sha='26d1bfb56',
               backbone='efficientnet-b0',
               proba_threshold=0.15,
               )

