import os
import time

import torch

on_kaggle = True
if on_kaggle:
    import notebook663855f654 as util
    import notebookbad39f7028 as glomerus_detector
else:
    import util
    import glomerus_detector

import pdb


def main():
    if True and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    net = glomerus_detector.GlomerusDetector9()
    model = "hub_detector9_5120_model_dice0.9162.pt"
    model_path = f"{'../input/hubdetector' if on_kaggle else './models/'}"
    net.load_state_dict(
        torch.load(os.path.join(model_path, model), map_location=device))
    net = net.eval()
    net = net.to(device)
    threshold = 0.60

    submission_path = f"{'/kaggle/working' if on_kaggle else './submission'}"
    if on_kaggle:
        submission_file = "submission.csv"
    else:
        submission_file = f"submission_{model}_{threshold}.csv"

    with open(os.path.join(submission_path, submission_file),
              "w") as submission:
        submission.write("id,predicted\n")
        if on_kaggle:
            path = "../input/hubmap-kidney-segmentation/test"
        else:
            path = "./data/train"
        for file in sorted(
            [f for f in os.listdir(path) if f.split('.')[1] == "tiff"]):
            submission.write(f"{file.split('.')[0]},")
            img = util.read_image(os.path.join(path, file))
            patch_size = 1024 * 5
            mask = util.predict_and_threhold_mask(img, net, threshold, device,
                                                  [patch_size, patch_size], 1)
            del img
            encoding = util.encode_mask_rle(mask, offset=1, step=4)
            if len(encoding) > 0:
                encoding = " ".join([f"{val:d}" for val in encoding])
                submission.write(f"{encoding}\n")
            submission.write("\n")
            if not on_kaggle or True:
                print(f"{file} : {time.asctime()}")


if __name__ == "__main__":
    main()
