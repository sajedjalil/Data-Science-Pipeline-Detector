import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from functools import partial
from tqdm import tqdm
import itertools
import multiprocessing as mp
from multiprocessing import set_start_method
from collections import Counter
import subprocess


# Need to install the following two libraries:
subprocess.run("pip install facenet_pytorch", shell=True, check=True)
subprocess.run("pip install mmcv", shell=True, check=True)

from facenet_pytorch import MTCNN
import mmcv

# Some constants
BASE_PATH = "../input/deepfake-detection-challenge/"
OUTPUT_PATH =  "/kaggle/working/train_bboxes.parquet"
BASE_Path = Path(BASE_PATH) / "face_extracted_from_train"
# TODO: Change this to another video. 
TRAIN_FOLDER = Path(BASE_PATH) / "train_sample_videos"
TEST_VIDEO_PATH = TRAIN_FOLDER / "abofeumbvv.mp4"
FRAMES_TO_SKIP = 15
METADATA_PATH = TRAIN_FOLDER / "metadata.json"




def get_frames_from_video(video_path, frames_to_skip=FRAMES_TO_SKIP):
    video = mmcv.VideoReader(video_path.as_posix())
    # Is the video color space really BGR?
    frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video[::frames_to_skip]]
    return frames


def process_faces_detection_one_video(video_path, model, real_filenames, debug):
    # Process only real videos.
    if video_path.name not in real_filenames:
        return "skipped"
    data = []
    frames = get_frames_from_video(video_path)
    for frame_index, frame in enumerate(frames):    
        # Detect faces
        boxes, _ = model.detect(frame)
        if boxes is not None:
            # Take first face. TODO: What about more than one face?
            boxe = boxes[0]
            x_1, y_1, x_2, y_2 = [int(b) for b in boxe]
            filename = video_path.name
            video_folder = video_path.stem
            parent_folder = video_path.parent.stem
            data.append({"x_1": x_1, "x_2": x_2, "y_1": y_1, "y_2": y_2, "filename": filename, "folder": parent_folder, 
                         "frame": frame_index})
            cropped_frame = frame.crop((x_1, y_1, x_2, y_2))
            folder = Path("/kaggle/working/") / parent_folder / video_folder
            path = folder / f'{frame_index}.png'
            folder.mkdir(parents=True, exist_ok=True)
            cropped_frame.save(path.as_posix())
            if debug:
                draw = ImageDraw.Draw(frame)
                draw.rectangle(((x_1, y_1), (x_2, y_2)), outline="red", width=5)
                debug_path =  folder / f'{frame_index}_debug.png'
                frame.save(debug_path.as_posix())
    # Free cache once all the frames have been processed.
    torch.cuda.empty_cache()
    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df)
    pq.write_to_dataset(table , root_path = OUTPUT_PATH)
    return "processed"



def test_pipeline(model):
    process_faces_detection_one_video(TEST_VIDEO_PATH, real_filenames=["abofeumbvv.mp4"], model=model, debug=True)
    df = pd.read_parquet(OUTPUT_PATH)
    assert "abofeumbvv.mp4" in df["filename"].unique()
    assert "train_sample_videos" in df["folder"].unique()

# TODO: Try Dask to // the processing?
def main(debug=True):
    # Init the model
    # TODO: Maybe need to run on CPU for || processing?
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # TODO: Try other models later?
    FD_MODEL = MTCNN(keep_all=True, device=device)
    if debug:
        test_pipeline(FD_MODEL)
    metadata_df = pd.read_json(METADATA_PATH).T
    metadata_df["filename"] = metadata_df.index
    REAL_FILENAMES = metadata_df.loc[lambda df: df["label"] == "REAL", "filename"].tolist()
    # TO avoid issue with GPU.
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    video_paths = TRAIN_FOLDER.glob("*.mp4")
    if debug:
        video_paths = itertools.islice(video_paths, 1)
    num_cpus = min(mp.cpu_count(), len(REAL_FILENAMES))
    print(f"Extracting faces using {num_cpus} CPU workers.")
    # NOTE: this is required for the ``fork`` method to work
    FD_MODEL.share_memory()
    max_iterations = len(REAL_FILENAMES) if not debug else 1
    print("Start!")
    results = []
    with mp.Pool(num_cpus) as pool:
        for result in tqdm(pool.imap_unordered(partial(process_faces_detection_one_video, model=FD_MODEL, real_filenames=REAL_FILENAMES, 
                                    debug=debug), video_paths), total=max_iterations):
            results.append(result)
    print(Counter(results))
    print("Done!")


if __name__ == "__main__":
    main(debug=False)
