"""
Max time for:
    - GPU: 30h
    - TPU: 20h
"""
import time
MAX_TRAIN_TIME = 60*60*5 #60*60*28 # Default 24hours 
GPU_START = time.perf_counter()
import gc
import installer_app_ras
#from dataset_wrapper_app_ras import DatasetWrapper
from model_app_ras import build_resnet50, build_efficientnetb4, dump_model, load_model
from rasterizer_app_ras import build_custom_rasterizer
import evaluation_app_ras as eval_util_functions
import logger_app_ras as logging
import timer_app_ras as timer

from sys import stdin
from signal import SIGINT, signal, getsignal
from tempfile import gettempdir
from datetime import datetime
from torch import nn, load, save
import numpy as np
import torch
from torch.utils.data import DataLoader
from adabound import AdaBound
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset
from l5kit.evaluation.metrics import *
from tqdm import tqdm
import datetime
import traceback
from torch import nn, load, save
from torchvision.models.regnet import regnet_y_8gf, regnet_x_400mf
# TPU imports
#from torch_xla.core import xla_model

# Create output directory for evaluation results
import os
os.system("mkdir /kaggle/working/results")
start_time=None

def save_w_optim(model, optimizer, PATH):
    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, PATH)

def load_w_optim(PATH):
    checkpoint = torch.load(PATH)
    model = build_regnet_small((13,100),None)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser = AdaBound(model.parameters(), lr=1e-3, final_lr=0.01)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optim

def build_regnet_small(in_out_dims, file=None): 
    '''
    Builds a regnet model.

    Params:
        in_out_dims     a tuple(in_channels, out_features)
        file            a tuple(path to model, in_channels, out_features),
                        where 'in_channels' and 'out_features' are referring to
                        the model that is being loaded
    '''
    # NOTE Define model
    model = regnet_x_400mf(pretrained=True)
    print(model)

    # NOTE: Helper function to adjust dimension to fit input and output
    def adjust_layers(in_channels, out_features):
        model.stem[0] = nn.Conv2d(
            in_channels,
            model.stem[0].out_channels,
            kernel_size=model.stem[0].kernel_size,
            stride=model.stem[0].stride,
            padding=model.stem[0].padding,
            bias=False,
        )
        model.fc = nn.Linear(
            in_features=400,
            out_features=out_features,
        )

    # NOTE: Restore model from file
    try:

        # NOTE: Do not forget to adjust layers before restoring parameters
        adjust_layers(*file[1:])
        load_model(model, file[0])
    except: 
        print("file issue\n unable to load model! ")

    # NOTE: Final layer adjustment
    if file is None or in_out_dims != file[1:]:
        adjust_layers(*in_out_dims)

    return model

def build_regnet(in_out_dims, file=None): 
    '''
    Builds a regnet model.

    Params:
        in_out_dims     a tuple(in_channels, out_features)
        file            a tuple(path to model, in_channels, out_features),
                        where 'in_channels' and 'out_features' are referring to
                        the model that is being loaded
    '''
    # NOTE Define model
    model = regnet_y_8gf(pretrained=True)
    #print(model)

    # NOTE: Helper function to adjust dimension to fit input and output
    def adjust_layers(in_channels, out_features):
        model.stem[0] = nn.Conv2d(
            in_channels,
            model.stem[0].out_channels,
            kernel_size=model.stem[0].kernel_size,
            stride=model.stem[0].stride,
            padding=model.stem[0].padding,
            bias=False,
        )
        model.fc = nn.Linear(
            in_features=2016,
            out_features=out_features,
        )

    # NOTE: Restore model from file
    try:

        # NOTE: Do not forget to adjust layers before restoring parameters
        adjust_layers(*file[1:])
        load_model(model, file[0])
    except:
        adjust_layers(*file[1:])
        print("file issue\n unable to load model! ")

    # NOTE: Final layer adjustment
    if file is None or in_out_dims != file[1:]:
        adjust_layers(*in_out_dims)

    return model

def build_model(cfg, build_model, file=None):
    '''
    Builds regression model using CNN.

    Model inputs:   selected history frames(box) + present frame(semantic)
    Model outputs:  target coords(x, y) for each predicted future frame

    Params:
        cfg     config dict
        file    a tuple(path to model, in_channels, out_features),
                where 'in_channels' and 'out_features' are referring to
                the model that is being loaded
    Returns:
        the configured model
    '''
    # NOTE: Calculate number of input channels
    # NOTE: We multiply by 2 since each frame consists of an agent and an ego
    #       image
    history_box_frames = 2 * len(cfg['model_params']['history_box_frames'])
    # NOTE: We add 3 since the semantic rasterizer images always have 3 channels
    num_in_channels = 3 + history_box_frames

    # NOTE: Calculate output dimensions
    # NOTE: We multiply by 2 since we're predicting x and y coords for eac
    num_targets = 2 * cfg['model_params']['future_num_frames']

    return build_model((num_in_channels, num_targets), file)


def forward(device, model, criterion, data):
    '''
    Do forward pass on data using the given model

    Params:
        device      the device to run the model on
        model       the model
        criterion   the metric/criterion used to calculate the loss
        data        the input data
    '''
    inputs = data['image'].to(device)
    target_avails = data['target_availabilities'].unsqueeze(-1).to(device)
    targets = data['target_positions'].to(device)

    # NOTE: Do forward pass
    outputs = model(inputs).reshape(targets.shape)
    loss = criterion(outputs, targets)

    # NOTE: Filter out output from loss via target availabilities
    loss *= target_avails

    return loss.mean(), outputs

def train_model(cfg, dm, device, model, optimizer, criterion, logger):
    train_cfg = cfg['train_data_loader']
    num_workers = train_cfg['num_workers']

    # NOTE: Setup signal handler to manual interruption
    original_handler = getsignal(SIGINT)
    state = { 'interrupted': False}
    def handler(_signum, _frame):
        state['interrupted'] = True
        signal(SIGINT, original_handler)
    signal(SIGINT, handler)

    num_epochs = 0
    dirs = [*map(lambda d: f"{train_cfg['key']}/{d}", next(os.walk(f"/kaggle/input/{train_cfg['key']}"))[1])]
    logger.info(f"all zarrs: {dirs}")
    to_skip = train_cfg["split_offset"]
    while not state['interrupted']:
        logger.info("AAAAAnd another round")
        for train_path in dirs:
            if state['interrupted']:
                #dump_model(model, './model_reg.pt')
                save_w_optim(model,optimizer,"./checkpoint_reg_s.pt")
                break
            if to_skip > 0:
                to_skip -= 1
                continue
            logger.info(f"Start training with {train_path}")
            # NOTE: Load training dataset
            gc.collect()
            dataset = ChunkedDataset(dm.require(train_path)).open()
            dataset = AgentDataset(
                cfg,
                dataset,
                build_custom_rasterizer(cfg, dm),
                min_frame_history=0,
                min_frame_future=10,
            )
            num_max_batches = np.ceil(
                len(dataset) / train_cfg['batch_size'],
            ).astype(int)
            dataloader_opts = dict(
                shuffle=train_cfg['shuffle'],
                batch_size=train_cfg['batch_size'],
                num_workers=num_workers,
            )
            if num_workers > 0:
                dataloader_opts['prefetch_factor'] = 1
            train_dataloader = DataLoader(
                dataset,
                **dataloader_opts
            )

            # NOTE: Switch to training mode
            model.train()

            # NOTE: Training loop
            losses = []
            num_batches = 0
            num_batches_per_checkpoint = cfg['train_params']['checkpoint_every_n_steps']
            start_time = None
            progress = tqdm(iter(train_dataloader), leave=False)
            for data in progress:
                if time.perf_counter() - GPU_START >= MAX_TRAIN_TIME: # Stop training after specified time
                    logger.info("Max training time reached")
                    state['interrupted'] = True
                    #dump_model(model, './model_reg.pt')
                    save_w_optim(model,optimizer,"./checkpoint_reg_s.pt")
                    break
                    
                num_batches += 1

                # NOTE: Normalize image
                data['image'] = data['image'].float() / 255

                # NOTE: Measure start time
                if start_time is None:
                    start_time = datetime.datetime.now()

                # NOTE: Do forward pass
                loss, outputs = forward(device, model, criterion, data)

                # NOTE: Do backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # NOTE: Keep track of losses
                losses.append(loss.item())
                losses = losses[-10*num_batches_per_checkpoint:]
                progress.set_description(
                    ', '.join([
                        f'epoch: {num_epochs+1}({num_batches}/{num_max_batches})',
                        f'loss: {loss.item()}',
                        f'loss(r-avg): {np.mean(losses)}',
                    ]),
                )

                # EVALUATION
                gt_train_path = "/kaggle/working/results/gt_train.csv"
                pred_path = "/kaggle/working/results/pred_train.csv"
                eval_util_functions.write_csvs(data, gt_train_path, pred_path, device, outputs)

                # calculate metrics
                metrics = eval_util_functions.compute_metrics_csv(gt_train_path, pred_path, [
                                                                    neg_multi_log_likelihood,
                                                                    rmse,
                                                                    average_displacement_error_oracle,
                                                                    average_displacement_error_mean,
                                                                    final_displacement_error_oracle,
                                                                    final_displacement_error_mean])        
                # create csv file
                eval_util_functions.save_metrics('results/metrics.csv', loss.item(), np.mean(losses), metrics) 
                
            save_w_optim(model,optimizer,"./checkpoint_reg_s.pt")
            # dump_model(model, './model_reg.pt')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  
        num_epochs += 1
    print(f'Received interrupt. Stopping training at epoch {num_epochs+1}')
    

    # NOTE: Release memory after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

"""
def train_model(cfg, dm, device, model, optimizer, criterion, logger):
    # NOTE: Load training dataset
    train_cfg = cfg['train_data_loader']
    num_workers = train_cfg['num_workers']
    dataset = ChunkedDataset(dm.require(train_cfg['key'])).open()
    dataset = AgentDataset(
        cfg,
        dataset,
        build_custom_rasterizer(cfg, dm),
        min_frame_history=0,
        min_frame_future=10,
    )
    num_max_batches = np.ceil(
        len(dataset) / train_cfg['batch_size'],
    ).astype(int)
    dataloader_opts = dict(
        shuffle=train_cfg['shuffle'],
        batch_size=train_cfg['batch_size'],
        num_workers=num_workers,
    )

    dataloader_opts['prefetch_factor'] = 1
    train_dataloader = DataLoader(
        dataset,
        **dataloader_opts
    )

    # NOTE: Setup signal handler to manual interruption
    original_handler = getsignal(SIGINT)
    state = { 'interrupted': False }
    def handler(_signum, _frame):
        state['interrupted'] = True
        signal(SIGINT, original_handler)
    signal(SIGINT, handler)

    # NOTE: Switch to training mode
    model.train()

    # NOTE: Training loop
    tr_it = iter(train_dataloader)
    losses = []
    num_epochs, num_batches = 0, 0
    num_batches_per_checkpoint = cfg['train_params']['checkpoint_every_n_steps']
    start_time = None
    while not state['interrupted']:
        progress = tqdm(range(num_batches_per_checkpoint), leave=False)
        for _ in progress:
            if time.perf_counter() - GPU_START >= MAX_TRAIN_TIME: # Stop training after specified time
                logger.info("Max training time reached")
                state['interrupted'] = True
            
            # DEBUG
            if _ > 10: 
                state['interrupted'] = True
            # DEBUG END
            
            if state['interrupted']: break

            # NOTE: Fetch next batch
            try:
                data = next(tr_it)
                num_batches += 1
            except StopIteration:
                tr_it = iter(train_dataloader)
                data = next(tr_it)
                num_epochs += 1
                num_batches = 0

            # NOTE: Normalize image
            data['image'] = data['image'].float() / 255

            # NOTE: Measure start time
            if start_time is None:
                start_time = datetime.datetime.now()

            # NOTE: Do forward pass
            loss, outputs = forward(device, model, criterion, data)
            
            # NOTE: Do backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # NOTE: Keep track of losses
            losses.append(loss.item())
            losses = losses[-10*num_batches_per_checkpoint:] # DENNIS WHY ?????
            progress.set_description(
                ', '.join([
                    f'epoch: {num_epochs+1}({num_batches}/{num_max_batches})',
                    f'loss: {loss.item()}',
                    f'loss(r-avg): {np.mean(losses)}',
                ]),
            )
            
            # EVALUATION
            gt_train_path = "/kaggle/working/results/gt_train.csv"
            pred_path = "/kaggle/working/results/pred_train.csv"
            eval_util_functions.write_csvs(data, gt_train_path, pred_path, device, outputs)

            # calculate metrics
            metrics = eval_util_functions.compute_metrics_csv(gt_train_path, pred_path, [
                                                                neg_multi_log_likelihood,
                                                                rmse,
                                                                average_displacement_error_oracle,
                                                                average_displacement_error_mean,
                                                                final_displacement_error_oracle,
                                                                final_displacement_error_mean])        
            # create csv file
            eval_util_functions.save_metrics('results/metrics.csv', loss.item(), np.mean(losses), metrics) 
            # END

        # NOTE: Persist model to disk
        dump_model(model, './model.pt')
    logger.info(f'Received interrupt. Stopping training at epoch {num_epochs+1}')
    logger.info(f'Training took {datetime.datetime.now() - start_time}')

    # NOTE: Release memory after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache() """

def main(cfg, timer, logger):
    start = time.time()
    
    # NOTE: Setup data manager and load config
    #cfg = load_config_data('../configs/agent_motion_config.yaml')
    #os.environ["L5KIT_DATA_FOLDER"] = "/kaggle/input"
    dm = LocalDataManager('/kaggle/input/')

    # NOTE: Build rasterizer
    logger.info(f"{20*'='} Build rasterizer {20*'='}")
    timer.start("build_rasterizer")
    rasterizer = build_custom_rasterizer(cfg, dm)
    timer.end("build_rasterizer")

    # NOTE: Define which device to use and build model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = xla_model.xla_device()
    logger.info(f"Train with device: {device}")
    timer.start("build_model")
    model = build_model(
        cfg,
        build_regnet_small,
        # TODO: Handle dimensions better
        ('../input/model-checkpoints/model_reg2.pt', 13, 100),    # TODO: Change for new model
    ).to(device)
    timer.end("build_model")
    optimizer = AdaBound(model.parameters(), lr=1e-3, final_lr=0.01)
    # NOTE: Run training
    logger.info(f"{20*'='} Start training {20*'='}")
    timer.start("train")
    try:
        model, optimizer = load_w_optim("./checkpoint_reg_s.pt")
    except:
        print("couldn't load model")
    train_model(
        cfg,
        dm,
        device,
        model,
        optimizer,
        torch.nn.MSELoss(reduction='none'),
        logger
    )
    timer.end("train")
    total = time.time() - start
    logger.info(f"{50*'='}")
    logger.info(f"Finished program execution after {total}s | {total/60}min | {total / (60*60)}h")
    timer.log_times()

def get_agent_motion_config():
    amc = {
        'format_version': 4,
        'model_params': {
            'model_architecture': 'regnet_x_400mf',
            'history_num_frames': 5,
            'history_box_frames': [0, 1, 2, 4, 8],
            'future_num_frames': 50,
            'step_time': 0.1,
            'render_ego_history': True
        },
        'raster_params': {
            'raster_size': [384, 192],
            'pixel_size': [0.25, 0.25],
            'ego_center': [0.25, 0.5],
            'map_type': 'box_semantic_fast',
            'satellite_map_key': 'lyft-motion-prediction-autonomous-vehicles/aerial_map/aerial_map.png',
            'semantic_map_key': 'lyft-motion-prediction-autonomous-vehicles/semantic_map/semantic_map.pb',
            'dataset_meta_key': 'lyft-motion-prediction-autonomous-vehicles/meta.json',
            'filter_agents_threshold': 0.5,
            'disable_traffic_light_faces': False,
            'set_origin_to_bottom': True,
        },
        'train_data_loader': {
            'key': "lyft-full-chopped", #"lyft-full-training-set/train_full.zarr",# TODO: set to directory with all chunks
            'batch_size': 32,
            'shuffle': True,
            'num_workers': 0,
            'split_offset': 5               # TODO: CHANGE IF TRAINING SHOULD BE RESUMED FROM ANOTHER CHUNK
        },
        'val_data_loader': {
            'key': 'lyft-motion-prediction-autonomous-vehicles/scenes/validate.zarr',
            'batch_size': 18,
            'shuffle': False,
            'num_workers': 2,
        },
        'train_params': {
            'checkpoint_every_n_steps': 1000,
            'max_num_steps': 1000,
            'eval_every_n_steps': 10000,
        },
    }
    return amc

if __name__ == "__main__":
    # Initialize logger
    now = datetime.datetime.now()
    now = now.strftime('%Y-%m-%d_%H:%M:%S')
    VERSION = f"{now}"
    logging.create_logger(VERSION)
    logger = logging.get_logger(VERSION) 
    logger.info(f"Initialized Logger {VERSION}")
    
    # initialize timer
    timer = timer.Timer(logger)
    logger.info("Initialized timer")
    
    # start training
    logger.info("************** Model Training **************")
    cfg = get_agent_motion_config()
    main(cfg, timer, logger)