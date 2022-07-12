import numpy as np

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from PIL import Image, ImageDraw


import os

try:
    import moviepy
except:
    print('installing moviepy')
    os.system('pip install moviepy')

from moviepy.editor import ImageSequenceClip


from pathlib import Path
    
data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')


cmap_lookup = [
        '#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]
cmap_lookup = [ np.array( [int(x[1:3],16), int(x[3:5],16), int(x[5:],16)])  for x in cmap_lookup]


def def_cmap(x):
    """ 
        Translate a task matrix to a color coded version
        
        arguments
            x : a h x w task matrix
        returns 
            a h x w x 3 matrix with colors instead of numbers
    """
    y = np.zeros((*x.shape, 3))
    y[x<0, :] = np.array([112,128,144])
    y[x>9,:] = np.array([255,248,220])
    for i, c in enumerate(cmap_lookup):        
        y[x==i,:] = c
    return y
    
def draw_one(x, k=20, cmap=None):
    """
        Create a PIL image from a task matrix, the task will be 
        drawn using the default color coding with grid lines
        
        arguments
            x : a task matrix
            k = 20 : an up scaling factor
        returns
            a PIL image 
            
    """
    if cmap is None:
        cmap = def_cmap
    img = Image.fromarray(cmap(x).astype(np.uint8)).resize((x.shape[1]*k, x.shape[0]*k), Image.NEAREST )
    
    draw = ImageDraw.Draw(img)
    for i in range(x.shape[0]):
        draw.line((0, i*k, img.width, i*k), fill=(80, 80, 80), width=1)   
    for j in range(x.shape[1]):    
        draw.line((j*k, 0, j*k, img.height), fill=(80, 80, 80), width=1)
    return img


def vcat_imgs(imgs, border=10):
    """
        Concatenate images vertically
        
        arguments:
            imgs : an array of PIL images
            border = 10 : the size of space between images
        returns:
            a PIL image
    """
    
    h = max(img.height for img in imgs)
    w = sum(img.width for img in imgs)
    res_img = Image.new('RGB', (w + border*(len(imgs)-1), h), color=(255, 255, 255))

    offset = 0
    for img in imgs:
        res_img.paste(img, (offset,0))
        offset += img.width + border
        
    return res_img




def plot_task(task):
    """
        Plot a task
        
        arguments:
            task : either a task read with `load_data` or a task name
    """
    
    if isinstance(task, str):
        task_path = next( data_path / p /task for p in ('training', 'evaluation','test') if (data_path / p / task).exists() )
        task = load_data(task_path)
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(n*4, 8))
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    
    def go(ax, title, x):
        if x is not None:
            ax.imshow(draw_one(x), interpolation='nearest')
        ax.set_title(title)
        ax.set_yticks([])
        ax.set_xticks([])
        
    for i, t in enumerate(task["train"]):
        go(axs[0][fig_num], f'Train-{i} in', t["input"])
        go(axs[1][fig_num], f'Train-{i} out', t["output"])
        fig_num += 1
    for i, t in enumerate(task["test"]):
        go(axs[0][fig_num], f'Test-{i} in', t["input"])
        go(axs[1][fig_num], f'Test-{i} out', t.get("output"))
        fig_num += 1
    
    plt.tight_layout()
    plt.show()
    
    
    
def trace_automata(step_fn, input, n_iter, n_hidden, loadbar=True):
    """
        Execute an automata and return all the intermediate states
        
        arguments:
            step_fn : transition rule function, should take two arguments `input` and `hidden_i`, 
                should return an output grid an a new hidden hidden grid
            n_iter : num of iteration to perform
            n_hidden: number of hidden grids, if set to 0 `hidden_i` will be set to None
            laodbar = True: weather display loadbars
        returns:
            an array of tuples if output and hidden grids
    """
    
    hidden = np.zeros((n_hidden, *input.shape)) if n_hidden > 0 else None
    
    trace = [(input, hidden)]
    
    its = range(n_iter)
    if loadbar:
        its = tqdm(its, desc='Step')
    for _ in its:
        output, hidden = step_fn(input, hidden)
        trace.append((output, hidden))        
        input = output
    return trace


def vis_automata_trace(states, loadbar=True, prefix_image=None): 
    """
        Create a video from an array of automata states
        
        arguments:
            states : array of automata steps, returned by `trace_automata()`
            loadbar = True: weather display loadbars
            prefix_image = None: image to add to the beginning of each frame 
        returns 
            a moviepy ImageSequenceClip
    """
    frames = []
    if loadbar:
        states = tqdm(states, desc='Frame')
    for i, (canvas, hidden) in enumerate(states):
        
        frame = []
        if prefix_image is not None:
            frame.append(prefix_image)
        frame.append(draw_one(canvas))
        if hidden is not None:
            frame.extend(draw_one(h) for h in hidden)
        frames.append(vcat_imgs(frame))            
        
    return ImageSequenceClip(list(map(np.array, frames)), fps=10)


from moviepy.editor import clips_array, CompositeVideoClip


from moviepy.video.io.html_tools import html_embed, HTML2

def display_vid(vid, verbose = False, **html_kw):
    """
        Display a moviepy video clip, useful for removing loadbars 
    """
    
    rd_kwargs = { 
        'fps' : 10, 'verbose' : verbose 
    }
    
    if not verbose:
         rd_kwargs['logger'] = None
    
    return HTML2(html_embed(vid, filetype=None, maxduration=60,
                center=True, rd_kwargs=rd_kwargs, **html_kw))


def vis_automata_task(tasks, step_fn, n_iter, n_hidden, vis_only_ix=None):
    """
        Visualize the automata steps during the task solution
        arguments:
            tasks : the task to be solved by the automata
            step_fn : automata transition function as passed to `trace_automata()`
            n_iter : number of iterations to perform
            n_hidden : number of hidden girds
    """
    
    n_vis = 0        
    
    def go(task, n_vis):
        
        if vis_only_ix is not None and vis_only_ix != n_vis:
            return 
        
        trace = trace_automata(step_fn, task['input'], n_iter, n_hidden, loadbar=False)
        vid = vis_automata_trace(trace, loadbar=False, prefix_image=draw_one(task['output']))
        display(display_vid(vid))
        
    
        
    for task in (tasks['train']):
        n_vis += 1
        go(task, n_vis)
        
    for task in (tasks['test']):
        n_vis += 1
        go(task, n_vis)
    
    
#
# Data IO
#

import os
import json

training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))

def load_data(p, phase=None):
    """
        Load task data
        
    """
    if phase in {'training', 'test', 'evaluation'}:
        p = data_path / phase / p
    
    task = json.loads(Path(p).read_text())
    dict_vals_to_np = lambda x: { k : np.array(v) for k, v in x.items() }
    assert set(task) == {'test', 'train'}
    res = dict(test=[], train=[])
    for t in task['train']:
        # assert set(t) == {'input', 'output'}
        res['train'].append(dict_vals_to_np(t))
    for t in task['test']:
        # assert set(t) == {'input', 'output'}
        res['test'].append(dict_vals_to_np(t))
        
    return res
