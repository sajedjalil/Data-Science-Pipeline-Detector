import os
import sys
import multiprocessing
import pathlib 


import numpy as np
import librosa
import cv2
import pandas as pd
import time
from timeit import default_timer as timer
import warnings
import torch


#### setting ########################################################################################
print('torch version:', torch.__version__)
print('\ttorch.cuda.get_device_properties() = %s' % torch.cuda.get_device_properties(0))
print('\tcpu_count = %d' % multiprocessing.cpu_count())
print('\tram = %d MB' % (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') /(1024.**3)))
print('')

# https://www.kaggle.com/shonenkov/sample-submission-using-custom-check
# https://www.kaggle.com/c/birdsong-recognition/discussion/159993

if 0: #local
    ADD_DIR    = os.path.dirname(__file__)
    checkpoint = '/root/share1/kaggle/2020/birdsong/result/reference/resnet50/best_model.pth'
    TEST_CSV   = '/root/share1/kaggle/2020/birdsong/data/other/sample_test/test.csv'
    TEST_AUDIO_DIR = '/root/share1/kaggle/2020/birdsong/data/other/sample_test/audio'

if 1:# kaggle
   
    ADD_DIR  = '../input/bird00/old'
    TEST_AUDIO_DIR = '../input/birdsong-recognition/test_audio'
    TEST_CSV = '../input/birdsong-recognition/test.csv'

    if not os.path.exists('../input/birdsong-recognition/test_audio'):
        TEST_AUDIO_DIR = '../input/birdcall-check/test_audio'
        TEST_CSV = '../input/birdcall-check/test.csv'

#-------------------------
sys.path.append(ADD_DIR)
print('sys.path.append(ADD_DIR) OK!')
from model import *

net = Net()
state_dict = torch.load(ADD_DIR + '/checkpoint/best_model.pth', map_location=lambda storage, loc: storage)
net.load_state_dict(state_dict, strict=True)  # True
print('net.load_state_dict() OK!')
print('')

########################################################################################################

NAME_TO_LABEL = {
    'aldfly': 0, 'ameavo': 1, 'amebit': 2, 'amecro': 3, 'amegfi': 4,
    'amekes': 5, 'amepip': 6, 'amered': 7, 'amerob': 8, 'amewig': 9,
    'amewoo': 10, 'amtspa': 11, 'annhum': 12, 'astfly': 13, 'baisan': 14,
    'baleag': 15, 'balori': 16, 'banswa': 17, 'barswa': 18, 'bawwar': 19,
    'belkin1': 20, 'belspa2': 21, 'bewwre': 22, 'bkbcuc': 23, 'bkbmag1': 24,
    'bkbwar': 25, 'bkcchi': 26, 'bkchum': 27, 'bkhgro': 28, 'bkpwar': 29,
    'bktspa': 30, 'blkpho': 31, 'blugrb1': 32, 'blujay': 33, 'bnhcow': 34,
    'boboli': 35, 'bongul': 36, 'brdowl': 37, 'brebla': 38, 'brespa': 39,
    'brncre': 40, 'brnthr': 41, 'brthum': 42, 'brwhaw': 43, 'btbwar': 44,
    'btnwar': 45, 'btywar': 46, 'buffle': 47, 'buggna': 48, 'buhvir': 49,
    'bulori': 50, 'bushti': 51, 'buwtea': 52, 'buwwar': 53, 'cacwre': 54,
    'calgul': 55, 'calqua': 56, 'camwar': 57, 'cangoo': 58, 'canwar': 59,
    'canwre': 60, 'carwre': 61, 'casfin': 62, 'caster1': 63, 'casvir': 64,
    'cedwax': 65, 'chispa': 66, 'chiswi': 67, 'chswar': 68, 'chukar': 69,
    'clanut': 70, 'cliswa': 71, 'comgol': 72, 'comgra': 73, 'comloo': 74,
    'commer': 75, 'comnig': 76, 'comrav': 77, 'comred': 78, 'comter': 79,
    'comyel': 80, 'coohaw': 81, 'coshum': 82, 'cowscj1': 83, 'daejun': 84,
    'doccor': 85, 'dowwoo': 86, 'dusfly': 87, 'eargre': 88, 'easblu': 89,
    'easkin': 90, 'easmea': 91, 'easpho': 92, 'eastow': 93, 'eawpew': 94,
    'eucdov': 95, 'eursta': 96, 'evegro': 97, 'fiespa': 98, 'fiscro': 99,
    'foxspa': 100, 'gadwal': 101, 'gcrfin': 102, 'gnttow': 103, 'gnwtea': 104,
    'gockin': 105, 'gocspa': 106, 'goleag': 107, 'grbher3': 108, 'grcfly': 109,
    'greegr': 110, 'greroa': 111, 'greyel': 112, 'grhowl': 113, 'grnher': 114,
    'grtgra': 115, 'grycat': 116, 'gryfly': 117, 'haiwoo': 118, 'hamfly': 119,
    'hergul': 120, 'herthr': 121, 'hoomer': 122, 'hoowar': 123, 'horgre': 124,
    'horlar': 125, 'houfin': 126, 'houspa': 127, 'houwre': 128, 'indbun': 129,
    'juntit1': 130, 'killde': 131, 'labwoo': 132, 'larspa': 133, 'lazbun': 134,
    'leabit': 135, 'leafly': 136, 'leasan': 137, 'lecthr': 138, 'lesgol': 139,
    'lesnig': 140, 'lesyel': 141, 'lewwoo': 142, 'linspa': 143, 'lobcur': 144,
    'lobdow': 145, 'logshr': 146, 'lotduc': 147, 'louwat': 148, 'macwar': 149,
    'magwar': 150, 'mallar3': 151, 'marwre': 152, 'merlin': 153, 'moublu': 154,
    'mouchi': 155, 'moudov': 156, 'norcar': 157, 'norfli': 158, 'norhar2': 159,
    'normoc': 160, 'norpar': 161, 'norpin': 162, 'norsho': 163, 'norwat': 164,
    'nrwswa': 165, 'nutwoo': 166, 'olsfly': 167, 'orcwar': 168, 'osprey': 169,
    'ovenbi1': 170, 'palwar': 171, 'pasfly': 172, 'pecsan': 173, 'perfal': 174,
    'phaino': 175, 'pibgre': 176, 'pilwoo': 177, 'pingro': 178, 'pinjay': 179,
    'pinsis': 180, 'pinwar': 181, 'plsvir': 182, 'prawar': 183, 'purfin': 184,
    'pygnut': 185, 'rebmer': 186, 'rebnut': 187, 'rebsap': 188, 'rebwoo': 189,
    'redcro': 190, 'redhea': 191, 'reevir1': 192, 'renpha': 193, 'reshaw': 194,
    'rethaw': 195, 'rewbla': 196, 'ribgul': 197, 'rinduc': 198, 'robgro': 199,
    'rocpig': 200, 'rocwre': 201, 'rthhum': 202, 'ruckin': 203, 'rudduc': 204,
    'rufgro': 205, 'rufhum': 206, 'rusbla': 207, 'sagspa1': 208, 'sagthr': 209,
    'savspa': 210, 'saypho': 211, 'scatan': 212, 'scoori': 213, 'semplo': 214,
    'semsan': 215, 'sheowl': 216, 'shshaw': 217, 'snobun': 218, 'snogoo': 219,
    'solsan': 220, 'sonspa': 221, 'sora': 222, 'sposan': 223, 'spotow': 224,
    'stejay': 225, 'swahaw': 226, 'swaspa': 227, 'swathr': 228, 'treswa': 229,
    'truswa': 230, 'tuftit': 231, 'tunswa': 232, 'veery': 233, 'vesspa': 234,
    'vigswa': 235, 'warvir': 236, 'wesblu': 237, 'wesgre': 238, 'weskin': 239,
    'wesmea': 240, 'wessan': 241, 'westan': 242, 'wewpew': 243, 'whbnut': 244,
    'whcspa': 245, 'whfibi': 246, 'whtspa': 247, 'whtswi': 248, 'wilfly': 249,
    'wilsni1': 250, 'wiltur': 251, 'winwre3': 252, 'wlswar': 253, 'wooduc': 254,
    'wooscj2': 255, 'woothr': 256, 'y00475': 257, 'yebfly': 258, 'yebsap': 259,
    'yehbla': 260, 'yelwar': 261, 'yerwar': 262, 'yetvir': 263
}
LABEL_TO_NAME = {v: k for k, v in NAME_TO_LABEL.items()}

IMAGE_SIZE = 224
SR  = 32000 #sampling rate
LEN = 5 # 5 sec window
MELSPECTRUM = {
    'n_mels': 128,
    'fmin'  :  20,
    'fmax'  : 16000,
}

#------------------------------
def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError

#------------------------------
def melspec_norm_value(m):
    eps = 1e-6
    mean = m.mean()
    std  = m.std()
    m = (m-mean) / (std + eps)
    min, max = m.min(), m.max()
    if (max - min) > eps:
        m = (m - min) / (max - min)
    else:
        m = np.zeros_like(m)
    return m


def melspec_norm_size(m):
    height, width = m.shape
    m = cv2.resize(m, dsize=(int(width * IMAGE_SIZE / height), IMAGE_SIZE))
    return m

def make_batch(wave, second):
    melspec = []
    for s in second:
        t0 = int((s-5)*SR)
        t1 = int(s*SR)
        x = wave[t0:t1]
        m = librosa.feature.melspectrogram(x, sr=SR, **MELSPECTRUM)
        m = librosa.power_to_db(m)
        m = m.astype(np.float32)
        m = melspec_norm_value(m)
        m = melspec_norm_size(m)
        melspec.append(m)
    melspec = np.stack(melspec)
    return melspec


#augmentation by shift
def make_batch1(wave, second):
    wave = np.concatenate([np.zeros(int(0.1*SR),np.float32), wave])
    melspec = []
    for s in second:
        t0 = int((s-5)*SR)
        t1 = int(s*SR)
        x = wave[t0:t1]
        m = librosa.feature.melspectrogram(x, sr=SR, **MELSPECTRUM)
        m = librosa.power_to_db(m)
        m = m.astype(np.float32)
        m = melspec_norm_value(m)
        m = melspec_norm_size(m)
        melspec.append(m)
    melspec = np.stack(melspec)
    return melspec
#------------------------------

def run_submit():

    net.eval().cuda()

    #----
    df_submit = pd.DataFrame(columns=('row_id','birds'))
    df_test = pd.read_csv(TEST_CSV)
    # site,row_id,seconds,audio_id


    start_timer = timer()
    warnings.filterwarnings('ignore')
    for audio_id in df_test.audio_id.unique():
        df = df_test[df_test.audio_id == audio_id].reset_index(drop=True).sort_values('seconds')
        wave, _ = librosa.load( TEST_AUDIO_DIR + '/%s.mp3'%audio_id, sr=SR, mono=True, res_type='kaiser_fast')
        wave = wave.astype(np.float32)

        L = len(wave)
        site = df.site.values[0]
        if site == 'site_3':
            second = (np.arange(L//(SR*LEN))+1)*5
        else:
            second = df.seconds.values.astype(np.int32)

        print(audio_id, site, time_to_str(timer() - start_timer, 'min'))
        #print('\tlen = %0.2f, num_sec = %d '%(L/SR, len(second)), second[:5], '...',)
        
        #------
        melspec = make_batch(wave, second)

        probability = []
        L = len(melspec)
        batch_size = 16
        for m in np.array_split(melspec, int(np.ceil(L/batch_size))):
            #print('\tmelspec:', m.shape, '%0.2f mb'%(m.nbytes/1024/1024))

            m = torch.from_numpy(m).unsqueeze(1).cuda()
            with torch.no_grad():
                logit = net(m)
                p = F.sigmoid(logit)
                p = p.data.cpu().numpy()
            probability.append(p)

        probability = np.concatenate(probability)
        
        #------
        melspec1 = make_batch1(wave, second)

        probability1 = []
        L = len(melspec1)
        batch_size = 16
        for m in np.array_split(melspec1, int(np.ceil(L/batch_size))):
            #print('\tmelspec:', m.shape, '%0.2f mb'%(m.nbytes/1024/1024))

            m = torch.from_numpy(m).unsqueeze(1).cuda()
            with torch.no_grad():
                logit = net(m)
                p = F.sigmoid(logit)
                p = p.data.cpu().numpy()
            probability1.append(p)

        probability1 = np.concatenate(probability1)
        #------
        
        probability = (probability+probability1)/2
        predict = probability>=0.65
        if site == 'site_3':
            predict = predict.max(0, keepdims=True)
        #print('\tpredict:', predict.shape)

        if audio_id=='41e6fe6504a34bf6846938ba78d13df1' or audio_id=='07ab324c602e4afab65ddbcc746c31b5': #debug
            print(probability.reshape(-1)[:50], '\n')

        for b,row_id in enumerate(df.row_id.values):
            bird = np.where(predict[b])[0]
            if len(bird)==0:
                bird = 'nocall'
            else:
                bird = list(map(lambda i: LABEL_TO_NAME[i], bird))
                bird = ' '.join(bird)

            df_submit = df_submit.append({'row_id': row_id, 'birds': bird}, ignore_index=True)
    print('')

    #-----
    df_submit.to_csv('submission.csv', index=False)
    print('submission.csv')
    print(df_submit)

##########################################################################################
run_submit()

'''
41e6fe6504a34bf6846938ba78d13df1 site_1  0 hr 00 min
[9.6844548e-01 4.4374904e-08 2.8657375e-06 7.6718761e-06 2.2678425e-06
 9.7583381e-07 6.9485391e-06 3.9677274e-05 1.5725307e-05 6.4035603e-03
 2.9590586e-07 4.2635641e-08 1.3278045e-06 3.9484166e-04 2.3019457e-09
 7.7411505e-06 7.1928621e-04 6.6618441e-08 1.5015754e-08 5.6424742e-06
 2.0788360e-05 8.3228406e-07 2.4241261e-08 4.0079800e-07 2.4664134e-08
 5.3267147e-05 1.6295008e-06 3.0227498e-11 9.8451055e-06 2.3996377e-05
 3.0674728e-06 2.7398003e-06 2.8863001e-06 2.3529196e-06 1.4004944e-04
 9.4320836e-07 5.6368211e-08 1.0179133e-09 2.6775278e-11 2.2645603e-08
 7.3218524e-07 3.6195725e-06 7.0180278e-08 1.4193380e-05 6.1931922e-07
 1.6738483e-07 4.6992882e-05 4.1729211e-07 9.9444674e-07 2.1350264e-05]
 
 
submission.csv
                                        row_id          birds
0    site_1_41e6fe6504a34bf6846938ba78d13df1_5         aldfly
1   site_1_41e6fe6504a34bf6846938ba78d13df1_10         aldfly
2   site_1_41e6fe6504a34bf6846938ba78d13df1_15         aldfly
3   site_1_41e6fe6504a34bf6846938ba78d13df1_20         nocall
4   site_1_41e6fe6504a34bf6846938ba78d13df1_25         aldfly
..                                         ...            ...
71     site_3_9cc5d9646f344f1bbb52640a988fe902  aldfly comyel
72     site_3_a56e20a518684688a9952add8a9d5213         aldfly
73     site_3_96779836288745728306903d54e264dd  aldfly hamfly
74     site_3_f77783ba4c6641bc918b034a18c23e53         nocall
75     site_3_856b194b097441958697c2bcd1f63982         aldfly
'''












