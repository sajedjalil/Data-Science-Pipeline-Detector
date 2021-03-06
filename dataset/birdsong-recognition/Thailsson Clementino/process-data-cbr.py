import cv2
import warnings
import random
import numpy as np
import os
import pandas as pd
import librosa

warnings.filterwarnings('ignore')

def get_clip_sr(path,offset=0,duration=None):
    clip, sr_native = librosa.core.audio.__audioread_load(path, offset=offset, duration=duration, dtype=np.float32)
    clip = librosa.to_mono(clip)
    sr = 22050
    if sr_native > 0:
        clip = librosa.resample(clip, sr_native, sr, res_type='kaiser_fast')
    return clip, sr

BIRD_CODE = {
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

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

PATH_TRAIN = "../input/birdsong-recognition/train_audio/"
train_info = pd.read_csv('../input/birdsong-recognition/train.csv')

# The Function return the segments extracted from audio and the rest of audio
def segmentation(x,tr,sr):
    
    resp = [];
    
    while(len(x) >= (sr*2)):
        if(max(x) < tr):
            break
            
        time_amplitude_max = np.argmax(x)
        
        #Higher amplitude is before 1s 
        if(time_amplitude_max < ((1)*sr)):
            resp.append(x[:2*sr])
            x = x[2*sr:]
            
        #Higher amplitude is on the last 1s     
        elif(time_amplitude_max > (len(x) - ((1)*sr))):
            resp.append(x[-(2*sr):])
            x = x[:-2*sr]
            
        else:
            resp.append(x[time_amplitude_max-int((1)*sr):time_amplitude_max+int((1)*sr)])
            x = np.concatenate((x[:time_amplitude_max-int((1)*sr)-1],x[time_amplitude_max+int((1)*sr)+1:]))
            
            
    if(len(resp) == 0):
        resp = None
        
    return resp,x 

def normaliza(x):
    mi = x.mean()
    sigma = np.std(x)
    
    x = x-mi;
    return (x/sigma) 

def mono_to_color(X: np.ndarray,
                  mean=None,
                  std=None,
                  norm_max=None,
                  norm_min=None,
                  eps=1e-6):
    """
    Code from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data
    """
    # Stack X as [X,X,X]
    #X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V

def get_image(full_path):
    y, sr = get_clip_sr(full_path)
    
    y = normaliza(y)
    seg,_ = segmentation(y,10,sr)
    
    if(seg == None):
        return None
    
    else:
        ret = []
        for s in seg:
            
            #separating harmonic and percussive
            data_h, data_p = librosa.effects.hpss(s,margin=8)
            
            melspec = librosa.feature.melspectrogram(s,sr=sr,fmin=50,fmax=16000)
            melspec = librosa.power_to_db(melspec).astype(np.float32)
            
            melspec_h = librosa.feature.melspectrogram(data_h,sr=sr,fmin=50,fmax=16000)
            melspec_h = librosa.power_to_db(melspec_h).astype(np.float32)
            
            melspec_p = librosa.feature.melspectrogram(data_p,sr=sr,fmin=50,fmax=16000)
            melspec_p = librosa.power_to_db(melspec_p).astype(np.float32)
            
            image = mono_to_color(melspec)
            image_h = mono_to_color(melspec_h)
            image_p = mono_to_color(melspec_p)
            
            X = np.stack([image, image_h, image_p], axis=-1)       
            X = cv2.resize(X, (200, 40))
            
            ret.append(X)
            
            try: 
                #Gaussian Noise
                noise = np.random.randn(len(s))

                #Random audios without birds
                bad,sr = get_clip_sr('../input/files-to-help-process/BAD_Augmentation/BAD_Augmentation/{}.wav'.format(random.randint(0, 7709))
                                     ,offset=random.randint(0, 8),duration=2)
                bad = normaliza(bad) 


                b = False
                #add Gaussian
                if(random.random() > 0.5):
                    b = True
                    s = s + (random.randint(1, 5) * noise)

                #add Bad
                if(random.random() > 0.5):
                    b = True
                    s = s + (random.randint(1, 10) * bad)    


                if(b == True):
                    #separating harmonic and percussive
                    data_h, data_p = librosa.effects.hpss(s,margin=8)

                    melspec = librosa.feature.melspectrogram(s,sr=sr,fmin=50,fmax=16000)
                    melspec = librosa.power_to_db(melspec).astype(np.float32)

                    melspec_h = librosa.feature.melspectrogram(data_h,sr=sr,fmin=50,fmax=16000)
                    melspec_h = librosa.power_to_db(melspec_h).astype(np.float32)

                    melspec_p = librosa.feature.melspectrogram(data_p,sr=sr,fmin=50,fmax=16000)
                    melspec_p = librosa.power_to_db(melspec_p).astype(np.float32)

                    image = mono_to_color(melspec)
                    image_h = mono_to_color(melspec_h)
                    image_p = mono_to_color(melspec_p)

                    X = np.stack([image, image_h, image_p], axis=-1)       
                    X = cv2.resize(X, (200, 40))


                    ret.append(X)
            except:
                print("sad")
                
        return ret
    
    
os.mkdir('train_np')

images = []
labels = []

cont = 0

sp_cont = 0;

cont_ant =0
spec_ant = ''

for index, row in train_info.iterrows():

    ebird_code = row['ebird_code']
    if(ebird_code == 'bulori'):
        break
    if(ebird_code != 'aldfly' and spec_ant == ''):
        continue   
        
    if(ebird_code != spec_ant):
        print(spec_ant,'=',cont-cont_ant)
        print(ebird_code,cont)
        #os.mkdir('train/{}'.format(ebird_code))
        spec_ant = ebird_code
        cont_ant = cont
        sp_cont += 1

    duration = row['duration']
    if(duration < 2):
        continue
    full_path = PATH_TRAIN + ebird_code +'/' + row['filename']
    
    try:
        image = get_image(full_path);

        if(image == None):
            continue
        for im in image:

            #cv2.imwrite('{}.png'.format(str(cont)),im)
            np.savez_compressed('train_np/{}.npz'.format(str(cont)),im)

            images.append("{}".format(cont));
            labels.append(BIRD_CODE[ebird_code])
            cont += 1
    except:
        print('Fail!') 

np.save('filenames.npy',images)
np.save('labels.npy',labels)            
os.system("zip -r train_np.zip train_np")   
os.system("rm -r train_np") 