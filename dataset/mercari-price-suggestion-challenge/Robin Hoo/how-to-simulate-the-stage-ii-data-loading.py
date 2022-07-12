import gc,re,time,random,psutil,string,warnings,numpy as np,pandas as pd
warnings.filterwarnings("ignore")
##############################################
#
# PUBLIC FUNCTIONS
#
##############################################
def tick2time(tick):
    tick=int(tick)
    sec=tick % 60
    tick=tick/60
    mnt=tick%60
    tick=tick/60
    hrs=tick%24
    dys=tick//24
    if dys<1:
        return "%02d:%02d:%02d"%(hrs,mnt,sec)
    else:
        return "%dD:%02d:%02d:%02d"%(dys,hrs,mnt,sec)

def memory_checkpoint():
    if not hasattr(memory_checkpoint, "last_space"):
        memory_checkpoint.last_space=0
    gc.collect()
    used_mem=psutil.Process().memory_info().rss /1024**2
    diff_mem=used_mem-memory_checkpoint.last_space
    memory_checkpoint.last_space=used_mem
    return '>>Used Mem:%.3fM | Diff Mem:%.3fM<<'%(used_mem,diff_mem)
    
def time_checkpoint(msg=None,reset=True):
    if not hasattr(time_checkpoint, "start_tick"):
        time_checkpoint.start_tick=time.time()
        time_checkpoint.last_tick=time.time()
    if msg is None:
        print("[%s] Total Processed:%s|Last Step Processd:%s\n%s\n"%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),tick2time(time.time()-time_checkpoint.start_tick),tick2time(time.time()-time_checkpoint.last_tick),memory_checkpoint()),flush=True)
    else:
        if reset:
            time_checkpoint.last_tick=time.time()
        print("<%s> %s" % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),msg),flush=True)
    return

##############################################
#
# SPECIFIED FUNCTIONS
#
##############################################
def load_data_file(file_name,names):
    return pd.read_csv(file_name, sep='\t', engine='python', encoding='utf-8').rename(columns=names)

def shuffle_strings(x):
    global salts
    x = repr(x).split(' ')
    if len(salts)>0:
        for i in random.sample(range(len(x)),random.randint(0, len(x))):
            x[i]=salts.pop()
            if len(salts)<1:
                break
    return ' '.join(x)
    
def random_words(wc):
    wc=int(wc)
    ret=[]
    for _ in range(wc):
        ret.append(''.join(random.sample(string.ascii_letters, random.randrange(3, 15))))
    return ret
   
def shuffle_series(data):
    global salts
    for name in ['item_description','name']:
        data[name]=data[name].apply(shuffle_strings)
    return data


if __name__ == '__main__':
    time_checkpoint('Initializing Data...')
    path='../input/'
    time_checkpoint()

    time_checkpoint('Loading Source Data...')
    train=load_data_file(path+'train.tsv',{'train_id':'id'})
    test=pd.concat([load_data_file(path+'test.tsv',{'test_id':'id'}),
                    load_data_file(path+'test.tsv',{'test_id':'id'}),
                    load_data_file(path+'test.tsv',{'test_id':'id'}),
                    load_data_file(path+'test.tsv',{'test_id':'id'}),
                    load_data_file(path+'test.tsv',{'test_id':'id'})])
    word_counts=531906 #Word Numbers in Stage I denpends on the tokenizer
    salts=random_words(0.4*word_counts)
    data=pd.concat([train,shuffle_series(test)])
    del train,test
    gc.collect()
    time_checkpoint()
