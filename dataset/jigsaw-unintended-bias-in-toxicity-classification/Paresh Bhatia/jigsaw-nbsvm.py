# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import re
import time
import warnings
import sqlite3
# from sqlalchemy import create_engine # database connection
import csv
import os
warnings.filterwarnings("ignore")
import datetime as dt
import numpy as np
import string
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import CountVectorizer

import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.metrics.classification import accuracy_score, log_loss
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from scipy.sparse import hstack
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold 
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import math
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from scipy import sparse
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from mlxtend.classifier import StackingClassifier
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from tqdm import tqdm
import pickle
import gc
import os
from xgboost import XGBClassifier,train,DMatrix,plot_importance
from sklearn.metrics import roc_auc_score
gc.enable();


train_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv",usecols=['comment_text','target','id'])
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")
#train_df = pd.read_csv("C:/Users/PareshBhatia/Downloads/Learning/Data Science/analytics vidya/bigmart/train/train.csv")
print('train data shape :',train_df.shape)
print('test data shape :',test_df.shape)

y_train = np.where(train_df['target'] > 0.5,1,0)

#Align training and test dataset to find common features
train_df,test_df = train_df.align(test_df,join='inner',axis=1)

print('Training Features shape: ', train_df.shape)
print('Testing Features shape: ', test_df.shape)

#let's join train and test data
data = pd.concat([train_df,test_df],axis=0)
data.head()

# expand the contractions
# https://stackoverflow.com/a/47091490/4084039
import re

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


# Combining all the above stundents 
from tqdm import tqdm
preprocessed_comments = []
# tqdm is for printing the status bar
for sentance in tqdm(data['comment_text'].values):
    sent = decontracted(sentance)
    sent = sent.replace('\\r', ' ')
    sent = sent.replace('\\"', ' ')
    sent = sent.replace('\\n', ' ')
    sent = sent.replace('\\t', ' ')
    # https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python
    sent = re.sub('http[s]?://\S+', '', sent) # remove html links
    sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
    # https://gist.github.com/sebleier/554280
    #sent = ' '.join(e for e in sent.split() if e not in stopwords)
    preprocessed_comments.append(sent.lower().strip())
    
data['clean_comment'] = preprocessed_comments

del preprocessed_comments;
gc.collect();
# Make sure all comment_text values are strings
data['comment_text'] = data['comment_text'].astype(str)
data['clean_comment'] = data['clean_comment'].astype(str)

# https://github.com/neptune-ml/open-solution-toxic-comments/blob/master/external_data/compiled_bad_words.txt
# list of bad words
bad_words = ['cockknocker','n1gger','f ing','fukker','nympho','fcuking','gook','freex','arschloch','fistfucked',
             'chinc','raunch','fellatio','splooge','nutsack','lmfao','wigger','bastard','asses','fistfuckings',
             'blue waffle','beeyotch','pissin','dominatrix','fisting','vullva','paki','cyberfucker','chuj',
             'penuus','masturbate','b00b*','fuks','sucked','fuckingshitmotherfucker','feces','panty','coital',
             'wh00r. whore','condom','hells','foreskin','wanker','hoer','sh1tz','shittings','wtf','recktum',
             'dick*','pr0n','pasty','spik','phukked','assfuck','xxx','nigger*','ugly','s_h_i_t','mamhoon',
             'pornos','masterbates','mothafucks','Mother Fukkah','chink','pussy palace','azazel','fistfucking',
             'ass-fucker','shag','chincs','duche','orgies','vag1na','molest','bollock','a-hole','seduce','Cock*',
             'dog-fucker','shitz','Mother Fucker','penial','biatch','junky','orifice','5hit','kunilingus','cuntbag','hump','butt fuck',
             'titwank','schaffer','cracker','f.u.c.k','breasts','d1ld0','polac','boobs','ritard','fuckup','rape','hard on',
             'skanks','coksucka','cl1t','herpy','s.o.b.','Motha Fucker','penus','Fukker','p.u.s.s.y.','faggitt','b!tch',
             'doosh','titty','pr1k','r-tard','gigolo','perse','lezzies','bollock*','pedophiliac','Ass Monkey','mothafucker',
             'amcik','b*tch','beaner','masterbat*','fucka','phuk','menses','pedophile','climax','cocksucking','fingerfucked',
             'asswhole','basterdz','cahone','ahole','dickflipper','diligaf','Lesbian','sperm','pisser','dykes','Skanky',
             'puuker','gtfo','orgasim','d0ng','testicle*','pen1s','piss-off','@$$','fuck trophy','arse*','fag','organ',
             'potty','queerz','fannybandit','muthafuckaz','booger','pussypounder','titt','fuckoff','bootee','schlong',
             'spunk','rumprammer','weed','bi7ch','pusse','blow job','kusi*','assbanged','dumbass','kunts','chraa','cock sucker','l3i+ch','cabron','arrse','cnut','how to murdep','fcuk','phuked','gang-bang','kuksuger','mothafuckers','ghey','clit licker','feg','ma5terbate','d0uche','pcp','ejaculate','nigur','clits','d0uch3','b00bs','fucked','assbang','mutha','goddamned','cazzo','lmao','godamn','kill','coon','penis-breath','kyke','heshe','homo','tawdry','pissing','cumshot','motherfucker','menstruation','n1gr','rectus','oral','twats','scrot','God damn','jerk','nigga','motherfuckin','kawk','homey','hooters','rump','dickheads','scrud','fist fuck','carpet muncher','cipa','cocaine','fanyy','frigga','massa','5h1t','brassiere','inbred','spooge','shitface','tush','Fuken','boiolas','fuckass','wop*','cuntlick','fucker','bodily','bullshits','hom0','sumofabiatch','jackass','dilld0','puuke','cums','pakie','cock-sucker','pubic','pron','puta','penas','weiner','vaj1na','mthrfucker','souse','loin','clitoris','f.ck','dickface','rectal','whored','bookie','chota bags','sh!t','pornography','spick','seamen','Phukker','beef curtain','eat hair pie','mother fucker','faigt','yeasty','Clit','kraut','CockSucker','Ekrem*','screwing','scrote','fubar','knob end','sleazy','dickwhipper','ass fuck','fellate','lesbos','nobjokey','dogging','fuck hole','hymen','damn','dego','sphencter','queef*','gaylord','va1jina','a55','fuck','douchebag','blowjob','mibun','fucking','dago','heroin','tw4t','raper','muff','fitt*','wetback*','mo-fo','fuk*','klootzak','sux','damnit','pimmel','assh0lez','cntz','fux','gonads','bullshit','nigg3r','fack','weewee','shi+','shithead','pecker','Shytty','wh0re','a2m','kkk','penetration','kike','naked','kooch','ejaculation','bang','hoare','jap','foad','queef','buttwipe','Shity','dildo','dickripper','crackwhore','beaver','kum','sh!+','qweers','cocksuka','sexy','masterbating','peeenus','gays','cocksucks','b17ch','nad','j3rk0ff','fannyflaps','God-damned','masterbate','erotic','sadism','turd','flipping the bird','schizo','whiz','fagg1t','cop some wood','banger','Shyty','f you','scag','soused','scank','clitorus','kumming','quim','penis','bestial','bimbo','gfy','spiks','shitings','phuking','paddy','mulkku','anal leakage','bestiality','smegma','bull shit','pillu*','schmuck','cuntsicle','fistfucker','shitdick','dirsa','m0f0','Fukkin','testis','ejaculatings','phuq','Shitty','crap','hooker','niggaz','fucknut','cyalis','anus','crabs','asswipes','cameltoe','cuntlicking','cuntz','corksucker','peepee','thug','jiz','gayz','fag*','cumstain','nepesaurio','dike*','8ss','shited','snatch','dick shy','opiate','butthole','whores','boner','pimpis','motherfuckka','slut','testee','futkretzn','mothafucka','cyberfuckers','cuntlicker','adult','tramp','blumpkin','fannyfucker','beotch','flange','dik','dildos','nipple','queers','boink','shamedame','shitty','tits','felching','felcher','gangbangs','punkass','orgasims','kunt','boob','sniper','pussee','pussys','cunillingus','pula','50 yard cunt punt','wad','s t f u','jism','Felcher','son-of-a-bitch','shitty','cocksucked','faen','basterds','heeb','hebe','n1gga','phallic','pube','fck','cunt hair','fatass','cunthunter','prig','Phuck','dickbag','titi','Fukken','balls','pissers','gooks','muff puff','ballbag','eat a dick','clit','kyrpa*','knobed','penetrate','ballsack','ejakulate','c0cks','bowels','f u c k','suka','Fudge Packer','hui','buttfucker','goatse','smut','bosom','Fukah','pastie','assholz','boooobs','l3itch','lezbo','godamnit','fuckheads','g-spot','niggers','w00se','wichser','v14gra','orgasmic','hitler','helvete','snuff','master-bate','motherfuck','bust a load','sissy','s-h-i-t','steamy','sucking','damned','pricks','fukkin','willies','erect','knulle','fistfuck','pisses','toots','bone','tinkle','punky','nads','goddamn','pimp','arian','frigg','f uck','jack-off','vomit','butt','peyote','muie','lust','dickdipper','goddammit','racy','v1gra','orgasm','nazism','flog the log','buttfuck','clitty','dumass','Poonani','fondle','amateur','Mutha Fucker','faggit','bitching','cocksuck','Phuk','puto','corp whore','kinky','japs','Mother Fukah','pussy','monkleigh','muthafucker','cocks','h0mo','fuckme','dumbasses','ejaculated','carpetmuncher','pollock','bollocks','honkey','bitchers','a s s','shitt','pigfucker','lusty','sleaze','teabagging','mothafucking','qweir','pawn','twunt','skurwysyn','motherfucking','muthrfucking','gassy ass','dominatrics','pantie','masterbations','hookah','bimbos','a55hole','loins','c-0-c-k','fvck','slave','masterb8','hore','cockface','sh1t','cum guzzler','how to kill','muschi','sluts','fook','pr1ck','knobjokey','t1tt1e5','niggas','packi','mothafucked','lesbian','bitch','gangbang','muthafuckker','booooobs','undies','gay','goldenshower','cockhead','quicky','vulva','junkie','shemale','gai','shiting','c-u-n-t','beastial','vodka','lezbos','sh!t*','beardedclam','bitched','skankee','stiffy','spac','scroat','beatch','d*ck','Fukkah','fisted','buceta','dominatricks','revue','arsehole','pot','dawgie-style','assfukka','packy','ash0le','fuckwhit','tosser','bangbros','lesbo','pierdol*','cuntface','asswipe','kondum','kuntz','blowjobs','felch','Shyt','ovum','stfu','jerk0ff','bastardz','spic','jackoff','fisty','chodes','bellend','orgasim','s-o-b','muffdiver','fagged','diddle','slut bucket','wench','kikes','moron','doofus','dinks','guiena','ar5e','mothafuckin','hotsex','womb','cocain','c.0.c.k','birdlock','stoned','nazi','ficken','beastiality','pussy fart','sodom','jerked','tubgirl','opium','Fu(*','mthrfucking','fuck-ass','phukking','Lipshitz','hootch','bung','fucknugget','fingerfuckers','shitting','douchebags','reetard','testes','dipship','bitch tit','jisim','poop','fucktard','tittywank','fanculo','busty','faggs','valium','fucks','fuk','murder','s-h-1-t','crack','hussy','orafis','mothafuckings','kooches','vixen','dimwit','extasy','herpes','hoorem whore','tard','wang','foobar','xrated','c.o.c.k.','sharmute','h0m0','ass-hole','cum','cervix','azz','ham flap','pillowbiter','nappy','orifiss','s.h.i.t.','knobead','assbangs','hooter','Lipshits','fucktoy','faigs','ayir','bunny fucker','scantily','cokmuncher','menstruate','bullturds','enlargement','herp','scum','semen','gonad','dyke','knob','woody','m-fucking','cumdump','ninny','bukkake','rtard','dingle','uterus','pissed','teets','jizzed','arse','lezbians','assmunch','rapist','fag1t','h0r','bitchy','napalm','fagging','glans','fuckin','testicle','rum','Mother Fukker','preteen','cunt','bullshitted','fooker','lezzie','vittu','strip','cawks','shithouse','bloody','queaf','t1t','shitfull','cunilingus','anilingus','ovums','skank','bitchin','fagot','scrog','mothafuckas','dziwka','asholes','whoar','wank*','Mutha Fukah','h0ar','cummer','clusterfuck','cunt-struck','cummin','bootie','dicksipper','whore','homoey','whoring','douche','poontang','blow me','kums','ruski','Fukk','need the dick','schlampe','d!ck','cok','piss','dild0s','faggot','blow mud','fuckers','shit fucker','f_u_c_k','s hit','lesbians','jizm','enculer','fagg','fuck puppet','dick-ish','mofo','hardcoresex','maxi','shitter','ejackulate','Shyte','screwed','twatty','gangbanged','meth','dirty Sanchez','wh0reface','cawk','cockblock','fistfucks','cyberfuc','teat','he11','ovary','zabourah','humped','boozer','stroke','LEN','andskota','pedophilia','qweerz','wedgie','smutty','crappy','sh1tter','commie','phuks','leper','autoerotic','tit wank','cockmunch','reich','b1tch','bong','Motha Fuker','Fotze','titt*','masterbation','yed','coons','fuc','slutdumper','pubis','fanny','hooch','vulgar','a_s_s','babes','sexual','packie','fags','masterbat3','floozy','virgin','gae','whorealicious','scheiss*','drunk','Mutha Fuker','assrammer','klan','queero','fistfuckers','bod','cocksmoker','fudgepacker','mams','kock','ganja','mutherfucker','faget','clitty litter','sadist','voyeur','bitcher','doggiestyle','nigger','cyberfuck','pussi','niglet','nigg4h','fcuker','perversion','porn','rimming','areola','c0cksucker','bosomy','unwed','ass hole','wazoo','jiz','cock pocket','goddam','gspot','fuker','shitted','whoralicious','bitches','asshat','Sh!t','masturbat*','knobjocky','Biatch','buttplug','Fukin','prude','lezbian','Slutty','f4nny','bi+ch','dickweed','gayboy',"bang (one's) box",'horniest','enema','booze','cumslut','bawdy','mothafuckaz','*dyke','c0k','slope','bowel','mof0','cnts','titfuck','chode','d1ck','pissoff','booobs','Carpet Muncher','cornhole',
             'fuck-tard','cocksucker','picka','cunnilingus','h0re','cunt*','hiv','jizm','motherfucka','oriface',
             'omg','dickish','sh1ter','whorehopper','jerk-off','fu','h00r','kuk','dammit','hobag','zoophile',
             'Mother Fuker','daygo','phalli','orgy','polak','masterbaiter','transsexual','uzi','Flikker','mothafuck','fecker','Assface','barf','jiss','sharmuta','humping','booty','bassterds','panties','jerkoff','retarded','boozy','slutkiss','fuckhead','fagz','cuntlick','slutz','stupid','wanky','cox','motherfucks','assmaster','bitch*','sex','cockholster','pedo','ghay','fuck yo mama','shagging','suck','vajina','dick','scrotum','Phuker','shithole','booby','dike','god','nob jokey','trashy','lez','anal','motherfuckings','aeolus','dickhead','mouliewop','areole','*fuck*','douchey','teste','cum dumpster','piss*','polack','masstrbate','niggers','dupa','Phuc','doggin','thrust','knobend','cumshots','rectum','fuckface','assmucus','bum','tittiefucker','doggie style','whitey','sausage queen','a$$','beater','dummy','sandbar','puss','dagos','assholes','fagots','bastards','choade','guido','pissflaps','testical','Goddamn','hoorm whore','douch3','dink','cut rope','dickzipper','moolie','queer','skanck','bollok','paky','labia','twunter','teez','pms','orgasms','breast','niiger','seaman','whorehouse','fartknocker','buttmuch','mtherfucker','butt-pirate','tit','ass','merd*','negro','peinus','horny','peeenusss','phonesex','Ekto','vagina','Blow Job','d4mn','azzhole','f u c k e r','c0ck','Motha Fukkah','boobies','pee','cock','tittyfuck','muther','fuckwit','muthafecker','m0fo','wetback','doggie-style','nobhead','Kurac','incest','assho1e','g00k','shits','ma5terb8','bugger','dopey','knobs','blow','pizda','weenie','babe','w t f','shite','x-rated','lube','cunts','toke','Mutha Fukkah','masochist','god-damned','Motha Fukker','niggle','hoer*','facial','willy','idi0t','shit','dild0','hemp','boffing','feck','@ss','urinal','paska*','god damn','cock snot','aryan','feltch','mother-fucker','penile','shiz','fuckings','qahbeh','cockmuncher','god-dam','honky','orospu','feltcher','niigr','dlck','shiteater','fart','pule','big tits','masturbating','hoor','Huevon','massterbait','fingerfucks','Lezzian','dick hole','vag','injun','spierdalaj','orgasum','fuck-bitch','poontsee','ash0les','peenus','fuckwad','m45terbate','knobhead','b!+ch','dillweed','nut butter','masturbation','jizz','d1ldo','asshole','tittyfucker','kinky Jesus','fingerfucker','yobbo','caca','handjob','extacy','lech','shitters','pussies','twat','pinko','pissoff','cocksukka','sh#t','kwif','fukwhit','motherfuckers','kanker*','fingerfucking','tampon','cum freak','whoreface','skag','f-u-c-k','shitey','gaygirl','masstrbait','hoar','penisfucker','fu ck','cum chugger','erection','ejaculates','cuntlicker','rautenberg','skankey','kummer','weirdo','p0rn','*damn','fagit','nooky','bigtits','nimrod','lezzy','assh0le','booooooobs','jackhole','homoerotic','bra','nazis','phuck','bareback','doggy-style','rimjaw','kurwa','w0p','kootch','mutherfucking','fuckmeat','numbnuts','fukwit','jerk-off','boners','beer','raped','shipal','kondums','*shit*','mafugly','cock-head','cyberfucked','assfucker','fxck','wop','nob','wank','s0b','shagger','essohbee','anal impaler','boned','terd','urine','faig','s.o.b','cyberfucking','titties','c.u.n.t','masokist','sh1ts','fudge packer','prostitute','nobjocky','nastt','niggah','skribz','dilld0s','fingerfuck','fat','shitfuck','analprobe','gey','motherfucked','4r5e','porno','gringo','gaysex','Mutha Fukker','retard','rimjob','ejaculating','fux0r','knobz','h4x0r','lusting','tittie5','reefer','poon','orally','donkeyribber','t1tties','screw','cunny','c-o-c-k','dong','cumming','prick','viagra','shaggin','vagiina','pr1c','twathead','hell']

# Get length in words and characters
data["raw_word_len"] = data["comment_text"].apply(lambda x: len(x.split()))
data["raw_char_len"] = data["comment_text"].apply(lambda x: len(x))


# Get the new length in words and characters
data["clean_word_len"] = data["clean_comment"].apply(lambda x: len(x.split()))
data["clean_char_len"] = data["clean_comment"].apply(lambda x: len(x))

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))

# Check number of upper case, if you're angry you may write in upper case
data["nb_upper"] = data["comment_text"].apply(lambda x: count_regexp_occ(r"[A-Z]", x))
# Number of F words - f..k contains folk, fork,
data["nb_fk"] = data["comment_text"].apply(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
# Number of S word
data["nb_sk"] = data["comment_text"].apply(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
# Number of D words
data["nb_dk"] = data["comment_text"].apply(lambda x: count_regexp_occ(r"[dD]ick", x))
# Number of occurence of You, insulting someone usually needs someone called : you
data["nb_you"] = data["comment_text"].apply(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
# Just to check you really refered to my mother ;-)
data["nb_mother"] = data["comment_text"].apply(lambda x: count_regexp_occ(r"\Wmother\W", x))
data['contain_bad_words'] = data['clean_comment'].apply(lambda x: 0 if list(set(x.split(' '))& set(bad_words))==[] else 1)
                                                        
# number of bad words
data['num_bad_words'] = data['clean_comment'].apply(lambda x:  len(list(set(x.split(' '))& set(bad_words))))                                                        

train_df = data[:len(train_df)]
test_df = data[len(train_df):]
print('Training data shape: ', train_df.shape)
print('Testing data shape: ', test_df.shape)

del data;
gc.collect();
print('Delete unused variables 1!')
num_features = ['raw_word_len','raw_char_len','clean_word_len','clean_char_len','contain_bad_words',
                'num_bad_words','nb_upper','nb_fk','nb_sk','nb_dk','nb_you','nb_mother']

# normalize numeric
norm = Normalizer()
norm.fit(train_df[num_features])
train_num_feat = norm.transform(train_df[num_features])
test_num_feat = norm.transform(test_df[num_features])

text = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return text.sub(r' \1 ', s).split()

# vectorize text
Vectorize = TfidfVectorizer(ngram_range=(1,3), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1)
Vectorize.fit(train_df['clean_comment'])
train_df_text = Vectorize.transform(train_df['clean_comment'])
test_df_text = Vectorize.transform(test_df['clean_comment'])
del Vectorize,norm;
print(train_df_text.shape)
print(test_df_text.shape)

final_trn = hstack([train_df_text,train_num_feat]).tocsr()
final_test = hstack([test_df_text,test_num_feat]).tocsr()
submission = pd.DataFrame.from_dict({'id': test_df['id']})
del train_df,test_df,train_df_text,train_num_feat,test_df_text,test_num_feat;
gc.collect();

print("Data Preprocessing done!")

# https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline
class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual=False, n_jobs=1,max_iter=100):
        self.C = C
        self.dual = dual
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        y = y
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)
        
        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs,max_iter=self.max_iter).fit(x_nb, y)
        return self


NbSvm = NbSvmClassifier(C=1.5, dual=True, n_jobs=-1)
NbSvm.fit(final_trn, y_train)

print("Model fit done!")
MODEL_NAME = 'NbSvm'


submission["prediction"] = NbSvm.predict_proba(final_test)[:, 1]
submission.to_csv("submission.csv", index=False)