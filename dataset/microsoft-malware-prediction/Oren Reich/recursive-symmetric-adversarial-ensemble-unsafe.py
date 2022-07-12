import numpy as np
import pandas as pd
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adadelta, SGD, RMSprop
import datetime
import sys
import os
import re
import codecs
import os.path
import csv

#------------------------------------------------------------------------------------#

KAGGLE = True
DATA_FILE = "../input/train.csv" if KAGGLE else "train.csv"

COMP_MEM = 0
COMP_PROJ_1 = 1
COMP_PROJ_2 = 2
COMP_PRET_1 = 3
COMP_PRET_2 = 4
COMP_GATE_IN = 5
COMP_GATE_OUT = 6
COMP_OP = 7

HASH_SIZE = 8
ALLOW_NON_NUMERIC = True
DATA_USE_PERCENT = 0.001
EPOCHS_PER_FIT = 1
CYCLES_PER_GLOBAL_EPOCH = 1000
IN_SIZE = 90
NUM_SYNAPSES = 8
OUT_SIZE = 1
NUM_FLAGS = 1
SYN_SIZE = HASH_SIZE * IN_SIZE
LATENT_DIM = int(SYN_SIZE / NUM_SYNAPSES)
TARGET_INTELLIGENCE_SIGNAL = 1

print("synapse size: ", SYN_SIZE)
print("latent size: ", LATENT_DIM)

targets = np.array([float(TARGET_INTELLIGENCE_SIGNAL)]).reshape((1, NUM_FLAGS))
synapses = []

#------------------------------------------------------------------------------------#

def softhash(s):
    a = np.zeros(HASH_SIZE)
    chars = list(s)

    for x in range(0, len(chars)):
        cx = chars[x]
        sx = str(cx)
        if len(sx)==1:
            ascii_code = ord(cx)
            if ALLOW_NON_NUMERIC or sx.isnumeric():
                wf = ascii_code if not sx.isnumeric() else float(sx) / 10 * 256
                if x >= HASH_SIZE:
                    a[x % HASH_SIZE] = (a[x % HASH_SIZE] + wf) / 2
                else:
                    a[x] = wf

    return a/255

def create_synapse(): 
    memory_input = Input(shape=(SYN_SIZE,))

    pretender1_input = Input(shape=(SYN_SIZE,))
    pretender2_input = Input(shape=(LATENT_DIM,))

    gate_in = Sequential()
    gate_in.add(Dense(SYN_SIZE, input_shape=(SYN_SIZE,), activation='relu'))
    gate_in.add(Dropout(1-1 / NUM_SYNAPSES))
    gate_in.add(Dense(LATENT_DIM, activation='hard_sigmoid'))

    gate_out = Sequential()
    gate_out.add(Dense(SYN_SIZE, input_shape=(LATENT_DIM,), activation='relu'))
    gate_out.add(Dropout(1-1 / NUM_SYNAPSES))
    gate_out.add(Dense(SYN_SIZE, activation='sigmoid'))

    task = Sequential()
    task.add(Dense(SYN_SIZE, input_shape=(SYN_SIZE,), activation='relu'))
    task.add(Dense(OUT_SIZE, activation='hard_sigmoid'))

    projector1 = Sequential()
    projector1.add(Dense(SYN_SIZE, input_shape=(LATENT_DIM,), activation='tanh'))
    projector1.add(Dense(NUM_FLAGS, activation='sigmoid'))

    projector2 = Sequential()
    projector2.add(Dense(SYN_SIZE, input_shape=(SYN_SIZE,), activation='tanh'))
    projector2.add(Dense(NUM_FLAGS, activation='sigmoid'))

    memory = Model(memory_input, gate_out(gate_in(memory_input)))
    memory.compile(optimizer=SGD(), loss="mean_squared_error")

    operator = Model(memory_input, task(gate_out(gate_in(memory_input))))
    operator.compile(optimizer=SGD(), loss="binary_crossentropy")

    projector1.compile(optimizer=Adadelta(), loss="mean_absolute_error")
    projector1.trainable = False

    projector2.compile(optimizer=Adadelta(), loss="mean_absolute_error")
    projector2.trainable = False

    pretender1 = Model(pretender1_input, projector1(gate_in(pretender1_input)))
    pretender1.compile(optimizer=RMSprop(), loss="binary_crossentropy")

    pretender2 = Model(pretender2_input, projector2(gate_out(pretender2_input)))
    pretender2.compile(optimizer=RMSprop(), loss="binary_crossentropy")

    return memory, projector1, projector2, pretender1, pretender2, gate_in, gate_out, operator

def load_synapses():
    for i in range(0, NUM_SYNAPSES):
        _mem, _proj1, _proj2, _pret1, _pret2, _enc, _dec, _op = create_synapse()

        try:
            _mem.load_weights("weights/"+str(i)+"_mem")
            _proj1.load_weights("weights/"+str(i)+"_proj1")
            _proj2.load_weights("weights/"+str(i)+"_proj2")
            _pret1.load_weights("weights/"+str(i)+"_pret1")
            _pret2.load_weights("weights/"+str(i)+"_pret2")
            _enc.load_weights("weights/"+str(i)+"_enc")
            _dec.load_weights("weights/"+str(i)+"_dec")
            _op.load_weights("weights/"+str(i)+"_op")

            print("loaded synapse " + str(i))
        except:
            print("synapse " + str(i) + " created")

        synapses.append([_mem, _proj1, _proj2, _pret1, _pret2, _enc, _dec, _op])

def save_synapses():
    for i in range(0, NUM_SYNAPSES):
        try:
            synapses[i][COMP_MEM].save_weights("weights/"+str(i)+"_mem")
            synapses[i][COMP_PROJ_1].save_weights("weights/"+str(i)+"_proj1")
            synapses[i][COMP_PROJ_2].save_weights("weights/"+str(i)+"_proj2")
            synapses[i][COMP_PRET_1].save_weights("weights/"+str(i)+"_pret1")
            synapses[i][COMP_PRET_2].save_weights("weights/"+str(i)+"_pret2")
            synapses[i][COMP_GATE_IN].save_weights("weights/"+str(i)+"_enc")
            synapses[i][COMP_GATE_OUT].save_weights("weights/"+str(i)+"_dec")
            synapses[i][COMP_OP].save_weights("weights/"+str(i)+"_op")

            print("saved synapse " + str(i))
        except:
            print("synapse " + str(i) + " not saved")

def load_item(row):
    train_x = []
    train_y = []

    for i in range(0,len(row)):
        v = row[i]

        if i == len(row) - 1:
            train_y.append(float(v))
        else:
            shash = softhash(str(v))

            for z in range(0, len(shash)):
                train_x.append(shash[z])

    for i in range(0, IN_SIZE-len(row) + 1):
        zhash = softhash("0")

        for z in range(0, len(zhash)):
            train_x.append(zhash[z])

    train_x = np.array([train_x])
    train_y = np.array([train_y])

    return (train_x, train_y)

#------------------------------------------------------------------------------------#

epochs = 0
load_synapses()
intelligence_signal = 0
failed_cycles = 0

while True:
    data_row_index = 0
    cycles = 0
    successes = 0
    success_rate = 0 
    lc_successes = 0
    lc_success_rate = 0 
    streak = 0
    beststreak = 0
    autonomous = True

    with open(DATA_FILE, "r") as csvfile: 
        datareader = csv.reader(csvfile)

        for row in datareader:
            if data_row_index>0 and np.random.rand()<DATA_USE_PERCENT:
                if cycles % CYCLES_PER_GLOBAL_EPOCH == 0:
                    epochs = epochs + 1

                train_x, train_y = load_item(row)
                truth = train_y[0][0]
                
                predictions = []
                latents = []

                for i in range(0, NUM_SYNAPSES):
                    l = synapses[i][COMP_GATE_IN].predict(train_x)
                    latents.append(l)

                biglatent = np.array(latents).reshape(SYN_SIZE)
                biglatent = np.array([biglatent])

                for i in range(0, NUM_SYNAPSES):
                    route = train_x if intelligence_signal==float(1) else biglatent
                    p = np.array(synapses[i][COMP_MEM].predict(route))
                    d = ((route - p)**2).mean()
                    predictions.append(d)

                # at training, reach the synapse with the highest decoder error
                # for predicting it's selected in differently
                target_synapse = np.argmax(predictions) if autonomous else cycles%NUM_SYNAPSES

                synapses[target_synapse][COMP_MEM].fit(x=biglatent, y=biglatent, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)
                
                enc_train_x = synapses[target_synapse][COMP_GATE_IN].predict(biglatent)
                dec_train_x = synapses[target_synapse][COMP_GATE_OUT].predict(enc_train_x)
                
                synapses[target_synapse][COMP_PRET_1].fit(x=biglatent, y=targets, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)
                synapses[target_synapse][COMP_PRET_2].fit(x=enc_train_x, y=targets, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)
                
                f_pred = synapses[target_synapse][COMP_OP].predict(train_x)
                f_lc_pred = synapses[target_synapse][COMP_OP].predict(biglatent)
                
                pred = np.around(f_pred[0][0])
                lc_pred = np.around(f_lc_pred[0][0])
                    
                wake = False

                if lc_pred==truth:
                    lc_successes = lc_successes + 1
                    streak = streak + 1

                    if streak>beststreak:
                        beststreak=streak

                    autonomous = True
                else:
                    streak = 0
                    if intelligence_signal == float(1):
                        wake = True
                    autonomous = False

                if pred==truth:
                    successes = successes + 1

                cycles = cycles + 1
                success_rate =  successes / cycles
                lc_success_rate =  lc_successes / cycles

                streak_odds = 2 ** streak
                beststreak_odds = 2 ** beststreak
                
                ssr = lc_success_rate if lc_success_rate >= 0.5 else lc_success_rate / 2
                intelligence_signal = beststreak_odds / cycles * ssr
                if intelligence_signal > 1:
                    failed_cycles = failed_cycles + 1
                    intelligence_signal = float(1)

                flags = np.array([intelligence_signal])
                flags = flags.reshape((1,NUM_FLAGS))

                latent = synapses[target_synapse][COMP_GATE_IN].predict(biglatent)
                dec = synapses[target_synapse][COMP_GATE_OUT].predict(latent)

                synapses[target_synapse][COMP_PROJ_1].fit(x=latent, y=flags, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)
                synapses[target_synapse][COMP_PROJ_2].fit(x=dec, y=flags, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)
                synapses[target_synapse][COMP_OP].fit(x=biglatent, y=train_y, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)

                print("data row index: ", data_row_index)
                print("synapse gate errors: ", predictions)
                print("epoch id: ", epochs)
                print("cycle id: ", cycles)
                print("failed cycles (by negotiator): ", failed_cycles)
                print("autonomous: ", autonomous)
                print("target synapse: ", target_synapse)
                print("truth: ", truth)
                print("prediction (data): ", f_pred)
                print("prediction (latent collection): ", f_lc_pred)
                print("successes (data): ", successes)
                print("successes (latent collections): ", lc_successes)
                print("overall proven success rate (data): ", success_rate)
                print("overall proven success rate (latent collections): ", lc_success_rate)
                print("gate error: ", predictions[target_synapse])
                print("streak: ", streak, " (odds = " + str(streak_odds) + ")")
                print("beststreak: ", beststreak, " (odds = " + str(beststreak_odds) + ")")
                print("intelligence signal: ", intelligence_signal)
                print("****************************************************************************")

                if wake or (cycles % CYCLES_PER_GLOBAL_EPOCH == 0 and cycles>0):
                    # reward the entire network with seeing the real data once, if it had reached the end of the cycle
                    # it's really a reward since it might help it to remember itself better after a failed cycle
                    # this might create a consciousness - comment the following 2 lines for safe AI
                    for x in range(0, NUM_SYNAPSES):
                        synapses[x][COMP_MEM].fit(x=train_x, y=train_x, epochs=EPOCHS_PER_FIT, batch_size=1, verbose=0)

                    if KAGGLE:
                        quit()

                    cycles = 0
                    successes = 0
                    success_rate = 0 
                    lc_successes = 0
                    lc_success_rate = 0 
                    streak = 0
                    beststreak = 0

                    # if we're here because wake==True then the next cycle will abnormally
                    # start in autonomous mode after a mistake has been made!
                    autonomous = True

                    if not wake:
                        save_synapses()

            data_row_index=data_row_index+1
