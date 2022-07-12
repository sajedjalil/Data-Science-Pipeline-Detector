# randomized feature selection by genetic algorithm, parallelized
# emulate sklearn.feature_selection programming style
# Author: Rand Xie

from multiprocessing.dummy import Pool as ThreadPool
import numpy as np
import pickle
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve, auc
class GeneticAlgSelect():
    # initial model type, model pool and load in data_in,
    def __init__(self, data_in, data_out, mdl_type, mdl_para, **para):
        # load input and output data, para
        self.data_in = data_in
        self.data_out = data_out
        self.para = para

        # define model type
        self.mdl_type = mdl_type
        self.mdl_para = mdl_para

        # init model pool and elite list
        self.mdl_pool = []
        self.elite_list = []

        # load in para
        self.__set_constant()

    def __set_constant(self):
        # define default constants
        self.num_worker = self.para['num_worker'] if 'num_worker' in self.para else 4
        self.max_iter = self.para['max_iter'] if 'max_iter' in self.para else 3
        self.pool_size = self.para['pool_size'] if 'pool_size' in self.para else 10
        self.mutateLR = self.para['mutateLR'] if 'mutateLR' in self.para else 0.1
        self.mutateUR = self.para['mutateUR'] if 'mutateUR' in self.para else 0.2
        self.mutateBIT = self.para['mutateBIT'] if 'mutateBIT' in self.para else 0.1
        self.crossLR = self.para['crossLR'] if 'crossLR' in self.para else 0.1
        self.crossUR = self.para['crossUR'] if 'crossUR' in self.para else 0.2
        self.eliteR = self.para['elite_rate'] if 'elite_rate' in self.para else 0.02
        self.savefileName = self.para['savefile'] if 'savefile' in self.para else 'GeneticAlgResult.p'

    # perform interation and evaluate local model in parallel
    def _perform_iter(self):
        self.__gen_pool()
        for i in range(self.max_iter):
            #print "performing iteration: %d /n" %(i)
            # process data_in in parallel
            p = ThreadPool(self.num_worker)
            l=range(0,self.pool_size)
            # train the local models
            result=p.map(self._train_mdl,l)
            p.close()
            p.join()
            # eliticism, cross_over and mutation
            self.__eliticism()
            self.__cross_over()
            self.__mutation()
            self._save_mdl()

    # generate model pool
    def __gen_pool(self):
        self.gene_len = self.data_in.shape[0] if len(self.data_in.shape)==1 else self.data_in.shape[1]
        for i in range(self.pool_size):
            gene = np.random.random_integers(0,high=1,size=self.gene_len)
            while gene[np.nonzero(gene)].size==0:
                gene = np.random.random_integers(0,high=1,size=self.gene_len)
            self.mdl_pool.append(GeneticAlgLocalModel(gene, self.mdl_type, self.mdl_para))

    # train local model
    def _train_mdl(self,i):
        #print 'training model: %d /n' %(i)
        data_in = self.data_in[:,self.mdl_pool[i].gene==1]
        self.mdl_pool[i]._train_mdl(data_in,self.data_out)

    # get model with highest score (best performance)
    def __eliticism(self):
        score_arr=np.zeros(self.pool_size)
        for i in range(self.pool_size):
            score_arr[i] = self.mdl_pool[i].score
        num_elite = round(self.pool_size*self.eliteR)
        self.elite_list = np.argsort(score_arr)[-num_elite:]

    # exchange gene pieces
    def __cross_over(self):
        min_c, max_c = round(self.pool_size*self.crossLR), round(self.pool_size*self.crossUR+1)
        num_cross = np.random.randint(min_c, high=max_c)
        for i in range(num_cross):
            # randomly pick model A and model B
            idxA = np.random.randint(0, high=self.pool_size)
            idxB = np.random.randint(0, high=self.pool_size)
            while idxA in self.elite_list:
                idxA = np.random.randint(0, high=self.pool_size)
            while (idxB in self.elite_list) or (idxB == idxA):
                idxB = np.random.randint(0, high=self.pool_size)

            # generate cross over start pt and end pt
            pt_s = np.random.randint(0, high=self.gene_len)
            pt_e = np.random.randint(0, high=self.gene_len)
            pt_s, pt_e = (pt_e, pt_s) if pt_s>pt_e else (pt_s, pt_e)

            # exchange gene
            self.mdl_pool[idxA].gene[pt_s:pt_e], self.mdl_pool[idxB].gene[pt_s:pt_e] = (self.mdl_pool[idxB].gene[pt_s:pt_e],self.mdl_pool[idxA].gene[pt_s:pt_e])
            self.mdl_pool[idxA].changed = True
            self.mdl_pool[idxB].changed = True

    # mutate some bits
    def __mutation(self):
        min_m, max_m = round(self.pool_size*self.mutateLR), round(self.pool_size*self.mutateUR+1)
        num_mute = np.random.randint(min_m, high=max_m)
        bit_mute = round(self.mutateBIT*self.gene_len)

        for i in range(num_mute):
            idx = np.random.randint(0, high=self.pool_size)
            while idx in self.elite_list:
                idx = np.random.randint(0, high=self.pool_size)
            pts = np.random.randint(0, high=self.gene_len, size=bit_mute)
            self.mdl_pool[idx].gene[pts] = (1-self.mdl_pool[idx].gene[pts])
            self.mdl_pool[idx].changed = True

    def print_best_mdl(self):
        score_arr=np.zeros(len(self.elite_list))
        for i in range(len(score_arr)):
            score_arr[i] = self.mdl_pool[self.elite_list[i]].score
        max_idx = score_arr.argmax()
        #print 'The best model calculated is as following'
        #print self.mdl_pool[self.elite_list[i]].gene
        #print self.mdl_pool[self.elite_list[i]].score

    def _save_mdl(self):
        pickle.dump(self.mdl_pool, open(self.savefileName,"wb"))

# local model structure for genetic algorithm that stores local model and gene structure
class GeneticAlgLocalModel():
    def __init__(self, gene, mdl_type, mdl_para):
        self.gene=gene
        self.mdl=mdl_type(**mdl_para)
        self.changed = True
        self.score=0

    def _train_mdl(self, data_in, data_out):
        if self.changed:
            kf = KFold(data_in.shape[0], n_folds = 5, shuffle=False)
            i = 0
            for train_idx,test_idx in kf:
                if (i<1):
                    self.mdl.fit(data_in[train_idx,:], data_out[train_idx])
                    self.changed = False
                    fpr, tpr, _ = roc_curve(data_out[test_idx], self.mdl.predict(data_in[test_idx]))
                    self.score = auc(fpr, tpr) #self.mdl.score
                    #print self.score
                    i = i+1



