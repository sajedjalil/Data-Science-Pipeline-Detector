import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Conv2D, Input, concatenate, Layer, BatchNormalization, Reshape, Lambda, ReLU
import tensorflow.keras.activations as act

from scipy.stats import ortho_group

import numpy as np
import pickle

predict = False   # compute predictions on test dataset
load = False      # load the weights from a previous run

num_epochs = 1
valid_size = 0  # rough number of samples kept aside for validation. A value of 0 turns off validaton
learning_rate = 0.0005  

save_interval = 3 # how often (epoch-wise) do we save the model's weights
num_ave = 10      # number of epochs to average the predictions over
rav = 0.2         # 1/half-life for the exponential moving average of cross validation predictions

bs =  48 # batch size, such that last batch is not too small (85003 % 92 = 87) 
test_bs = 193 # assuming 8091 validation samples
pred_bs = 194 # 45772 % 194 = 182 

use_bonds = True
use_pos = False
use_dist_pow = 4
bond_vec_only = True 
use_angles = True
use_charge = True
use_mag_contrib = True
   
#use_shield = 'trace'  
use_shield = 'eigenvalues' 
#use_shield = 'tensor'   # doesn't seem to work, even in fixed coo system
random_frame = True
random_pos = 0.0

weight_file = "squid-weights/weights.h5"
perm_file = "squid-weights/permutation.p"

contrib_weight = 0.2
charge_weight = 0.2
shield_weight = 0.2

# it would be better to get this from the dataset...
num_mag_contrib = 4
num_atom_per_mol = 29
num_pair_type = 8
num_atom_type = 5
num_bond_type = 5
num_bond_geom_feat = 3 + use_dist_pow + (1 if use_angles else 0) 
num_atom_geom_feat = 3 

# squid network geometry parameters

gcvs = 384          # vertex size for graph convs
gces = 48           # edge size for graph convs
ovs = 512           # on-vertex FC size
oes = 384           # on-edge FC size
tevs = 128          # vertex size before vertex -> edge conv 
tees = 128          # edge size after vertex -> edge conv
    

print("Input features:")
print(" - atomic number distinctions")
if use_bonds: print(" - basic chemical bond types")
if bond_vec_only: print(" - bond vectors")
else: print(" - vectors between any two atoms, divided by distance")
if use_dist_pow > 0: print(" - inverse distance between all atom pairs and its powers up to", use_dist_pow)
if use_angles: print(" - angles for 2JXX pairs and dihedral angles for 3JXX pairs")
if use_pos: print(" - atom positions")

print("Auxiliary targets:")
if use_charge: print(" - Mulliken charges", charge_weight)
if use_shield == 'tensor': print(" - Shielding tensors", shield_weight)
if use_shield == 'trace': print(" - Traces of the shielding tensors", shield_weight)
if use_shield == 'eigenvalues': print(" - Eigenvalues of the shielding tensors", shield_weight)
if use_mag_contrib: print(" - All magnetic coupling contributions", contrib_weight)
    
print("Data extension:")
if random_frame: print(" - Randomized reference frame")
if random_pos > 0.0: print(" - Noise on positions", random_pos)
    
root = "../input/"

from tensorflow.python.keras.utils.data_utils import Sequence

# does 0-hot for category 0
def one_hot(x, num_cat):
    if len(x.shape)==2:
        return np.eye(num_cat+1, dtype=np.bool)[x][:,:,1:] 
    elif len(x.shape)==1:
        return np.eye(num_cat+1, dtype=np.bool)[x][:,1:] 
    else:
        print("one_hot: unexpected shape:", x.shape)
        exit(1)

def make_geom(atom_pos, atom_type, bond_type, pair_type, pair_angles):     
    n = atom_pos.shape[0] #num_atom_per_mol
    k = num_atom_type
    
    geom = np.zeros((n,n, num_bond_geom_feat)) 

    # vector between every pairs
    delta = atom_pos.reshape(1,n,3) - atom_pos.reshape(n,1,3)
    
    # compute distances
    ir = np.reshape(np.sqrt(np.sum(delta**2, axis=2)), [n,n])
    nonzero = ir > 1e-6
    ir[nonzero] = 1/ir[nonzero]
    bond_mask = (bond_type > 0).reshape(n,n,1)
    bond_vec = delta * ir.reshape(n,n,1) * bond_mask
    
    # store unit bond vectors
    if bond_vec_only: geom[:,:,:3] = bond_vec # bond_vec * ir.reshape(n,n,1) #delta * bond_mask
    else: geom[:,:,:3] = delta * ir.reshape(n,n,1)**2   # or all vectors / distance
    ind = 3

    # store powers of the inverse distance between any two atoms 
    if use_dist_pow > 0: 
        geom[:,:,ind] = ir ; ind += 1
        for e in range(1,use_dist_pow):
            geom[:,:,ind] = geom[:,:,ind-1]*ir ; ind += 1

    # cos of angle between pairs bonded to the same atom
    if use_angles:
        geom[:,:,ind] = pair_angles 
        
    return geom

def process_mol(atom_pos, atom_type, bond_type, pair_type, pair_angles, in_ring):
    geom = make_geom(atom_pos, atom_type, bond_type, pair_type, pair_angles)
    atom = np.concatenate((one_hot(atom_type, num_atom_type), np.expand_dims(in_ring, -1)), axis=-1)
    bond = one_hot(bond_type, num_bond_type)
    # add adjacency
    nv = bond.shape[0]
    identity = np.expand_dims(np.eye(nv, dtype=np.bool), -1)
    adjacency = np.sum(bond, axis=-1, keepdims=True) + identity
    bond = np.concatenate((adjacency, bond), axis=-1)
    return geom, atom, bond

class Generator(Sequence):
    def __init__(self, data, batch_size=92, extra=[], shuffle=False, use_mag_contrib=False, use_charge=False, use_shield=False, predict=False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_mag_contrib = use_mag_contrib
        self.use_charge = use_charge
        self.use_shield = use_shield
        self.predict = predict

        self.data = data
        self.data_extra = extra

        self.random_frame = random_frame if shuffle else False
        self.random_pos = random_pos if shuffle else 0.0
        
        self.num_sample = len(data) 

        self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch'
        return int(np.ceil(self.num_sample / self.batch_size))

    def get_indexes(self, index):
        return self.indexes[index*self.batch_size:min((index+1)*self.batch_size, self.num_sample)]

    def __getitem__(self, index):
        'Generate one batch of data'
        return self.__data_generation(self.get_indexes(index))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.num_sample)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # needs to be called before rather than after the epoch since shuffling happens at epoch end...
    def get_pair_ids(self):
        num_mol = len(self.indexes)
        nam = num_atom_per_mol
        pair_ids = [] 
        pairs = np.zeros((num_mol, nam, nam), dtype=np.bool)

        for m in self.indexes:
            _, _, _, pair_type, _, _, pair_id = self.data[m]
            na = pair_type.shape[0]
            pairs[m,:na,:na] = np.triu(pair_type > 0)
            pair_ids.extend(pair_id[pairs[m,:na,:na]])

        return pair_ids, pairs
    
    def get_targets(self):
        num_mol = len(self.indexes)
        nam = num_atom_per_mol
        all_pair_type = np.zeros((num_mol, nam, nam), dtype=np.int32)
        mag_tot = np.zeros((num_mol, nam, nam), dtype=np.float32)

        for m in self.indexes:
            _, _, _, pair_type, _, _, raw_mag = self.data[m]
            na = pair_type.shape[0]
            all_pair_type[m,:na,:na] = pair_type
            i = raw_mag[:,1].astype(np.int)
            j = raw_mag[:,2].astype(np.int)
            mag_tot[m, i, j] = mag_tot[m, j, i] = raw_mag[:,3]

        return mag_tot, all_pair_type

    def __data_generation(self, indexes):
        
        data = [self.data[i] for i in indexes] 

        num_mol = len(data)
        nam = num_atom_per_mol
        atom = np.zeros((num_mol, nam, num_atom_type+1), dtype=np.bool)  # +1 for ring atoms
        bond = np.zeros((num_mol, nam, nam, num_bond_type+1), dtype=np.bool)  # +1 for adjacency
        geom = np.zeros((num_mol, nam, nam, num_bond_geom_feat), dtype=np.float32)
        pair = np.zeros((num_mol, nam, nam, num_pair_type), dtype=np.bool)

        if use_pos:
            atom_geom = np.zeros((num_mol, nam, num_atom_geom_feat), dtype=np.float32)

        extra = len(self.data_extra) > 0

        if extra:
            data_extra = [self.data_extra[i] for i in indexes]
            atom_charge = np.zeros((num_mol, nam, 1), dtype=np.float32)
            if self.use_shield == 'tensor':
                atom_shield = np.zeros((num_mol, nam, 3,3), dtype=np.float32)
            elif self.use_shield == 'trace':
                atom_shield = np.zeros((num_mol, nam, 1), dtype=np.float32)
            elif self.use_shield == 'eigenvalues':
                atom_shield = np.zeros((num_mol, nam, 3), dtype=np.float32)

        if not self.predict:
            mag = np.zeros((num_mol, nam, nam, num_mag_contrib), dtype=np.float32)
            mag_tot = np.zeros((num_mol, nam, nam), dtype=np.float32)
                        
        for m, mol in enumerate(data):
            atom_pos_abs, atom_type, bond_type, pair_type, pair_angles, in_ring, _ = mol
            
            # random rotation matrix, to remove dependence on the reference frame
            if random_frame: 
                R = ortho_group.rvs(dim=3).astype(np.float32)
                atom_pos = atom_pos_abs @ R 
            else:
                atom_pos = atom_pos_abs
            
            if self.random_pos > 0.0: atom_pos += np.random.normal(scale = self.random_pos, size=atom_pos.shape)

            na = pair_type.shape[0]
            pair[m,:na,:na,:] = one_hot(pair_type, num_pair_type)
    
            geom[m,:na,:na,:], atom[m,:na,:], bond[m,:na,:na,:] = process_mol(atom_pos, atom_type, bond_type, pair_type, pair_angles, in_ring)
                
            if use_pos:
                atom_geom[m,:na,:] = atom_pos

            if not self.predict:
                raw_mag = mol[-1]
                mag_tot[m, raw_mag[:,1].astype(np.int), raw_mag[:,2].astype(np.int)] = raw_mag[:,3]
                mag_tot[m, raw_mag[:,2].astype(np.int), raw_mag[:,1].astype(np.int)] = raw_mag[:,3]

                if self.use_mag_contrib:         
                    mag[m, raw_mag[:,1].astype(np.int), raw_mag[:,2].astype(np.int), :] = raw_mag[:,4:8]
                    mag[m, raw_mag[:,2].astype(np.int), raw_mag[:,1].astype(np.int), :] = raw_mag[:,4:8]
                                
                if self.use_charge and extra: 
                    atom_charge[m, :na, 0] = data_extra[m][:, 0] 
                    
                if self.use_shield and extra: 
                    mol_extra = data_extra[m]
                    T = np.reshape(mol_extra[:, 1:], (na, 3,3))
                    # assuming both index transform as inverse of coordinates. Is that correct?
                    # also we need to worry about the potential pseudo-ness of the tensor, given we use the whole ortho group
                    if random_frame and self.use_shield == 'tensor':
                        T = np.einsum('ij,aik->ajk', R, np.einsum('aij,jk->aik', T, R))

                    if self.use_shield == 'tensor':
                        atom_shield[m, :na, :,:] = T 
                    elif self.use_shield == 'trace':
                        atom_shield[m, :na, 0] = np.einsum('aii->a', T)
                    elif self.use_shield == 'eigenvalues':
                        atom_shield[m, :na, :] = np.linalg.eigvalsh(T) 
                    else:
                        print("unknown shielding tensor use type:", self.use-shield)
                        exit(1)

        if use_bonds: inputs = [atom, bond, geom, pair]
        else: inputs = [atom, bond, geom, pair]
        if use_pos: inputs.append(atom_geom)
                
        if not self.predict:
            targets = [mag_tot]
            if self.use_mag_contrib: targets.append(mag)
            if self.use_charge and extra: targets.append(atom_charge)
            if self.use_shield and extra: targets.append(atom_shield)
            return inputs, targets
        else: 
            return inputs

        
print("loading the data")
 
train_data = pickle.load(open(root+"pickled-molecules/molecules.p", "rb"))
pred_data = pickle.load(open(root+"pickled-molecules/molecules_test.p", "rb"))
extra_data = pickle.load(open(root+"pickled-molecules/molecules_extra.p", "rb"))

num_tot = len(train_data)
params = { 'use_mag_contrib': use_mag_contrib, 
           'use_charge': use_charge,
           'use_shield': use_shield }

if valid_size > 0:
    if load:
        print("loading choice of validation instances")
        perm, num_test = pickle.load( open( root+perm_file, "rb" ) )
    else:
        num_test = valid_size + ((num_tot - valid_size) % bs)
        perm = np.random.permutation(range(num_tot))
        
    pickle.dump((perm, num_test), open( "permutation.p", "wb" ))
        
    def sep(X): return [X[i] for i in perm[:num_test]], [X[i] for i in perm[num_test:]] 

    print(f"setting {num_test} samples aside for validation")

    valid_data, train_data = sep(train_data)
    valid_extra, extra_data = sep(extra_data)
    validation_generator = Generator(valid_data, shuffle=False, extra=valid_extra, batch_size=test_bs, **params) 

training_generator = Generator(train_data, shuffle=True, extra=extra_data, batch_size=bs, **params)
prediction_generator = Generator(pred_data, predict=True, shuffle=False, batch_size=pred_bs, **params)

print("building the network") 

class toEdges(Layer):

    def __init__(self, num_edge_filter, relu=False, **kwargs):
        self.ney = num_edge_filter
        self.relu = relu
        super(toEdges, self).__init__(**kwargs)

    def build(self, input_shape):
        shape_V = input_shape

        self.nv = int(shape_V[1])
        self.nvx = int(shape_V[2])

        self.W = self.add_weight(name='kernel', shape=(self.nvx, self.nvx, self.ney), initializer='uniform', trainable=True)
        self.b = self.add_weight(name='kernel', shape=(self.ney,), initializer='zeros', trainable=True)

        super(toEdges, self).build(input_shape)

    def call(self, V):

        aux = tf.einsum('bux,xyf->buyf', V, self.W)
        E = tf.einsum('buyf,bvy->buvf', aux, V) + self.b

        return act.relu(E) if self.relu else E

    def compute_output_shape(self, input_shape):
        return  (self.nv, self.ney, self.nv)


class GraphConv(Layer):

    def __init__(self, num_vertex_filter, relu=False, **kwargs):
        self.nvy = num_vertex_filter
        self.relu = relu
        super(GraphConv, self).__init__(**kwargs)

    def build(self, input_shape):
        shape_V, shape_E = input_shape
        self.nv = int(shape_V[1])
        self.nvx = int(shape_V[2])
        self.nex = int(shape_E[3])

        self.W = self.add_weight(name='kernel', shape=(self.nex, self.nvx, self.nvy), initializer='uniform', trainable=True)
        self.b = self.add_weight(name='kernel', shape=(self.nvy,), initializer='zeros', trainable=True)

        super(GraphConv, self).build(input_shape)

    def call(self, x):
        V, E = x
        
        aux = tf.einsum('buvx,bvy->buxy', E, V)
        V2 = tf.einsum('buxy,xyf->buf', aux, self.W) + self.b

        return act.relu(V2) if self.relu else V2

    def compute_output_shape(self, input_shape):
        return  (self.nv, self.nvy)


# some useful layers
relu = lambda x: ReLU()(x)
reban = lambda x: ReLU()(BatchNormalization()(x))
baner = lambda x: BatchNormalization()(ReLU()(x))

onEdge = lambda n: Conv2D(n, kernel_size=(1,1), strides=(1,1))
def onVertex(n, **kwargs): return Conv1D(n, kernel_size=(1,), strides=(1,), **kwargs)
def Symmetrize(perm=(0,2,1), **kwargs): return Lambda(lambda T: 0.5*(T + tf.transpose(T, perm=perm)), **kwargs)
def Trace(**kwargs): return Lambda(lambda T: tf.reshape(tf.trace(T), tf.concat([tf.shape(T)[:-2], [1]], axis=0)), **kwargs)

def SumPool(axis=-1, keepdims=False, **kwargs): 
    return Lambda(lambda arg: tf.reduce_sum(arg, axis=axis, keepdims=keepdims), **kwargs)

def MaskedSumPool( axis=-1, keepdims=False, **kwargs): 
    return Lambda(lambda args: tf.reduce_sum(args[0]*args[1], axis=axis, keepdims=keepdims), **kwargs)

def gcblock(m, v, e):
    return reban(GraphConv(m)([v, e]))
    
# the Squid network

inA = Input(shape=(num_atom_per_mol, num_atom_type+1))  
inB = Input(shape=(num_atom_per_mol, num_atom_per_mol, num_bond_type+1))  
inG = Input(shape=(num_atom_per_mol, num_atom_per_mol, num_bond_geom_feat)) 
inP = Input(shape=(num_atom_per_mol, num_atom_per_mol, num_pair_type))

inE = concatenate([inG, inB, inP]) 

if use_bonds:
    inE = concatenate([inG, inB, inP]) 
else:
    inE = concatenate([inG, inP]) 

if use_pos: 
    inX = Input(shape=(num_atom_per_mol, num_atom_geom_feat)) 
    inV = concatenate([inA, inX])
else:
    inV = inA

def init_edges(out_size):
    e = inE
    e = reban(onEdge(oes)(e))
    e = reban(onEdge(oes)(e))
    e = reban(onEdge(oes)(e))
    return reban(onEdge(out_size)(e))

def init_vertices(out_size):
    v = inV
    v = reban(onVertex(ovs)(v))
    v = reban(onVertex(ovs)(v))
    v = reban(onVertex(ovs)(v))
    return reban(onVertex(out_size)(v)) 

v = init_vertices(gcvs)
e = init_edges(gces)

v = gcblock(gcvs, v, e)
v = gcblock(gcvs, v, e)
v = gcblock(gcvs, v, e)


if use_charge:  
    v2 = relu(onVertex(ovs)(v))
    outC = onVertex(1,name="charge")(relu(onVertex(ovs)(v2)))  
    v = concatenate([outC, v])
    
if use_shield == 'tensor':   
    # this doesn't seem to work at all ... probably because they gave us the tensors in the wrong coo system
    v2 = relu(onVertex(ovs)(v))
    flatS = onVertex(9)(relu(onVertex(ovs)(v2))) 
    outS = Symmetrize(perm=(0,1,3,2), name="shield")(Reshape((num_atom_per_mol,3,3))(flatS))
    v = concatenate([Trace()(outS), v])
    
elif use_shield == 'trace':   
    v2 = relu(onVertex(ovs)(v))
    outS = onVertex(1, name="shield")(relu(onVertex(ovs)(v2))) 
    v = concatenate([outS, v])
    
elif use_shield == 'eigenvalues':   
    v2 = relu(onVertex(ovs)(v))
    outS = onVertex(3, name="shield")(relu(onVertex(ovs)(v2))) 
    v = concatenate([outS, v])

v = gcblock(gcvs, v, e)
v = gcblock(gcvs, v, e)

v = reban(GraphConv(tevs)([v, e]))
e = concatenate([reban(toEdges(tees)(v)), e])

e = reban(onEdge(oes)(e))
e = reban(onEdge(oes)(e))
e = reban(onEdge(oes)(e))
    
if use_mag_contrib: 
    es = [onEdge(num_pair_type)(e) for i in range(num_mag_contrib)]
    es = [MaskedSumPool(keepdims=True)([e, inP]) for e in es]
    outM = Symmetrize(perm=(0,2,1,3), name="contrib")(concatenate(es, axis=-1))
    outMtot = SumPool(name="mag")(outM)
else:
    e = onEdge(num_pair_type)(e)
    outMtot = Symmetrize(perm=(0,2,1), name="mag")(MaskedSumPool()([e, inP]))

outs = [outMtot]
if use_mag_contrib: outs.append(outM)
if use_charge: outs.append(outC)
if use_shield: outs.append(outS)

if use_pos:
    model = Model(inputs=[inA, inB, inG, inP, inX], outputs=outs)
else:
    model = Model(inputs=[inA, inB, inG, inP], outputs=outs)

def MagLoss(P):
    def loss(M_true, M_pred):
        err = 0.0
        for k in range(num_mag_contrib):
            for t in range(num_pair_type):
                err += tf.math.log(tf.reduce_mean(tf.abs(tf.boolean_mask(M_true[:,:,:,k] - M_pred[:,:,:,k], P[:,:,:,t]))))
        return err/(num_mag_contrib*num_pair_type)
    return loss

def TotMagLoss(P):
    def loss(M_true, M_pred):
        err = 0.0
        for t in range(num_pair_type):
            err += tf.math.log(tf.reduce_mean(tf.abs(tf.boolean_mask(M_true - M_pred, P[:,:,:,t]))))
        return err/num_pair_type
    return loss

def ChargeLoss(A):
    def loss(C_true, C_pred):
        return tf.math.log(tf.reduce_mean(tf.abs(tf.boolean_mask(C_true - C_pred, tf.reduce_sum(A, axis=2)))))
    return loss
   
def ShieldLoss(A):
    if use_shield == 'tensor':
        def loss(S_true, S_pred):
            mask = tf.reduce_sum(A, axis=2)
            return tf.math.log(tf.reduce_mean(tf.boolean_mask(tf.norm(S_true - S_pred, axis=[-2,-1], ord='fro'), mask)))
    elif use_shield == 'trace' or use_shield == 'eigenvalues':
        def loss(S_true, S_pred):
            mask = tf.reduce_sum(A, axis=2)
            return tf.math.log(tf.reduce_mean(tf.boolean_mask(tf.abs(S_true - S_pred), mask)))

    return loss    
    
losses = [TotMagLoss(inP)]
loss_weights = [1.0]
if use_mag_contrib: 
    losses.append(MagLoss(inP))
    loss_weights.append(contrib_weight)
if use_charge: 
    losses.append(ChargeLoss(inA))
    loss_weights.append(charge_weight)
if use_shield: 
    losses.append(ShieldLoss(inA))
    loss_weights.append(shield_weight)
    
opt = keras.optimizers.Adam(lr=learning_rate) 

model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights)

print("learning rate = ", keras.backend.get_value(model.optimizer.lr))


if load:
    print("loading weights from previous run", end="...")
    model.load_weights(root+weight_file)
    print("done")

if predict:
    print("producing prediction pair ids")
    ids, pred_pairs = prediction_generator.get_pair_ids()
    pred_ids = pd.Series(ids, name="id")

if valid_size > 0:
    print("producing validation targets")
    target_mag, val_pair = validation_generator.get_targets()
    
for epoch in range(num_epochs):

    print("epoch ", epoch+1, "/", num_epochs)
    
    model.fit_generator(generator = training_generator, epochs=1, verbose=2) 

    # cross validation. After 10 epochs, starts a running average to smooth out fluctuations due to ADAM
    if valid_size > 0:
        val_preds = model.predict_generator(generator = validation_generator, verbose=2) 
        if type(val_preds) == list: val_mag = val_preds[0]  
        else: val_mag = val_preds
            
        if epoch < 10: ave_val_mag = val_mag
        else: ave_val_mag = (1.0-rav)*ave_val_mag + rav*val_mag
            
        err_ave = 0.0
        for t in range(num_pair_type):
            mask = (val_pair==t+1)
            err_ave += np.log(np.mean(np.abs(target_mag[mask]-ave_val_mag[mask])))/num_pair_type
                
        print("validation accuracy: ", err_ave)
        
    # saves the model weights for continuing the learning in a different session
    if (epoch+1) % save_interval == 0 or epoch == num_epochs:
        filename = "weights.h5"
        model.save_weights(filename)
        print("saved model weights in "+filename)
        
    # prediction using the test dataset. Also makes an average, which is useful for large ADAM learning rate
    n = epoch - (num_epochs-1-num_ave)
    if predict and n >= 1:
        print("computing predictions")
        preds = model.predict_generator(generator = prediction_generator, verbose=2) 
        if type(preds) == list: pred_mag = preds[0]  
        else: pred_mag = preds
            
        if n==1: pred_ave = pred_mag[pred_pairs]
        else: pred_ave = ((n-1)*pred_ave + pred_mag[pred_pairs])/n

        preds = pd.Series(pred_ave, name="scalar_coupling_constant")
        pd.concat([pred_ids, preds], axis = 1).sort_values('id', inplace=False).to_csv("predictions.csv", index=False)
        

                