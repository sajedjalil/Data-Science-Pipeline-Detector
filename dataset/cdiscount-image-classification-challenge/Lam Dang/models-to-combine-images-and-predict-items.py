from keras.layers import *
from keras.models import Model
import keras.backend as K

# Any results you write to the current directory are saved as output.
def flat_model():
    in_ = Input(shape=(4,2048), name='img')
    mix_in = Flatten()(in_)
    mix_in = Dense(2048, activation='elu')(mix_in)
    mix_in = Dropout(0.1)(mix_in)
    mix_in = Dense(2048, activation='elu')(mix_in)
    mix_in = Dropout(0.1)(mix_in)
    out_ = Dense(5270, activation='softmax', name='target')(mix_in)
    model = Model([in_], [out_]) 
    return model

def RNN():
    in_ = Input(shape=(4,2048), name='img')
    mix_in = CuDNNGRU(2048)(in_)
    out_ = Dense(5270, activation='softmax', name='target')(mix_in)
    model = Model([in_], [out_]) 
    return model

def RNN_2():
    in_ = Input(shape=(4,2048), name='img')
    mix_in = CuDNNGRU(1500, return_sequences=True)(in_)
    mix_in = CuDNNGRU(1500)(mix_in)
    out_ = Dense(5270, activation='softmax', name='target')(mix_in)
    model = Model([in_], [out_]) 
    return model

class NetVlad(Layer):
    def __init__(self, n_centers, output_dim, **kwargs):
        self.n_centers = n_centers
        self.output_dim = output_dim
        super(NetVlad, self).__init__(**kwargs)

    def build(self, input_shape):
        try:
            assert len(input_shape) == 3
        except:
            raise ValueError('Input should have shape (batch_size, time_steps, dims)')
            
        self.centers = self.add_weight(name='centers',
                                       shape=(self.n_centers, input_shape[-1]),
                                       initializer='uniform', 
                                       trainable=True
                                      )
        self.weight = self.add_weight(name='weight',
                                       shape=(input_shape[-1], self.n_centers),
                                       initializer='glorot_uniform', 
                                       trainable=True
                                      )
        self.bias = self.add_weight(name='bias',
                                    shape=(self.n_centers,),
                                    initializer='zeros', 
                                    trainable=True
                                   )
        self.reduc_dim = self.add_weight(name='reduc_dim',
                                         shape=(input_shape[-1]*self.n_centers, self.output_dim),
                                         initializer='zeros', 
                                         trainable=True
                                        )
        super(NetVlad, self).build(input_shape)

    def call(self, x):
        # X is shape (bsize, time_steps, input_size)
        # Get weights: shape=(bsize, time_steps, n_centers)
        centers_weight = K.softmax(K.dot(x, self.weight) + self.bias)
        assert centers_weight.shape[2] == self.n_centers
        assert centers_weight.shape[1] == x.shape[1]
        
        # Agg by cluster: shape=(bsize, n_centers, input_size)
        x_agg_clusters = K.batch_dot(centers_weight, x, axes=1)
        
        # Sum weights ad repeates in time dimension shape=(bsize, input_size, n_centers)
        sum_weights = K.sum(centers_weight, axis=1)        
        repeat_weights = K.repeat(sum_weights, x.shape[-1])
        # transpose to dim (bsize, n_centers, input_size)
        repeat_weights = K.permute_dimensions(repeat_weights,
                                              pattern=(0,2,1))
        # get full representation and flatten
        full_rep = x_agg_clusters - repeat_weights * self.centers
        full_rep = K.l2_normalize(full_rep, axis=-1)
        full_rep = K.reshape(full_rep, shape=(-1, self.n_centers * x.shape[2].value))
        full_rep = K.l2_normalize(full_rep, axis=-1)
        # Reduce dimension
        result = K.dot(full_rep, self.reduc_dim)
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0],self.output_dim)   

def netvlad_model():
    in_ = Input(shape=(4,2048), name='img')
    
    rep = NetVlad(n_centers=32, output_dim=2048)(in_)  
    rep = Dropout(0.2)(rep)
    
    # context gating
    gate = Dense(2048, activation='sigmoid')(rep)
    rep = Multiply()([gate, rep])
    
    out_ = Dense(5270, activation='softmax', name='target')(rep)
    model = Model([in_], [out_]) 
    return model
    
    
def create_model_text(models):
    models_list = {}
    for model_func, w_path in models:
        m = model_func()
        m.load_weights(w_path)
        for l in m.layers:
            l.trainable = False
        models_list[model_func.__name__] = m
            
    in_ = Input(shape=(4,2048), name='img') 
    
    preds = []
    for m in models_list.values():
        p = m(in_)
        preds.append(Reshape(target_shape=(1, 5270))(p))
    combined = Concatenate(axis=1)(preds)
    out_ = GlobalAveragePooling1D()(combined)
    
    text_in = Input(shape=(150,), name='text')
    emb = Embedding(input_dim=VOCAB_SIZE, output_dim=256)(text_in)
    agg = Lambda(lambda x: K.sum(x, axis=1))(emb)    
    agg = Lambda(lambda x: K.l2_normalize(x, axis=-1))(agg)
    
    # context gating
    gate = Dense(256, activation='sigmoid')(agg)
    agg= Multiply()([gate, agg])
    
    residual = Dense(5270, activation='linear')(agg)
    logit = Lambda(lambda x: K.log(x + K.epsilon()))(out_)
    final_logit = Add()([logit, residual])
    
    out_ = Lambda(lambda x: K.softmax(x))(final_logit)
    model = Model([in_, text_in], [out_]) 
    return model