#Modification of Peter's GRU ATT kernel (All thanks to him)
#This Attlayer works with keras backend as Tensorflow
from keras.engine.topology import Layer, InputSpec

class AttLayer(Layer):
    def __init__(self, use_bias=True, activation ='tanh', **kwargs):
        self.init = initializers.get('normal')
        self.use_bias = use_bias
        self.activation = activation
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.add_weight(name='kernel', 
                                  shape=(input_shape[-1],1),
                                  initializer='normal',
                                  trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', 
                                  shape=(1,),
                                  initializer='zeros',
                                  trainable=True)
        else:
            self.bias = None
        super(AttLayer, self).build(input_shape) 

    def call(self, x, mask=None):
        eij = K.dot(x, self.W)
        if self.use_bias:
            eij =K.bias_add(eij, self.bias)
        if self.activation == 'tanh':
            eij = K.tanh(eij)
        elif self.activation =='relu':
            eij = K.relu(eij)
        else:
            eij = eij
        ai = K.exp(eij)
        weights = ai/K.sum(ai, axis=1, keepdims=True)
        weighted_input = x*weights
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = {
            'activation':self.activaion
        }
        base_config = super(AttLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

def encoder_model():
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32', name='main_input')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
    l_dense = TimeDistributed(Dense(200))(l_lstm)
    l_att = AttLayer(use_bias=False, activation='tanh')(l_dense)
    sentEncoder = Model(sentence_input, l_att)
    return sentEncoder

def decoder_model():
    review_input = Input(shape=(MAX_SENTS,MAX_SENT_LENGTH), dtype='int32')
    sentEncoder = encoder_model()
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
    l_att_sent = AttLayer(use_bias=False, activation='tanh')(l_dense_sent)
    preds = Dense(2, activation='softmax')(l_att_sent)
    model = Model(review_input, preds)
    return model

