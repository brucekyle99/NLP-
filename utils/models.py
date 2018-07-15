# /usr/bin/env python
# coding=utf-8
import keras
from keras.layers import *
from keras.layers.merge import *
from keras.layers.recurrent import *
from keras.models import *
from keras.optimizers import *
from keras.engine import Layer
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # W: (EMBED_SIZE, ATTENTION_SIZE)
        # b: (ATTENTION_SIZE, 1)
        # u: (ATTENTION_SIZE, 1)
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1], self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # input: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # et: (BATCH_SIZE, MAX_TIMESTEPS, ATTENTION_SIZE)
        et = K.tanh(K.dot(x, self.W) + self.b)
        # at: (BATCH_SIZE, MAX_TIMESTEPS)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))  # 将下标为axis的一维从张量中移除
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        # ot: (BATCH_SIZE, MAX_TIMESTEPS, EMBED_SIZE)
        # atx: (BATCH_SIZE, MAX_TIMESTEPS, ? EMBED_SIZE)
        atx = K.expand_dims(at, axis=-1)   # 在下标为dim的轴上增加一维
        ot = atx * x
        # output: (BATCH_SIZE, EMBED_SIZE)
        output = K.sum(ot, axis=1)  # max_timesteps 上面求和
        return output

    def compute_mask(self, input, input_mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
        
def dssm_model(config, word_emb=None, char_emb=None):
    input1 = Input(shape=(config.word_seq_length,))
    input2 = Input(shape=(config.word_seq_length,))
    
    if word_emb is not None and char_emb is not None:
        embed1 = Embedding(input_dim=config.word_vocab_size+1,
                           output_dim=config.embedding_dims, 
                           weights=[word_emb], 
                           trainable=config.emb_trainable)
    else:
        embed1 = Embedding(input_dim=config.word_vocab_size+1,
                           output_dim=config.embedding_dims)
                           
    lstm0 = LSTM(config.w_lstmsize,return_sequences = True)
    lstm1 = Bidirectional(LSTM(config.w_lstmsize))
    lstm2 = LSTM(config.w_lstmsize)
    att1 = Attention(config.att_units)
    
    v1 = embed1(input1)
    v2 = embed1(input2)
    v11 = lstm1(v1)  
    v22 = lstm1(v2)
    v1ls = lstm2(lstm0(v1))
    v2ls = lstm2(lstm0(v2))
    v1 = Concatenate(axis=1)([att1(v1),v11])
    v2 = Concatenate(axis=1)([att1(v2),v22])

    input1c = Input(shape=(config.char_seq_length,))
    input2c = Input(shape=(config.char_seq_length,))
    
    if word_emb is not None and char_emb is not None:
        embed1c = Embedding(input_dim=config.char_vocab_size+1,
                            output_dim=config.embedding_dims, 
                            weights=[char_emb], 
                            trainable=config.emb_trainable)
    else:
        embed1c = Embedding(input_dim=config.char_vocab_size+1,
                            output_dim=config.embedding_dims)

    lstm1c = Bidirectional(LSTM(config.c_lstmsize))
    att1c = Attention(config.att_units)

    v1c = embed1c(input1c)
    v2c = embed1c(input2c)
    v11c = lstm1c(v1c)
    v22c = lstm1c(v2c)
    v1c = Concatenate(axis=1)([att1c(v1c),v11c])
    v2c = Concatenate(axis=1)([att1c(v2c),v22c])

    mul = Multiply()([v1,v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1,v2]))
    maximum = Maximum()([Multiply()([v1,v1]),Multiply()([v2,v2])])
    cos = dot([v1, v2], axes=1, normalize=True)
    
    mulc = Multiply()([v1c,v2c])
    subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c,v2c]))
    maximumc = Maximum()([Multiply()([v1c,v1c]),Multiply()([v2c,v2c])])
    cosc = dot([v2c,v2c], axes=1, normalize=True)
    
    sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls,v2ls]))
    cos2 = dot([v1, v2], axes=1, normalize=True)
    
    matchlist = Concatenate(axis=1)([mul,sub,mulc,subc,maximum,maximumc,sub2, cos, cos2, cosc])
    matchlist = Dropout(config.dropout)(matchlist)

    matchlist = Concatenate(axis=1)([Dense(32,activation = 'relu')(matchlist),Dense(48,activation = 'sigmoid')(matchlist)])
    res = Dense(1, activation = 'sigmoid')(matchlist)

    model = Model(inputs=[input1, input2, input1c, input2c], outputs=res)

    return model

def dssm_cudnn_model(config, word_emb=None, char_emb=None):
    input1 = Input(shape=(config.word_seq_length,))
    input2 = Input(shape=(config.word_seq_length,))
    
    if word_emb is not None and char_emb is not None:
        embed1 = Embedding(input_dim=config.word_vocab_size+1,
                           output_dim=config.embedding_dims, 
                           weights=[word_emb], 
                           trainable=config.emb_trainable)
    else:
        embed1 = Embedding(input_dim=config.word_vocab_size+1,
                           output_dim=config.embedding_dims)
                           
    lstm0 = CuDNNLSTM(config.w_lstmsize,return_sequences = True)
    lstm1 = Bidirectional(CuDNNLSTM(config.w_lstmsize))
    lstm2 = CuDNNLSTM(config.w_lstmsize)
    att1 = Attention(config.att_units)
    
    v1 = embed1(input1)
    v2 = embed1(input2)
    v11 = lstm1(v1)  
    v22 = lstm1(v2)
    v1ls = lstm2(lstm0(v1))
    v2ls = lstm2(lstm0(v2))
    v1 = Concatenate(axis=1)([att1(v1),v11])
    v2 = Concatenate(axis=1)([att1(v2),v22])

    input1c = Input(shape=(config.char_seq_length,))
    input2c = Input(shape=(config.char_seq_length,))
    
    if word_emb is not None and char_emb is not None:
        embed1c = Embedding(input_dim=config.char_vocab_size+1,
                            output_dim=config.embedding_dims, 
                            weights=[char_emb], 
                            trainable=config.emb_trainable)
    else:
        embed1c = Embedding(input_dim=config.char_vocab_size+1,
                            output_dim=config.embedding_dims)
    lstm1c = Bidirectional(CuDNNLSTM(config.c_lstmsize))
    att1c = Attention(config.att_units)
    v1c = embed1c(input1c)
    v2c = embed1c(input2c)
    v11c = lstm1c(v1c)
    v22c = lstm1c(v2c)
    v1c = Concatenate(axis=1)([att1c(v1c),v11c])
    v2c = Concatenate(axis=1)([att1c(v2c),v22c])

    mul = Multiply()([v1,v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1,v2]))
    maximum = Maximum()([Multiply()([v1,v1]),Multiply()([v2,v2])])
    cos = dot([v1, v2], axes=1, normalize=True)
    
    mulc = Multiply()([v1c,v2c])
    subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c,v2c]))
    maximumc = Maximum()([Multiply()([v1c,v1c]),Multiply()([v2c,v2c])])
    cosc = dot([v2c,v2c], axes=1, normalize=True)
    
    sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls,v2ls]))
    cos2 = dot([v1, v2], axes=1, normalize=True)
    
    matchlist = Concatenate(axis=1)([mul,sub,mulc,subc,maximum,maximumc,sub2, cos, cos2, cosc])
    matchlist = Dropout(config.dropout)(matchlist)

    matchlist = Concatenate(axis=1)([Dense(32,activation = 'relu')(matchlist),Dense(48,activation = 'sigmoid')(matchlist)])
    res = Dense(1, activation = 'sigmoid')(matchlist)

    model = Model(inputs=[input1, input2, input1c, input2c], outputs=res)

    return model
    
def dssm_model_300(config, word_emb=None, char_emb=None, char_emb_300=None):
    input1 = Input(shape=(config.word_seq_length,))
    input2 = Input(shape=(config.word_seq_length,))
    
    if word_emb is not None and char_emb is not None and char_emb_300 is not None:
        embed1w = Embedding(input_dim=config.word_vocab_size+1,
                           output_dim=config.embedding_dims, 
                           weights=[word_emb], 
                           trainable=config.emb_trainable)
    else:
        embed1w = Embedding(input_dim=config.word_vocab_size+1,
                           output_dim=config.embedding_dims)
        
    lstm0w = LSTM(config.w_lstmsize,return_sequences = True)
    lstm1w = Bidirectional(LSTM(config.w_lstmsize))
    lstm2w = LSTM(config.w_lstmsize)
    att1w = Attention(config.att_units)
    
    v1w = embed1w(input1)
    v2w = embed1w(input2)
    v11w = lstm1w(v1w)  
    v22w = lstm1w(v2w)
    v1lsw = lstm2w(lstm0w(v1w))
    v2lsw = lstm2w(lstm0w(v2w))
    v1w = Concatenate(axis=1)([att1w(v1w),v11w])
    v2w = Concatenate(axis=1)([att1w(v2w),v22w])
    
    mulw = Multiply()([v1w,v2w])
    subw = Lambda(lambda x: K.abs(x))(Subtract()([v1w,v2w]))
    maximumw = Maximum()([Multiply()([v1w,v1w]),Multiply()([v2w,v2w])])
    cosw = dot([v1w, v2w], axes=1, normalize=True)
    
    input1c = Input(shape=(config.char_seq_length,))
    input2c = Input(shape=(config.char_seq_length,))
    
    if word_emb is not None and char_emb is not None and char_emb_300 is not None:
        embed1 = Embedding(input_dim=config.char_vocab_size+1,
                           output_dim=config.embedding_dims, 
                           weights=[char_emb], 
                           trainable=config.emb_trainable)
    else:
        embed1 = Embedding(input_dim=config.char_vocab_size+1,
                           output_dim=config.embedding_dims)
                           
    lstm0 = LSTM(config.w_lstmsize,return_sequences = True)
    lstm1 = Bidirectional(LSTM(config.w_lstmsize))
    lstm2 = LSTM(config.w_lstmsize)
    att1 = Attention(config.att_units)
    
    v1 = embed1(input1c)
    v2 = embed1(input2c)
    v11 = lstm1(v1)  
    v22 = lstm1(v2)
    v1ls = lstm2(lstm0(v1))
    v2ls = lstm2(lstm0(v2))
    v1 = Concatenate(axis=1)([att1(v1),v11])
    v2 = Concatenate(axis=1)([att1(v2),v22])
    
    if word_emb is not None and char_emb is not None and char_emb_300 is not None:
        embed1c = Embedding(input_dim=config.char_vocab_size+1,
                            output_dim=300,
                            weights=[char_emb_300], 
                            trainable=config.emb_trainable)
    else:
        embed1c = Embedding(input_dim=config.char_vocab_size+1,
                            output_dim=300)
    lstm1c = Bidirectional(LSTM(25))
    att1c = Attention(config.att_units)
    v1c = embed1c(input1c)
    v2c = embed1c(input2c)
    v11c = lstm1c(v1c)
    v22c = lstm1c(v2c)
    v1c = Concatenate(axis=1)([att1c(v1c),v11c])
    v2c = Concatenate(axis=1)([att1c(v2c),v22c])

    mul = Multiply()([v1,v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1,v2]))
    maximum = Maximum()([Multiply()([v1,v1]),Multiply()([v2,v2])])
    cos = dot([v1, v2], axes=1, normalize=True)
    
    mulc = Multiply()([v1c,v2c])
    subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c,v2c]))
    maximumc = Maximum()([Multiply()([v1c,v1c]),Multiply()([v2c,v2c])])
    cosc = dot([v2c,v2c], axes=1, normalize=True)
    
    sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls,v2ls]))
    cos2 = dot([v1, v2], axes=1, normalize=True)
    
    matchlist = Concatenate(axis=1)([mul,sub,mulc,subc,maximum,maximumc,sub2, cos, cos2, mulw, subw, maximumw])
    matchlist = Dropout(config.dropout)(matchlist)

    matchlist = Concatenate(axis=1)([Dense(32,activation = 'relu')(matchlist),Dense(48,activation = 'sigmoid')(matchlist)])
    # matchlist = Dense(128,activation = 'relu')(matchlist)
    res = Dense(1, activation = 'sigmoid')(matchlist)

    model = Model(inputs=[input1, input2, input1c, input2c], outputs=res)

    return model
    
def abcnn_model(config, word_emb=None, char_emb=None):
    pass

model_dict = {
    "dssm": dssm_model,
    "dssm_cudnn": dssm_cudnn_model,
    "abcnn": abcnn_model,
    "dssm_300": dssm_model_300,
}

def get_model(kernel_name):
    return model_dict[kernel_name]
