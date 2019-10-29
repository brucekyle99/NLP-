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

from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from sklearn.model_selection import train_test_split
import codecs
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback
from tqdm import tqdm
from utils.score import *

from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
import pandas as pd

maxlen = 100
config_path = '../chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../chinese_L-12_H-768_A-12/vocab.txt'

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

token_dict = {}
with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

tokenizer = OurTokenizer(token_dict)

# neg = pd.read_csv(r'D:\1python notes\ATEC\final\atec_nlp_2\input\atec_nlp_sim_train.csv', header=None, encoding="utf-8", sep="\t")
# neg = pd.read_csv(r'../input/atec_nlp_sim_train.csv', header=None, encoding="utf-8", sep=",")

data = []

with open("../input/atec_nlp_sim_train.csv", encoding="utf-8") as fp:
    ret = []
    for line in fp:
        q = {}
        lines = line.strip().split("\t")
        question = "__{}__{}".format(lines[1], lines[2])
        if (len(lines) == 4):
            label = lines[3].strip()
            data.append((question, label))

        # q['question1'] = seg(lines[1].decode('utf-8'), tongYiCiDict)
        # q['question2'] = seg(lines[2].decode('utf-8'), tongYiCiDict)
        # if (len(lines) == 4):
        #     q['label'] = lines[3].strip()
        # ret.append(q)

# 按照9:1的比例划分训练集和验证集
random_order = list(range(len(data)))
np.random.shuffle(random_order)
train_data = [data[j] for i, j in enumerate(random_order) if i % 10 != 0]
valid_data = [data[j] for i, j in enumerate(random_order) if i % 10 == 0]

def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])

class data_generator:
    # def __init__(self, data, batch_size=32):
    def __init__(self, data, batch_size=16):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []

def bertt_model():
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)

    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)
    p = Dense(1, activation='sigmoid')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(1e-5),  # 用足够小的学习率
        metrics=["accuracy", f1_score_metrics]
    )
    # model.summary()

    train_D = data_generator(train_data)
    valid_D = data_generator(valid_data)

    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=5,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D)
    )
    model.save_weights('best_model.weights')

if __name__ == "__main__":
    bertt_model()


















model_dict = {
    "bertt_model": bertt_model,
}

def get_model(kernel_name):
    return model_dict[kernel_name]
