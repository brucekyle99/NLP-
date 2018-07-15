# /usr/bin/env python
# coding=utf-8
import os
import sys
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from params_config import ParamsConfig as config
from utils.data_convert import *
from utils.models import get_model

def pred(config, inpath, outputpath):
    test_df_c = create_charlist(inpath, output_file=None)
    test_df_w = create_wordlist(inpath, output_file=None, tongyi_file=config.tongyici)
    
    # WORD LEVEL TOKEN
    df_w = pd.read_csv(config.input_file_word, encoding="utf-8")    # process_word.csv， 原始数据分词
    question1_w = df_w['question1'].values
    question2_w = df_w['question2'].values
    test_question1_w = test_df_w['question1'].values
    test_question2_w = test_df_w['question2'].values
    tokenizer = Tokenizer(num_words=config.word_vocab_size)
    tokenizer.fit_on_texts(list(question1_w) + list(question2_w))
    
    list_tokenized_question1_w = tokenizer.texts_to_sequences(test_question1_w)
    list_tokenized_question2_w = tokenizer.texts_to_sequences(test_question2_w)
    X_test_w_q1 = pad_sequences(list_tokenized_question1_w, maxlen=config.word_seq_length)
    X_test_w_q2 = pad_sequences(list_tokenized_question2_w, maxlen=config.word_seq_length)
    
    # CHAR LEVEL TOKEN
    df_c = pd.read_csv(config.input_file_char, encoding="utf-8")
    question1_c = df_c['question1'].values
    question2_c = df_c['question2'].values
    test_question1_c = test_df_c['question1'].values
    test_question2_c = test_df_c['question2'].values
    tokenizer = Tokenizer(num_words=config.char_vocab_size)
    tokenizer.fit_on_texts(list(question1_c) + list(question2_c))
    
    list_tokenized_question1_c = tokenizer.texts_to_sequences(test_question1_c)
    list_tokenized_question2_c = tokenizer.texts_to_sequences(test_question2_c)
    X_test_c_q1 = pad_sequences(list_tokenized_question1_c, maxlen=config.char_seq_length)
    X_test_c_q2 = pad_sequences(list_tokenized_question2_c, maxlen=config.char_seq_length)
    
    pred_oob = np.zeros(shape=(len(X_test_w_q1), 1))
    
    model = get_model(config.kernel_name2)(config, None,None)
    count = 0   
    for index in range(config.cv_folds):
        if index in [0, 3, 7, 8]:
            continue
        bst_model_path = config.model_dir2 + config.kernel_name2 + '_weight_%d.h5' % index
        model.load_weights(bst_model_path)
        y_predict = model.predict([X_test_w_q1, X_test_w_q2, X_test_c_q1, X_test_c_q2], batch_size=128, verbose=1)
        pred_oob += y_predict
        print("Epoch: %d is over." %(index))
        count += 1
    
    model = get_model(config.kernel_name1)(config, None,None)    
    for index in range(config.cv_folds):
        if index in [3, 7, 8]:
            continue
        bst_model_path = config.model_dir1 + config.kernel_name1 + '_weight_%d.h5' % index
        model.load_weights(bst_model_path)
        y_predict = model.predict([X_test_w_q1, X_test_w_q2, X_test_c_q1, X_test_c_q2], batch_size=128, verbose=1)
        pred_oob += y_predict
        print("Epoch: %d is over." %(index))
        count += 1

    # pred_oob /= config.cv_folds
    pred_oob /= count
    pred_oob1 = (pred_oob > 0.5).astype(int)
    
    idx = np.reshape(np.arange(1, len(pred_oob1)+1), [-1, 1])
    preds = np.reshape(pred_oob1, [-1, 1])
    data = np.concatenate((idx, preds),axis=1)
    dataframe = pd.DataFrame(data, columns=['idx','pred'])
    dataframe.to_csv(outputpath, index=None, header=None, sep='\t')
    
if __name__ == "__main__":
    inpath, outputpath = sys.argv[1:3]
    pred(config, inpath, outputpath)
