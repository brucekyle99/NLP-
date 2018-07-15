# /usr/bin/env python
# coding=utf-8

import pandas as pd
import re
import os
import jieba

from params_config import ParamsConfig as config
from utils.data_convert import *
from utils.vector_utils import word2vec

def train_data_prepare(config):
    if os.path.exists(config.data_dir) == False:
        os.mkdir(config.data_dir)
    
    df_w = create_wordlist(config.input_file, config.input_file_word, config.tongyici)
    df_c = create_charlist(config.input_file, config.input_file_char)
    
    df_w_comb = create_inter(config.input_file_word, config.combine_w_file)
    df_c_comb = create_inter(config.input_file_char, config.combine_w_file)
    
    df_w_diff = create_diff(config.input_file_word, config.diff_w_file)
    df_w_diff = create_diff(config.input_file_char, config.diff_w_file)
    
    all_question_w = np.concatenate((df_w['question1'], df_w['question2']))
    create_vocab(all_question_w, "w_all", config.data_dir)
    all_question_c = np.concatenate((df_c['question1'], df_c['question2']))
    create_vocab(all_question_c, "c_all", config.data_dir)
    
    word2vec(config.input_file_word, config.w2v_w_file, config.embedding_dims, config.window_size, config.min_count)
    word2vec(config.input_file_char, config.w2v_c_file, config.embedding_dims, config.window_size, config.min_count)

if __name__ == "__main__":
    train_data_prepare(config)
