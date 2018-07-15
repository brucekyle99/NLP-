# /usr/bin/env python
# coding=utf-8

MAX_TEXT_LENGTH_WORD_LEVEL = 50
MAX_TEXT_LENGTH_CHAR_LEVEL = 50

MAX_DIFF_LENGTH_WORD_LEVEL = 45
MAX_DIFF_LENGTH_CHAR_LEVEL = 45

MAX_SAME_LENGTH_WORD_LEVEL = 10
MAX_SAME_LENGTH_CHAR_LEVEL = 10

MAX_FEATURES_WORD_LEVEL = 9000
MAX_FEATURES_CHAR_LEVEL = 1700

class ParamsConfig(object):
    """parameter settings"""
    # model
    kernel_name1 = "dssm_300"
    kernel_name2 = "dssm"
    
    # data dirs
    input_file='./input/atec_nlp_sim_train.csv'
    input_file_word = "./input/process_word.csv"
    input_file_char = "./input/process_char.csv"

    diff_w_file = "./input/process_diff_w.csv"
    diff_c_file = "./input/process_diff_c.csv"

    combine_w_file = "./input/process_combine_w.csv"
    combine_c_file = "./input/process_combine_c.csv"
    
    diff_w_file = "./input/process_diff_w.csv"
    diff_c_file = "./input/process_diff_c.csv"

    w2v_w_file = "./data/word_vec"
    w2v_c_file = "./data/char_vec"

    w2v_c_file_300 = "./data/char_vec_300.npy"

    tongyici = "./input/tongyici.txt"
    
    data_dir = "./data/"
    model_dir1 = "./model/%s/" % kernel_name1
    model_dir2 = "./model/%s/" % kernel_name2
    
    # model parameters
    word_vocab_size = MAX_FEATURES_WORD_LEVEL
    char_vocab_size = MAX_FEATURES_CHAR_LEVEL
    word_seq_length = MAX_TEXT_LENGTH_WORD_LEVEL
    char_seq_length = MAX_TEXT_LENGTH_CHAR_LEVEL
    embedding_dims = 128
    
    w_lstmsize = 20
    c_lstmsize = 6
    att_units = 10
    
    # word2vec params
    window_size = 5
    min_count = 1
    
    # training parameters
    learning_rate = 0.001
    dropout = 0.05
    
    batch_size = 512
    num_epochs = 10
    
    emb_trainable = True
    is_load_embedding = True
    
    class_weight = {1: 1.2233, 0: 0.4472}
    
    seed = 2345
    cv_folds = 10
