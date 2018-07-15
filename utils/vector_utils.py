# /usr/bin/env python
# coding=utf-8
import logging
import time
import multiprocessing
from gensim.models import Word2Vec

def word2vec(data_file, w2v_file, embedding_dim, window=5, min_count=1):
    """Train and save word vectors"""
    def load_data(data_file):
        for line in open(data_file):
            line = line.strip().decode('utf-8').split(',')
            s1, s2 = line[1].split(' '), line[2].split(' ')
            yield s1, s2
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    s_time = time.time()
    s1, s2 = zip(*list(load_data(data_file)))
    sentences = s1 + s2
    size = embedding_dim

    model = Word2Vec(sentences, sg=1, size=size, window=window, min_count=min_count,
                     negative=3, sample=0.001, hs=1, workers=multiprocessing.cpu_count(), iter=20)
                     
    model.wv.save_word2vec_format(w2v_file, binary=False)
    print("Word2vec training time: %d s" % (time.time() - s_time))