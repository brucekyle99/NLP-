# /usr/bin/env python
# coding=utf-8
import re
import os
import sys
import codecs
import pandas as pd
import numpy as np
from collections import Counter
import jieba

from langconv import Converter

jieba.add_word('花呗', freq=10000000)
jieba.add_word('借呗')
jieba.add_word('余额宝')
jieba.add_word('支付宝')
jieba.add_word('蚂蚁借呗')
jieba.add_word('蚂蚁花呗')
jieba.add_word('双十一')
jieba.add_word('双十二')

jieba.add_word('花坝')
jieba.add_word('花贝')
jieba.add_word('花臂')

stopwords = ['的', '了']
stopwords = [i.decode('utf-8') for i in stopwords]

def tongYiCi(path):
    """Loading synonyms."""
    combine_dict = {}
    for line in open(path, "r"):
        seperate_word = line.strip().split(",")
        num = len(seperate_word)
        for i in range(1, num):
            combine_dict[seperate_word[i]] = seperate_word[0]
    return combine_dict
    
def tradition2simple(text):
    """Tradition Chinese corpus to simplify Chinese."""
    text = Converter('zh-hans').convert(text)
    return text
    
def clean_text(text):
    """Text filter for Chinese corpus, keep CN character and remove stopwords."""
    text = text.strip(' ')
    re_non_ch = re.compile(ur'[^\u4e00-\u9fa5(\*)+]+')
    text = re_non_ch.sub(''.decode('utf-8'), text)
    p = re.compile(ur"(\*)+(\1+)")
    text = p.sub(ur"\1",text)
    
    for w in stopwords:
        text = re.sub(w, '', text)
    return text

def create_charlist(input_file, output_file):
    """Text to character level split."""
    def getchar(text):
        text = tradition2simple(text)
        text = clean_text(text)
        char_list = []
        for char in text:
            char_list.append(char)
        return " ".join(char_list)

    with open(input_file) as fp:
        ret = []
        for line in fp:
            q={}
            lines=line.strip().split("\t")
            q['question1']=getchar(lines[1].decode('utf-8'))
            q['question2']=getchar(lines[2].decode('utf-8'))
            if(len(lines)==4):
                q['label']=lines[3]
            ret.append(q)
        df = pd.DataFrame(ret)
        if output_file is not None:
            df.to_csv(output_file, encoding="utf-8",index=False)
    return df

def create_wordlist(input_file, output_file, tongyi_file):
    """Text to word level split."""
    def seg(text, tongYiCiDict=[]):
        text = tradition2simple(text)
        text = clean_text(text)
        seg_list = jieba.cut(text)
        seg_list_new = []
        for word in seg_list:
            if word in tongYiCiDict:
                word = tongYiCiDict[word]
                seg_list_new.append(word)
            else:
                seg_list_new.append(word)
        return " ".join(seg_list_new)
    
    tongYiCiDict = tongYiCi(tongyi_file)
    with open(input_file) as fp:
        ret = []
        for line in fp:
            q={}
            lines=line.strip().split("\t")
            q['question1']=seg(lines[1].decode('utf-8'), tongYiCiDict)
            q['question2']=seg(lines[2].decode('utf-8'), tongYiCiDict)
            if(len(lines)==4):
                q['label']=lines[3].strip()
            ret.append(q)
        df = pd.DataFrame(ret)
        if output_file is not None:
            df.to_csv(output_file, encoding="utf-8", index=False)
    return df

def calculate_inter(list1, list2):
    """Calculating the intersection of two lists."""
    inter = list((set(list1).union(set(list2)))^(set(list1)^set(list2)))
    inter = ' '.join(inter)
    return inter

def create_inter(input_file, output_file):
    """Building the intersection of the tokens of two texts."""
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    
    data = pd.read_csv(input_file)
    data['inter'] = data.apply(lambda row: calculate_inter(row['question1'].split(' '), row['question2'].split(' ')), axis=1)
    
    data.to_csv(output_file, columns=['inter', 'label'], encoding="utf-8", index=False)
    return data[['inter', 'label']]

def calculate_diff(list1, list2):
    """Calculating the difference sets of two lists."""
    diff = list(set(list1)^set(list2))
    diff = ' '.join(diff)
    return diff

def create_diff(input_file, output_file):
    """Building the difference sets of the tokens of two texts."""
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    
    data = pd.read_csv(input_file)
    data['diff'] = data.apply(lambda row: calculate_diff(row['question1'].split(' '),row['question2'].split(' ')), axis=1)
    
    data.to_csv(output_file, columns=['diff', 'label'], encoding="utf-8", index=False)
    return data[['diff', 'label']]
    
def load_embedding(emb_path, index, dim, max_nb_words):
    """Loading pretrained word vectors."""
    embeddings_index = {}
    with open(emb_path,'r') as f:
        firstline = 0
        for i in f:
            if firstline == 0:
                firstline = 1
                continue
            values = i.strip().split(' ')
            word = values[0].decode('utf-8')
            embedding = np.asarray(values[1:],dtype='float')
            embeddings_index[word] = embedding
    print('word embedding',len(embeddings_index))
    
    # nb_words = min(max_nb_words, len(index))
    embedding_matrix = np.zeros((max_nb_words+1, dim))
    for word, i in index.items():
        if i > max_nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

def load_embedding_300(embeddings_300_path, index, max_nb_words):
    """Loading pretrained 300-dim word vectors."""
    embeddings_index = np.load(embeddings_300_path).tolist()
    # nb_words = min(max_nb_words, len(index))
    embedding_matrix = np.zeros((max_nb_words+1, 300))
    for word, i in index.items():
        if i > max_nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix
    
def create_vocab(Lists, name, datapath):
    """Creating dictionaries and counting words"""
    print("Creating vocabulary.")
    counter = Counter()
    for c in Lists:
        #c = c.decode('utf-8')
        c = c.split(' ')
        counter.update(c)
        
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= 0]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with codecs.open(datapath+'vocab_{}_count.txt'.format(name), "w", encoding="utf-8") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    with codecs.open(datapath+'vocab_{}.txt'.format(name), "w", encoding="utf-8") as f:
        f.write("\n".join(["%s" % w for w, c in word_counts]))
    print("Wrote vocabulary file:", 'vocab_{}.txt'.format(name))
