#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/5 16:43 
# ide： PyCharm

import os
import json
import torch
from config import Config
from relation_loader import RelationLoader
from tokenizer import Tokenizer
from tqdm import tqdm
import numpy as np

'''
构建训练语料
'''
class SemEvalCorpus(object):
    def __init__(self, config, loader):
        self.data_dir = config.data_dir
        self.rel2id, self.id2rel, self.class_num = loader.get_relation()
        self.max_length = config.max_length
        self.catch_dir = config.catch_dir
        self.tokenizer = Tokenizer(config)
        self.vocab = None

    # 加载数据
    def load_data(self, filetype):
        #先创建catch文件夹，将转为idx的训练数据保存到其中
        catch_file = os.path.join(self.catch_dir, filetype+'.pkl')
        # 如果文件存在，直接加载，不需要在执行后面的操作
        if os.path.exists(catch_file):
            data, labels = torch.load(catch_file)
        else:
            self.vocab = self.tokenizer.get_vocab()
            data_file = os.path.join(self.data_dir, filetype+".json")
            with open(data_file, 'r', encoding='utf-8') as read_file:
                read_data = json.load(read_file)
                for i in tqdm(range(len(read_data))):
                    label = read_data[i]['relation']
                    sentence = read_data[i]['sentence']
                    label_idx = self.rel2id[label]
                    sentence_idx = self.sentence2idx(sentence)
                    if sentence_idx==False:
                        print(read_data[i]['idx'])
                        exit()




    '''
        Args: 
                sentence(list)
        Return:
                sent(ids): [CLS] ... $ e1 $ ... # e2 # ... [SEP] [PAD]
                mask     :   1    3  4  4 4  3  5  5 5  3    2     0
    '''
    def sentence2idx(self,sentence):
        sentence_token = []
        sentence_mask = []
        # e1的位置(p11,p12),e2的位置(p21,p22)
        p11=p12=p21=p22=-1
        for word in sentence:
            word = word.lower()
            if word == '<e1>':
                p11 = len(sentence_token)
                sentence_token += self.vocab['$']
            elif word == '</e1>':
                p12 = len(sentence_token)
                sentence_token += self.vocab['$']
            elif word == '<e2>':
                p21 = len(sentence_token)
                sentence_token += self.vocab['#']
            elif word == '</e2>':
                p22 = len(sentence_token)
                sentence_token += self.vocab['#']
            else:
                sentence_token += self.vocab[word]
        if p11 == -1 or p12 ==-1 or p21==-1 or p22==-1:
            return False






if __name__ == '__main__':
    config = Config()
    relation_loader = RelationLoader(config)
    sem_eval_corpus = SemEvalCorpus(config, relation_loader)
    sem_eval_corpus.load_data('train')
