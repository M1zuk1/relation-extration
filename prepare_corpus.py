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





if __name__ == '__main__':
    config = Config()
    relation_loader = RelationLoader(config)
    sem_eval_corpus = SemEvalCorpus(config, relation_loader)
    print(sem_eval_corpus.__dict__)