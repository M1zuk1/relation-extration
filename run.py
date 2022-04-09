#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/7 22:35 
# ide： PyCharm

from tqdm import tqdm
from config import Config
from relation_loader import RelationLoader
from prepare_corpus import SemEvalDataLoader


import torch
import torch.nn as nn
import torch.optim as optim
from model import R_Bert
from evaluate import Eval



class Runner():
    def __init__(self, user_config, id2rel, loader):
        self.class_num = len(id2rel)
        self.id2rel = id2rel
        self.loader = loader
        self.user_config = user_config

        self.model = self.R_bert(user_config, self.class_num)
        self.model = self.model.to(user_config.device)
        self.eval = Eval(user_config)







if __name__ == '__main__':
    user_config = Config()
    relationloader = RelationLoader(user_config)
    rel2id, id2rel, class_num = relationloader.get_relation()
    loader = SemEvalDataLoader(user_config, rel2id)
    runner = Runner(user_config,id2rel, loader)