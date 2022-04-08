#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/8 19:51 
# ide： PyCharm


import torch
from config import Config
from model import R_Bert
from prepare_corpus import SemEvalDataLoader
from relation_loader import RelationLoader
from tqdm import tqdm


class Eval():
    def __init__(self, config):
        self.device = config.device

    def evaluate(self, model, data_loader):
        predict_label = []
        true_label = []
        total_loss = 0.0

        with torch.no_grad():
            model.eval()
            data_iterator = tqdm(data_loader, desc='Eval')
            for _, (data, label) in enumerate(data_iterator):
                print(data)
                exit()


if __name__ == '__main__':
        config = Config()
        eval = Eval(config)
        relation_loader = RelationLoader(config)
        rel2id, id2rel, class_num = relation_loader.get_relation()
        r_bert = R_Bert(config, class_num)
        #
        data_loader = SemEvalDataLoader(config,rel2id).get_train()
        eval.evaluate(r_bert, data_loader)

