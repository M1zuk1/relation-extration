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
import numpy as np
from sklearn.metrics import f1_score



class Eval():
    def __init__(self, config):
        self.device = config.device

    def evaluate(self, model, data_loader):
        predict_label = []
        true_label = []
        total_loss = 0.0

        with torch.no_grad():
            model.eval()
            data_iterator = tqdm(data_loader, desc="Eval")
            for step, (data, label) in enumerate(data_iterator):
                data = data.to(self.device)
                label = label.to(self.device)

                loss, logits = model(data, label)

                # 将loss从tensor转为float
                total_loss += loss.item()

                pred = torch.argmax(logits, dim=1)
                pred = pred.cpu().detach().numpy().reshape((-1, 1))
                label = label.cpu().detach().numpy().reshape((-1, 1))
                predict_label.append(pred)
                true_label.append(label)
            predict_label = np.concatenate(
                predict_label, axis=0).reshape(-1).astype(np.int64)
            true_label = np.concatenate(
                true_label, axis=0).reshape(-1).astype(np.int64)
            eval_loss = total_loss / len(data_loader)

        f1 = f1_score(true_label, predict_label, average='weighted')

        return f1, eval_loss, predict_label



if __name__ == '__main__':
        config = Config()
        eval = Eval(config)
        relation_loader = RelationLoader(config)
        rel2id, id2rel, class_num = relation_loader.get_relation()
        r_bert = R_Bert(config, class_num)
        #
        data_loader = SemEvalDataLoader(config,rel2id).get_test()
        eval.evaluate(r_bert, data_loader)

