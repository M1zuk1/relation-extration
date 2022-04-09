#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/7 22:35 
# ide： PyCharm

import torch
import torch.nn as nn
from transformers import BertConfig
from transformers import BertModel
from config import Config
from relation_loader import RelationLoader
from prepare_corpus import SemEvalDataLoader


class R_Bert(nn.Module):
    def __init__(self, user_config, class_num):
        super(R_Bert, self).__init__()
        self.class_num = class_num

        # 超参数设置
        bert_config = BertConfig.from_pretrained(user_config.ptlm_dir)
        self.bert = BertModel.from_pretrained(user_config.ptlm_dir)
        self.bert_hidden_size = bert_config.hidden_size

        self.max_len = user_config.max_length

        # 模型结构
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(user_config.dropout)

        self.cls_mlp = nn.Linear(
            in_features=self.bert_hidden_size,
            out_features=self.bert_hidden_size,
            bias=True
        )

        self.entity_mlp =  nn.Linear(
            in_features=self.bert_hidden_size,
            out_features=self.bert_hidden_size,
            bias=True
        )

        self.dense = nn.Linear(
            in_features=self.bert_hidden_size*3,
            out_features=self.class_num,
            bias=True
        )

        self.cretrion = nn.CrossEntropyLoss()
        self.init_parameters()

    # 初始化模型参数
    def init_parameters(self):
        nn.init.xavier_normal_(self.cls_mlp.weight)
        nn.init.constant_(self.cls_mlp.bias, 0.)
        nn.init.xavier_normal_(self.entity_mlp.weight)
        nn.init.constant_(self.entity_mlp.bias, 0.)
        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.)

    # bert层
    def bert_layer(self,input_ids, attention_mask, token_type_ids):
        output = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
        )
        hidden_output = output[0] # batch*max_len*hidden_size
        pooler_output = output[1] # batch*1
        return hidden_output, pooler_output

    # 实体表示求平均
    def avg_entity_reps(self, hidden_output, e_mask):
        e_mask_ad = e_mask.unsqueeze(dim=1).float()  # B*1*L
        sum_reps = torch.bmm(e_mask_ad, hidden_output)  # B*1*L * B*L*H -> B*1*H
        sum_reps = sum_reps.squeeze(dim=1)  # B*1*H -> B*H
        entity_lens = e_mask_ad.sum(dim=-1).float()  # B*1
        avg_reps = torch.div(sum_reps, entity_lens)
        return avg_reps

    def forward(self, data, label):
        input_ids = data[:,0,:] # batch * max_len
        mask = data[:,1,:] # batch * max_len

        attention_mask = mask.gt(0).float()
        token_type_ids = mask.gt(-1).long()

        hidden_outpt, pooler_output = self.bert_layer(input_ids, attention_mask, token_type_ids)

        cls_resp = self.dropout(pooler_output)
        cls_resp = self.tanh(self.cls_mlp(cls_resp))

        mask_e1 = mask.eq(4)
        e1_reps = self.avg_entity_reps(hidden_outpt, mask_e1)
        e1_reps = self.dropout(e1_reps)
        e1_reps = self.tanh(self.entity_mlp(e1_reps))

        mask_e2 = mask.eq(5)
        e2_reps = self.avg_entity_reps(hidden_outpt, mask_e2)
        e2_reps = self.dropout(e2_reps)
        e2_reps = self.tanh(self.entity_mlp(e2_reps))

        # 将cls e1 e2的表示连接起来
        reps = torch.cat([cls_resp, e1_reps, e2_reps], dim=1)
        reps = self.dropout(reps)
        logits = self.dense(reps)


        # 计算损失，torch.nn.CrossEntropyLoss()的input只需要是网络fc层的输出𝑦, 在torch.nn.CrossEntropyLoss()里它会自己把𝑦转化成𝑠𝑜𝑓𝑡𝑚𝑎𝑥(𝑦)
        loss = self.cretrion(logits, label)

        return loss, logits




if __name__ == '__main__':
    user_config = Config()
    relationloader = RelationLoader(user_config)
    rel2id, id2rel, class_num = relationloader.get_relation()

    r_bert = R_Bert(user_config, class_num)
    relation_loader = RelationLoader(user_config)
    loader = SemEvalDataLoader(user_config, relation_loader)
    ret = loader.get_train()
    for data in ret:
        r_bert(data[0],data[1])
    # r_bert.forward()
