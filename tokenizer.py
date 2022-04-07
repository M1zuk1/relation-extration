#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/4 19:49 
# ide： PyCharm

import os
import json
from transformers import BertTokenizer
from config import Config




class Tokenizer(object):
    # 初始化tokenize
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.ptlm_dir = config.ptlm_dir
        self.tokenizer, self.special_tokens = self.load_tokenize()

    # 加载tokenize
    def load_tokenize(self):
        #  加载Bert分词器
        tokenizer = BertTokenizer.from_pretrained(self.ptlm_dir)
        # 分词器中添加sepcial——tokens,根据论文中将<e1></e1>,<e2></e2>改为$$和##
        special_tokens = ['$', '#']
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        return tokenizer, special_tokens

    # 构建数据的词表
    def build_vocab(self):
        vocab = set()
        file_list = ['train', 'test']
        for file_name in file_list:
            file_path = os.path.join(self.data_dir, file_name+'.json')
            with open(file_path, 'r', encoding='utf-8') as read_file:
                data = json.load(read_file)
                for i in range(len(data)):
                   for token in data[i]['sentence']:
                       if token in ['<e1>', '</e1>', '<e2>', '</e2>']:
                           continue
                       vocab.add(token)
        return vocab



    # 将vocab中的word切分成subword，然后转化为token_idx
    def get_vocab(self):
        vocab_set = self.build_vocab()
        vocab_dict = {}
        extra_tokens = ['[CLS]', '[SEP]', '[PAD]'] + self.special_tokens
        for token in extra_tokens:
            vocab_dict[token] = [self.tokenizer.convert_tokens_to_ids(token)]


        for token in vocab_set:
            token = token.lower()
            if token in vocab_dict.keys():
                continue
            # 切分子词
            token_res = self.tokenizer.tokenize(token)
            if len(token_res) < 1:
                token_idx_list = self.tokenizer.convert_tokens_to_ids('[UNK]')
            else:
                token_idx_list = self.tokenizer.convert_tokens_to_ids(token_res)

            vocab_dict[token] = token_idx_list
        return vocab_dict





if __name__ == '__main__':
    config = Config()
    tokenizer = Tokenizer(config)
    tokenizer.get_vocab()
