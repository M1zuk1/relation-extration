#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/5 16:43 
# ide： PyCharm

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config import Config
from relation_loader import RelationLoader
from tokenizer import Tokenizer

'''
构建训练语料
'''
class SemEvalCorpus(object):
    def __init__(self, config, rel2id):
        self.data_dir = config.data_dir
        self.rel2id = rel2id
        self.max_length = config.max_length
        self.catch_dir = config.catch_dir
        self.tokenizer = Tokenizer(config)
        self.vocab = None

    # 加载数据
    def __load_data(self, filetype):
        #先创建catch文件夹，将转为idx的训练数据保存到其中
        catch_file = os.path.join(self.catch_dir, filetype+'.pkl')
        # 如果文件存在，直接加载，不需要在执行后面的操作
        if os.path.exists(catch_file):
            data, labels = torch.load(catch_file)
        else:
            self.vocab = self.tokenizer.get_vocab()
            data_file = os.path.join(self.data_dir, filetype+".json")
            data = []
            labels = []
            with open(data_file, 'r', encoding='utf-8') as read_file:
                read_data = json.load(read_file)
                for i in tqdm(range(len(read_data))):
                    label = read_data[i]['relation']
                    sentence = read_data[i]['sentence']
                    label_idx = self.rel2id[label]
                    input_sentence = self.process_sentence(sentence)
                    data.append(input_sentence)
                    labels.append(label_idx)
            input_data = [data, labels]
            torch.save(input_data, catch_file,_use_new_zipfile_serialization=True)
        return data, labels



    '''
        Args: 
                sentence(list)
        Return:
                sent(ids): [CLS] ... $ e1 $ ... # e2 # ... [SEP] [PAD]
                mask     :   1    3  4  4 4  3  5  5 5  3    2     0
    '''
    def process_sentence(self,sentence):
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


        # 构建sentence_mask
        sentence_mask = [3] * len(sentence_token)
        sentence_mask[p11:p12+1] = [4] * (p12-p11+1)
        sentence_mask[p21:p22+1] = [5] * (p22-p21+1)


        # 判断长度
        if len(sentence_token) > self.max_length-2:
            sentence_token = sentence_token[:self.max_length-2]
            sentence_mask = sentence_mask[:self.max_length-2]


        #填充PAD
        pad_length = self.max_length - 2 - len(sentence_token)
        mask = [1] + sentence_mask + [2] + [0] * pad_length
        input_ids = self.vocab['[CLS]'] + sentence_token + self.vocab['[SEP]'] + self.vocab['[PAD]'] * pad_length

        assert len(mask) == self.max_length
        assert len(input_ids) == self.max_length

        unit = np.asarray([input_ids,mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=[1, 2, self.max_length])
        return unit



    def load_corpus(self, filetype):
        file_list = ['train', 'test', 'dev']
        if filetype in file_list:
            return self.__load_data(filetype)
        else:
            raise ValueError('mode error!')


class SemEvalDataset(Dataset):
    def __init__(self, data, labels):
        self.dataset = data
        self.label = labels

    def __getitem__(self, idx):
        return self.dataset[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


class SemEvalDataLoader(object):
    def __init__(self, config, rel2id):
        self.config = config
        self.rel2id = rel2id
        self.corpus = SemEvalCorpus(config, rel2id)

    def __collate_fn(self, batch):
        data,label = zip(*batch)
        data = list(data)
        label = list(label)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data,label


    def __get_data(self, filetype, shuffle=False):
        data, labels = self.corpus.load_corpus(filetype)
        dataset = SemEvalDataset(data, labels)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.__collate_fn
        )
        return loader

    def get_train(self):
        ret = self.__get_data(filetype='train', shuffle=True)
        print('finish loading train!')
        return ret

    def get_test(self):
        ret = self.__get_data(filetype='test', shuffle=False)
        print('finish loading test!')
        return ret

    def get_dev(self):
        ret = self.__get_data(filetype='test', shuffle=False)
        print('finish loading dev!')
        return ret

if __name__ == '__main__':
    config = Config()
    relation_loader = RelationLoader(config)
    sem_eval_corpus = SemEvalCorpus(config, relation_loader)
    data, label = sem_eval_corpus.load_corpus('train')
    # sem_eval_corpus = SemEvalDataset(data, label)
    # loader = SemEvalDataLoader(config, relation_loader)
    # ret = loader.get_train()
    # for data in ret:
    #     print(data)
    #     exit()