#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/4 17:15 
# ide： PyCharm

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/2 23:37
# ide： PyCharm

import argparse
import json
import os
import random

import numpy as np
import torch


class Config(object):
    # Config初始化
    def __init__(self):
        # 获得初始设置的参数
        args = self.__get_config()
        # 设置参数
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # 选择设备
        self.device = None
        # if torch.cuda.is_available() and self.cuda >= 0:
        #     self.device = torch.cuda('cuda:{}'.format(self.cuda))
        # else:
        #     self.device = torch.device('cpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 参数相关的文件夹，判断是否存在，不存在的话创建

        # 创建输出的模型文件夹
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # 创建预训练语言模型的文件夹
        self.ptlm_dir = os.path.join(self.ptlm_root_dir, self.ptlm_name)
        if not os.path.exists(self.ptlm_dir):
            os.makedirs(self.ptlm_dir)

        # catch文件夹
        if not os.path.exists(self.catch_dir):
            os.makedirs(self.catch_dir)

        # 将参数写入文件中保存
        self.__store_config(args)

        # 设置随机数中心
        self.__set_random_seed(self.seed)

    # 创建私有方法保护参数
    def __get_config(self):
        # 实例化ArgumentParser
        parser = argparse.ArgumentParser(description="config for model to relation extration")

        # 添加参数
        parser.add_argument('--data_dir', type=str, default=r'./resource/data', help='direct to load data')
        parser.add_argument('--output_dir', type=str, default=r'./output', help='direct to save output data')
        parser.add_argument('--catch_dir', type=str, default=r'./catch', help='direct to save catch data')

        # 添加训练参数
        parser.add_argument('--model_name', type=str, default='R-Bert', help="name of the model")
        parser.add_argument('--mode', type=int, default=0, choices=[0, 1],
                            help='running mode: 0 for training, 1 for testing')
        parser.add_argument('--seed', type=int, default=1234, help='random seed')
        parser.add_argument('--cuda', type=int, default=0, help='num of gpu device, if -1, select cpu')

        # 预训练语言模型相关参数
        parser.add_argument('--ptlm_root_dir', type=str, default=r'./resource',
                            help='diret to load pre-trained language model')
        parser.add_argument('--ptlm_name', type=str, default='bert-base-uncased',
                            help='dir of pre-trained language model')

        # 超参数
        parser.add_argument('--epoch', type=int, default=100, help='max epoches to train the model')
        parser.add_argument('--max_length', type=int, default=128, help='max lengh of the sentence after tokenization')
        parser.add_argument('--lr', type=float, default=1e-5, help='learning rate in ptlm layer')
        parser.add_argument('--other_lr', type=float, default=2e-5,
                            help='learning rate in other layers except ptlm layer')
        parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        parser.add_argument('--weight_deca', type=float, default=0.0, help='weight decay')
        parser.add_argument('--dropout', type=float, default=0.1, help='the possibility of dropout')
        parser.add_argument('--warmup', type=float, default=0.1, help='proportion of linear warmup over warmup_steps')
        parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='epsilon for Adam optimizer')

        args = parser.parse_args()
        return args

    def __store_config(self, args):
        config_store_path = os.path.join(self.model_dir, 'config.json')
        with open(config_store_path, 'w', encoding='utf-8') as config_file:
            config_file.write(json.dumps(vars(args), ensure_ascii=False, indent=4))
        config_file.close()

    def __set_random_seed(self, seed=1234):
        os.environ['PYTHONHASHSEED'] = str(seed)
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        # set seed for cpu
        torch.manual_seed(seed)
        # set seed for current gpu
        torch.cuda.manual_seed(seed)
        # set seed for all gpu
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    config = Config()
