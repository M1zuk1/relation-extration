#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/4 19:49 
# ide： PyCharm

import torch
import os
import json
from torch.utils.data import DataLoader,Dataset

class Tokenizer(object):
    def __init__(self, config):
        self.data_dir = config.data_dir
        self.ptlm_dir = config.ptlm_dir
        self.tokenizer = self.__load_tokenize()

    def __load_tokenize(self):
        pass

    def __build_vocab(self):
        pass


    def get_vocab(self):
        pass
