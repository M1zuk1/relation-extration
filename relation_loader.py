#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/5 15:48 
# ide： PyCharm

import os

from config import Config


class RelationLoader(object):
    def __init__(self,config):
        self.data_dir = config.data_dir


    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.__load_relation()

if __name__ == '__main__':
    config = Config()
    relationloader = RelationLoader(config)
    rel2id, id2rel, class_num = relationloader.get_relation()
    print(rel2id)
    print(id2rel)
    print(class_num)
