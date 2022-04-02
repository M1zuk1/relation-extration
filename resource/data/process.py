#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/2 11:26
# ide： PyCharm

import json
import re

'''
处理数据集，对数据格式进行处理，方便进行训练和测试
'''


def split_sentence(sentence):
    # 找到两个实体
    e1 = re.findall(r'<e1>(.*)</e1>', sentence)[0]
    e2 = re.findall(r'<e2>(.*)</e2>', sentence)[0]

    sentence = sentence.replace('<e1>'+e1+'</e1>', '<e1> '+e1+' </e1>')
    sentence = sentence.replace('<e2>'+e2+'</e2>', '<e2> '+e2+' </e2>')
    sentence = sentence.replace('.', ' .')
    sentence = sentence.split(' ')
    return sentence





def convert(file_path, file_name):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.readlines()
    file.close()
    process_data = []

    with open(file_name, 'w', encoding='utf-8') as file:
        for i in range(0, len(data), 4):
            temp = {}
            idx, sentence = data[i].strip().split('\t')
            #去除双引号
            sentence = sentence[1:-1]
            sentence = split_sentence(sentence)
            temp['idx'] = idx
            temp['relation'] = data[i+1].strip()
            temp['sentence'] = sentence
            temp['comment'] = data[i+2].strip()[9:]
            process_data.append(temp)
        file.write(json.dumps(process_data, ensure_ascii=False,indent=4))
        file.close()





if __name__ == '__main__':
    tarin_path = 'FULL_TRAIN.txt'
    test_path = 'FULL_TEST.txt'

    convert(tarin_path, 'train.json')
    convert(test_path, 'test.json')