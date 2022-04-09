#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/7 22:35 
# ide： PyCharm
import torch
from tqdm import tqdm
from config import Config
from relation_loader import RelationLoader
from prepare_corpus import SemEvalDataLoader
import os
import torch.optim as optim
from model import R_Bert
from evaluate import Eval
from transformers import WEIGHTS_NAME, CONFIG_NAME



class Runner():
    def __init__(self, user_config, id2rel, loader):
        self.class_num = len(id2rel)
        self.id2rel = id2rel
        self.loader = loader
        self.user_config = user_config

        self.model = R_Bert(user_config, self.class_num)
        self.model = self.model.to(user_config.device)
        self.eval = Eval(user_config)

    def train(self):
        train_loader, test_loader, dev_loader = self.loader
        train_steps = len(train_loader) * self.user_config.epoch
        warmup_steps = train_steps * self.user_config.warmup

        # id():返回对象的内存地址
        bert_params = list(map(id, self.model.bert.parameters()))
        for parameter in self.model.bert.parameters():
            parameter.requires_grad = False
        # 过滤器，过滤掉在bert_params中的参数，lambda表示式为过滤条件
        # rest_params = filter(lambda p: id(p) not in bert_params, self.model.parameters())
        rest_params = filter(lambda p: p.requires_grad, self.model.parameters())
        # Todo: 添加warmup
        # 先不用warmup看看
        optimizer = optim.AdamW(rest_params, lr=self.user_config.lr)

        # 打印模型训练的参数
        print('----------------------------------------------')
        print('traning model parameters (except PLM layers):')
        for name, param in self.model.named_parameters():
            if id(param) in bert_params:
                continue
            elif param.requires_grad:
                print('{}: {}'.format(name, str(param.shape)))

        print('----------------------------------------------')
        print('start to train the model......')
        max_f1 = 0.0

        for epoch in range(1, 1 + self.user_config.epoch):
            self.model.train()
            train_loss = 0.0
            data_iterator = tqdm(train_loader, desc="Train")

            for step, (data, label) in enumerate(data_iterator):
                data = data.to(self.user_config.device)
                label = label.to(self.user_config.device)

                optimizer.zero_grad()
                loss, logits = self.model(data, label)
                loss.backward()
                train_loss += loss.item()

                # todo:torch.nn.utils.clip_grad_norm,防止梯度爆炸，查看loss情况再决定是否要添加

                optimizer.step()

            train_loss = train_loss / len(data_iterator)
            f1, eval_loss = self.eval.evaluate(self.model, dev_loader)
            print('epoch:{}｜ train_loss:{:.3f} | dev_loss:{:.3f} | f1 on dev:{:.4f}'.format(epoch, train_loss, eval_loss, f1))

            if f1 > max_f1:
                max_f1 = f1
                output_model_file = os.path.join(
                    self.user_config.model_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(
                    self.user_config.model_dir, CONFIG_NAME)

                model_to_save = self.model.module if hasattr(
                    self.model, 'module') else self.model
                model_to_save.bert.config.to_json_file(output_config_file)
                print('>>> save models!')




    def test(self):
        print('----------------------------------------------')
        print('start to load the model......')
        if not os.path.exists(self.user_config.model_dir):
            raise Exception('no model exists!')

        # 加载模型
        load_path = os.path.join(self.user_config.model_dir, WEIGHTS_NAME)
        state_dict = torch.load(
            load_path,
            map_location=self.user_config.device
        )
        self.model.load_state_dict(state_dict)

        print('----------------------------------------------')
        print('start to test......')
        _,test_loader,_ = self.loader
        f1, test_loss, predict_label = self.eval_tool.evaluate(self.model, test_loader)
        print('test_loss:{:.3f} | micro f1 on test:{:.4f}' % (test_loss, f1))
        return predict_label



if __name__ == '__main__':
    # 打印配置
    user_config = Config()
    print('----------------------------------------------')
    print('config:')
    for key in user_config.__dict__:
        config = "  " + key + ' = ' + str(user_config.__dict__[key])
        print(config)

    # 开始加载数据
    print('----------------------------------------------')
    print('start to load data')
    rel2id, id2rel, class_num = RelationLoader(user_config).get_relation()
    loader = SemEvalDataLoader(user_config, rel2id)

    # 判断是训练还是测试
    train_loader = None
    test_loader = None
    dev_loader = None
    if user_config.mode == 0:
        train_loader = loader.get_train()
        test_loader = loader.get_test()
        dev_loader = loader.get_dev()
    elif user_config.mode == 1:
        test_loader = loader.get_test()
    loader = [train_loader, test_loader, dev_loader]
    print("loading finished!")

    runner = Runner(user_config, id2rel, loader)

    # 训练
    if user_config.mode == 0:
        runner.test()

    # 测试
    elif user_config.mode == 1:
        runner.test()
