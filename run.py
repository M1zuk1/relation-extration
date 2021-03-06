#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： mizuki
# datetime： 2022/4/7 22:35
# ide： PyCharm
import os

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import WEIGHTS_NAME, CONFIG_NAME
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter

from config import Config
from evaluate import Eval
from model import R_Bert
from prepare_corpus import SemEvalDataLoader
from relation_loader import RelationLoader


class Runner():
    def __init__(self, user_config, id2rel, loader):
        self.class_num = len(id2rel)
        self.id2rel = id2rel
        self.loader = loader
        self.user_config = user_config

        self.model = R_Bert(user_config, self.class_num)
        # self.model = torch.nn.parallel(self.model)
        self.model = self.model.to(user_config.device)
        self.eval = Eval(user_config)

    def train(self):
        train_loader, test_loader, dev_loader = self.loader
        train_steps = len(train_loader) * self.user_config.epoch
        warmup_steps = train_steps * self.user_config.warmup

        # id():返回对象的内存地址
        bert_params = list(map(id, self.model.bert.parameters()))
        # 过滤器，过滤掉在bert_params中的参数，lambda表示式为过滤条件
        rest_params = filter(lambda p: id(p) not in bert_params, self.model.parameters())
        # rest_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Bert层和其它层使用不同的学习率进行训练
        optimizer_grouped_parameters = [
            {'params': self.model.bert.parameters()},
            {'params': rest_params, 'lr': self.user_config.other_lr},
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.user_config.lr,
            eps=self.user_config.adam_epsilon
        )
        # warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=train_steps
        )

        # 打印模型训练的参数
        print('----------------------------------------------')
        print('training model parameters (except PLM layers):')
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

                optimizer.step()
                scheduler.step()

            train_loss = train_loss / len(data_iterator)
            f1, eval_loss, predict_label = self.eval.evaluate(self.model, dev_loader)
            # 将 train accuracy 保存到 "tensorboard/train" 文件夹
            log_dir = os.path.join('tensorboard', 'F1-Score')
            train_writer = SummaryWriter(log_dir=log_dir)
            # 将 test accuracy 保存到 "tensorboard/test" 文件夹
            log_dir = os.path.join('tensorboard', 'Eval Loss')
            test_writer = SummaryWriter(log_dir=log_dir)

            # 绘制
            train_writer.add_scalar('Train', f1, epoch)
            test_writer.add_scalar('Train', eval_loss, epoch)
            print(
                'epoch:{}｜ train_loss:{:.3f} | dev_loss:{:.3f} | f1 on dev:{:.4f}'.format(epoch, train_loss, eval_loss,
                                                                                          f1))

            if f1 > max_f1:
                max_f1 = f1
                output_model_file = os.path.join(
                    self.user_config.model_dir, WEIGHTS_NAME)
                output_config_file = os.path.join(
                    self.user_config.model_dir, CONFIG_NAME)

                # 保存模型
                model_to_save = self.model.module if hasattr(
                    self.model, 'module') else self.model
                model_to_save.bert.config.to_json_file(output_config_file)
                torch.save(model_to_save.state_dict(), output_model_file)
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
        _, test_loader, _ = self.loader
        f1, test_loss, predict_label = self.eval.evaluate(self.model, test_loader)
        print('test_loss:{:.3f} | micro f1 on test:{:.4f}'.format(test_loss, f1))
        return predict_label


def print_result(predict_label, id2rel, start_idx=8001):
    predict_dir = './eval'
    predict_file = 'predicted_result.txt'
    predict_path = os.path.join(predict_dir, predict_file)
    if not os.path.exists(predict_dir):
        os.makedirs(predict_dir)
    with open(predict_path, 'w', encoding='utf-8') as wf:
        for i in range(0, predict_label.shape[0]):
            wf.write('{}\t{}\n'.format(
                start_idx + i, id2rel[int(predict_label[i])]))


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
        runner.train()
        predict_label = runner.test()
    # 测试
    elif user_config.mode == 1:
        predict_label = runner.test()
    print_result(predict_label, id2rel)
