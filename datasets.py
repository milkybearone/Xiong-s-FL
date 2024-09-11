import torchvision as tv
import torch
import json
import numpy as np

import torch.nn.functional as F
# 获取数据集
def get_dataset(dir, name):
    with open('./utils/conf.json', 'r') as f:
        conf = json.load(f)
    model_name=conf['model_name']
    batch_size=conf['batch_size']
    download_dataset = conf['download_dataset']
    if  download_dataset==0:
        download_flag=False
    else:
        download_flag=True

    if name == 'mnist':

        # root: 数据路径
        # train参数表示是否是训练集或者测试集
        # download=true表示从互联网上下载数据集并把数据集放在root路径中
        # transform：图像类型的转换
        if model_name=='alexnet':
            train_dataset = tv.datasets.MNIST(dir, train=True, download=download_flag,
                                              transform=tv.transforms.Compose([
                                                  tv.transforms.ToTensor(),
                                                #tv.transforms.Normalize((0.1307,), (0.3081,)),
                                           ]))

            eval_dataset = tv.datasets.MNIST(dir, train=False, download=download_flag,
                                             transform=tv.transforms.Compose([
                                                 tv.transforms.ToTensor(),
                                                #tv.transforms.Normalize((0.1307,), (0.3081,)),

                                           ]))
        elif model_name=='resnet50':
            train_dataset = tv.datasets.MNIST(dir, train=True, download=download_flag,
                                              transform=tv.transforms.ToTensor())
            eval_dataset = tv.datasets.MNIST(dir, train=False, transform=tv.transforms.ToTensor())

        elif model_name=='vgg16':
            train_dataset = tv.datasets.MNIST(dir, train=True, download=download_flag,
                                              transform=tv.transforms.Compose([
                                                  tv.transforms.Resize((64, 64)),
                                                  tv.transforms.ToTensor()]
                                              ))
            eval_dataset = tv.datasets.MNIST(dir, train=False, transform=tv.transforms.Compose([
                                                  tv.transforms.Resize((64,64)),
                                                    tv.transforms.ToTensor()]
                                                  ))
        elif model_name=='googlenet':
            train_dataset = tv.datasets.MNIST(dir, train=True, download=download_flag,
                                              transform=tv.transforms.Compose([
                                                  tv.transforms.ToTensor(),
                                                  tv.transforms.Normalize((0.15,), (0.30,)),

                                              ]))

            eval_dataset = tv.datasets.MNIST(dir, train=False, download=download_flag,
                                             transform=tv.transforms.Compose([
                                                 tv.transforms.ToTensor(),
                                                 tv.transforms.Normalize((0.15,), (0.30,)),

                                             ]))


            # train_dataset = tv.datasets.MNIST(dir, train=True, download=download_flag,
            #                                transform=tv.transforms.Compose([
            #                                     tv.transforms.ToTensor(),
            #                                ]))
            # eval_dataset = tv.datasets.MNIST(dir, train=False,
            #                                transform=tv.transforms.Compose([
            #                                     tv.transforms.ToTensor(),
            #                                ]))

    elif name == 'cifar':
            # 设置两个转换格式
            # transforms.Compose 是将多个transform组合起来使用（由transform构成的列表）
            transform_train = tv.transforms.Compose([
                # transforms.RandomCrop： 切割中心点的位置随机选取
                tv.transforms.RandomCrop(32, padding=4), tv.transforms.RandomHorizontalFlip(),
                tv.transforms.ToTensor(),
                # transforms.Normalize： 给定均值：(R,G,B) 方差：（R，G，B），将会把Tensor正则化
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            train_dataset = tv.datasets.CIFAR10(dir, train=True, download=True, transform=transform_train)
            eval_dataset = tv.datasets.CIFAR10(dir, train=False, transform=transform_test)

    return train_dataset, eval_dataset
