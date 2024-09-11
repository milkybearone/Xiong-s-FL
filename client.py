import torch
import json
import torch.nn as nn
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('./utils/conf.json', 'r') as f:
    conf = json.load(f)
local_epoch = conf['local_epochs']
al = conf["algorithm"]
poison = conf['poison_client']
q=conf['q']
b=conf['b']
f.close()

if al == "fedsgd":
    local_epoch = 1

# -*- coding:utf-8 -*-



class Client(object):
    def __init__(self, conf, model, train_dataset, id = 1,poison_flag=0):
        self.conf = conf
        self.local_model = model
        self.client_id = id
        self.train_dataset = train_dataset
        self.poison_flag = poison_flag
        #some config
        all_range = list(range(len(self.train_dataset)))
        data_len = int(len(self.train_dataset) / self.conf['no_models'])
        indices = all_range[id * data_len: (id + 1) * data_len]
        # 生成一个数据加载器 generate the data loader
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            # batch_size=32
            batch_size=conf["batch_size"],
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices)
        )



    # 模型本地训练函数
    def local_train(self, model,poison_trigger,first_time_flag):
        total_decription_time=0
        # 整体的过程：拉取服务器的模型，通过部分本地数据集训练得到 get the global model from the server
        if first_time_flag==1 or q==0:
            for name, param in model.state_dict().items():
                # 客户端首先用服务器端下发的全局模型覆盖本地模型 override local model with global model
                self.local_model.state_dict()[name].copy_(param.clone())
        else:
            for name, param in model.state_dict().items():
                # 客户端首先用服务器端下发的全局模型覆盖本地模型 override local model with global model
                #不是第一次的话就需要解密
                decription_time_s=time.time()
                param=(param-b)/q
                decription_time_e = time.time()

                decription_time_this_param = decription_time_e-decription_time_s
                total_decription_time+=decription_time_this_param
                self.local_model.state_dict()[name].copy_(param.clone())
        #保存服务器给的原始的模型  save the global model for FedProx

        ori_global_model = model
        ori_global_model=ori_global_model.cuda()

        # 定义最优化函数器用于本地模型训练
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.conf['lr'], momentum=self.conf['momentum'])
        if al=="fedprox":
            loss_function = nn.MSELoss().to(self.conf['DEVICE'])
        # 本地训练模型  train locally
        self.local_model.train()        # 设置开启模型训练（可以更改参数）
        # 开始训练模型 start training
        for e in range(local_epoch):
            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                # 加载到gpu
                if torch.cuda.is_available():
                    model = model.cuda()
                    data = data.cuda()
                    target = target.cuda()
                # 梯度
                optimizer.zero_grad()
                # 训练预测
                output = self.local_model(data)
                #计算本地模型和服务器给的初始模型的差距
                if al=='fedprox':
                    proximal_term =0.0
                    for w,w_t in zip(self.local_model.parameters(), ori_global_model.parameters()):
                        proximal_term +=(w - w_t).norm(2)
                    loss = torch.nn.functional.cross_entropy(output, target)
                    prox = (self.conf["mu_for_fedprox"]/2)*proximal_term
                    loss+=(self.conf["mu_for_fedprox"]/2)*proximal_term
                else :#这是其他聚合算法的正常loss计算
                    loss = torch.nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()

            #print("Epoch %d done" % e)
        # 创建差值字典（结构与模型参数同规格），用于记录差值
        #diff is the difference between local model and the global model after local training
        diff = dict()
        print("client ", self.client_id,end=' ')
        tensor_name = ''
        #tensor_name is the name of the last object in diff
        total_encription_time=0
        for name, data in self.local_model.state_dict().items():
            last_layer_name = name

            # poisondata = data
            if self.poison_flag == self.client_id and poison_trigger == 1:
                # if poison%2==0:
                #     diff[name] = (data - model.state_dict()[name])*0.5
                # else:
                diff[name] = data
            #if poison = 1, client will send back the local model's state_dict
            # instead of the diff between local trained model and global model
            # 计算训练后与训练前的差值
            else:
                diff[name] = (data - model.state_dict()[name])
            if q != 0:
                encription_t_s = time.time()
                diff[name]=diff[name]*q+b
                encription_e = time.time()
                encription_time = encription_e - encription_t_s
                total_encription_time+=encription_time
        # if poison % 2 == 0:
        #     print('偶数个投毒，使用系数而不是返回整个模型')
        # else:
        #     print('奇数个投毒，返回整个模型')
        if q!=0:
            print('加密时间:',total_encription_time,' 解密时间：',total_decription_time,end='')
            return diff, last_layer_name , total_encription_time , total_decription_time
        else:
            return diff, last_layer_name
