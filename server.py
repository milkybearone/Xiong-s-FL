import torch
from torch.nn import Sequential
from torchvision import models
import torch.nn as nn
import torchvision
from collections import OrderedDict

class Server(object):
    def __init__(self, conf, eval_dataset):
      # 导入配置文件 import config
        self.conf = conf
        modeule_name=str(self.conf['model_name'])
        # 根据配置获取模型文件 get the model based on conf.json
        if modeule_name == 'resnet50':
            self.global_model = models.get_model(conf['model_name'])
            self.global_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
            self.global_model.fc=nn.Linear(2048, 512)
            self.global_model.fc1=nn.Linear(512,conf['num_classes'])
            print(self.global_model)
            # 修改最后一层全连接层的数量，改为分类种类的数量 set the in channels as 1 and out features as num_classes(Here is 11)
        elif modeule_name == 'resnet18':
            self.global_model = models.get_model(conf['model_name'])
            # for param in self.global_model.parameters():
            #     param.requires_grad = False
            self.global_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
            self.global_model.fc = nn.Linear(512, 512)
            self.global_model.fc1 = nn.Linear(512, conf['num_classes'])
            print(self.global_model)
        elif modeule_name == 'alexnet':
            #net = alexnet.AlexNet().to(self.conf["DEVICE"])
            self.global_model = models.get_model(modeule_name)
            self.global_model.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                # 对于padding，它有两种类型，一是int型（如int 1）他是在特征矩阵上下左右各补一行（一列）零；
                # 另一个是tuple型（如tiple：（1，2）），1代表在特征矩阵上下各补一行零，2代表在特征矩阵左右两侧各补两行零。
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),  # inplace用这个是pytorch通过计算量减低内存使用，从而使内存加载更大的模型

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=2 ,stride=2),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),

                nn.MaxPool2d(kernel_size=3, stride=2),  # output[128, 6, 6]
                nn.ReLU(inplace=True),
            )
            self.global_model.fc1 = nn.Linear(256 * 3 * 3, 1024)
            self.global_model.fc2 = nn.Linear(1024, 512)
            self.global_model.fc3 = nn.Linear(512, 11)
            print(self.global_model)
        elif modeule_name == 'vgg16':
            self.global_model = models.vgg16()#plz change vgg16 ->
            self.global_model.classifier[6].out_features = conf['num_classes']
            print(self.global_model)
            #这里还没测试，跑完了测试一下，记得换一个新的 flag！！
        elif modeule_name == 'googlenet':
            self.global_model = models.googlenet()
            self.global_model.conv1 = nn.Conv2d(1,64,kernel_size=1,stride=1,padding=0)
            self.global_model.fc = nn.Linear(1024,conf['num_classes'])
        print("Loading "+self.conf["model_name"]+" result complete")


        # 生成一个测试集合加载器
        #generate the test data
        self.eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        # 设置单个批次大小32
        batch_size=self.conf["batch_size"],
        # 打乱数据集
        shuffle=True
        )

    # 全局聚合模型
    # weight_accumulator 存储了每一个客户端的上传参数变化值/差值
    # weight_accumulator has saved the sum of diff, diff comes from all clients
    def model_aggregate(self, weight_accumulator):
      # 遍历服务器的全局模型
        print(self.conf["algorithm"]+"ing...")
        for name, data in self.global_model.state_dict().items():
        # 更新每一层乘上学习率
            update_per_layer = weight_accumulator[name] * self.conf["lambda"]
        # 累加和
            if data.type() != update_per_layer.type():
            # 因为update_per_layer的type是floatTensor，所以将起转换为模型的LongTensor（有一定的精度损失）
            # datetype changing
                data.add_(update_per_layer.to(torch.int64))
            else:
                data.add_(update_per_layer)

    # 评估函数
    def model_eval(self):

        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        dataset_size = 0
        # 遍历评估数据集合
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            # 获取所有的样本总量大小
            dataset_size += data.size()[0]
            # 存储到gpu save in gpu
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            # 加载到模型中训练
            output = self.global_model(data)
            # 聚合所有的损失 cross_entropy交叉熵函数计算损失
            total_loss += torch.nn.functional.cross_entropy(
                output,
                target,
                reduction='sum'
            ).item()
            # 获取最大的对数概率的索引值， 即在所有预测结果中选择可能性最大的作为最终的分类结果
            pred = output.data.max(1)[1]
            # 统计预测结果与真实标签target的匹配总个数
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        acc = 100.0 * (float(correct) / float(dataset_size))    # 计算准确率 calculate accuracy
        total_1 = total_loss / dataset_size                     # 计算损失值 cal the loss
        return acc, total_1
