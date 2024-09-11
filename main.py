import argparse
import random
import sys,time
import numpy as np
import datasets
from client import *
from server import *
import openpyxl
def euclidean_distance(x, y):
    return np.linalg.norm(x - y)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

title=['acc', 'loss', 'period', 'client with max time takes for this round', 'time takes']
#title for table
wb=openpyxl.Workbook()
ws=wb.active
ws.append(title)   # 写入表头
total_time=0

if __name__ == '__main__':
    # 设置命令行程序
    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    # 获取所有的参数
    args = parser.parse_args()

    # 读取配置文件，需要改为你自己的路径，你甚至可以自己写一个dist变量把config存在里面
    # read the conf.json, the purpose of all conf plz read Readme.md
    with open('./utils/conf.json', 'r') as f:
        conf = json.load(f)
    al = conf['algorithm']
    local_epoch = conf['local_epochs']
    # 获取数据集, 加载描述信息
    current_al = conf["algorithm"]
    dataset_name = conf["type"]
    model_name = conf["model_name"]
    global_epoches = conf["global_epochs"]
    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])
    poison = conf["poison_client"]
    no_models = conf["no_models"]
    krum = conf["krum"]
    krum_client = conf["krum_client"]
    poison_time = int(global_epoches/conf['poison_times'])
    f.close()

    #fedsgd 将本地训练改为一次  全局训练改为两者相乘的次数，总次数与fedavg相同，但是通信次数大大增加
    #if algorithm is FedSGD,then make the local epoch as 1 and the global epoch as local_epoch*global epoch
    #in our experiment, which is 1 local epoch and 120 global epochs
    if al == "fedsgd":
        global_epoches *= local_epoch
    print("current al:"+al,'  投毒次数:',conf['poison_times'])
    # 开启服务器 turn on the server
    server = Server(conf, eval_datasets)
    # 客户端列表  list of clients
    candidates = []


    # 添加若干个客户端到列表,默认无毒
    # add "no_models" client, which is 10, no poison client currently
    for c in range(conf["no_models"]):
        candidates.append(Client(conf, server.global_model, train_datasets, c,poison_flag=-1))

    #添加投毒的客户端
    #if poisoning, then the poison_flag of each client will be set as their client_id
    if poison>0:
        print("poison client ", end='')
        poison_client_id = random.sample(range(0,no_models),poison)
        for c in candidates:
            if c.client_id in poison_client_id:
                c.poison_flag = c.client_id
                print(str(c.poison_flag)+',',end='')

    print('\n')

    round_no=0
    ten_round_en_time = 0
    ten_round_de_time=0
    first_round=1
    #these are variables for time calculation

    # 全局模型训练 global model training
    for e in range(global_epoches):

        round_no+=1
        time_assingment_s = time.time()
        print("Global Epoch " +str(e)+'/'+ str(global_epoches)
              +'. original global epoch:'+str(conf['global_epochs']))

        #progress_bar(e, conf["global_epochs"])
        # 权重累计
        if(e%poison_time==0):
            poison_trigger=1
        else:
            poison_trigger=0
        weight_accumulator = {}

        # 初始化空模型参数 initializing weight_accumulator
        for name, params in server.global_model.state_dict().items():
            # 生成一个和参数矩阵大小相同的0矩阵
            # generate a matrix which share the same size of the state_dict
            weight_accumulator[name] = torch.zeros_like(params)

        time_assingment_e=time.time()
        total_time+=time_assingment_e-time_assingment_s
        #计算初始化的时间
        #time computing for initailizing
        time_this_round_max=0
        clientid_with_max_time=''
        #计算此轮训练最长的单个时间 calculate the longest training time for this global epoch
        all_diff = []
        all_last_layer_tensor=[]
        total_encription_time=0
        total_decription_time=0
        # 遍历客户端，每个客户端本地训练模型
        # local train one by one, only add the longest time taking by clients to the total time
        for c in candidates:
            torch.cuda.empty_cache()
            time_c_s=time.time()
            #这个是单个客户端所需时间

            #if qb-strategy is deployed, then client will return more imformation about time taking by de/encryption
            if q==0:
                diff , last_layer_name= c.local_train(server.global_model,poison_trigger,first_round)
            else:
                diff, last_layer_name ,total_time_for_this_client ,total_de_time= c.local_train(server.global_model, poison_trigger,first_round)


            #如若要使用krum，则将所有的字典先存起来
            #save all the diff from clients in order to plant Krum

            #计算加解密时间 calculate the time for en/decryption for each global epoch
            if q!=0:
                total_encription_time+=total_time_for_this_client
                total_decription_time+=total_de_time
                avg_encription_time = total_encription_time / no_models
                avg_decription_time = total_decription_time / no_models

            all_diff.append(diff)
            time_c_e=time.time()
            time_takes_for_this_client=time_c_e-time_c_s
            print(",总时间:" + str(int(time_takes_for_this_client))+'seconds')
            if time_takes_for_this_client>time_this_round_max:
                time_this_round_max=time_takes_for_this_client
                clientid_with_max_time=str(c.client_id)
        if q!=0:
            print(model_name,'模型这轮加密的平均时间是',avg_encription_time,' 解密时间是',avg_decription_time)
            ten_round_en_time+=avg_encription_time
            ten_round_de_time+=avg_decription_time
            if round_no%10==0:
                print('十轮的平均加密时间是：',ten_round_en_time/10)
            if (round_no-1)%10==0:
                print('十轮的平均解密时间是：', ten_round_de_time/10)
        for diff in all_diff:
            a=diff[last_layer_name]*100
            #*100方便计算
            #mul 100 is for calculating
            all_last_layer_tensor.append(a)
        if krum>0:
            dist_matrix = np.zeros((no_models,no_models))
            dist_per_client=[]
            for i in range(no_models):
                for j in range(i+1,no_models):
                    dist_matrix[i,j]=torch.pairwise_distance(all_last_layer_tensor[i],all_last_layer_tensor[j])
                    dist_matrix[j,i]=dist_matrix[i,j]
            for m in range(no_models):
                total_dist=0
                for n in range(no_models):
                    total_dist+=dist_matrix[m,n]
                dist_per_client.append(total_dist)
            krum_poison_id=[]
            #挑出krum client个有毒客户端
            #The difference between the last layer vectors of the model returned by all clients is
            #computed in pairs, and based on these results, the sum of the distances of each vector to
            #several other vectors, the largest of which krum identifies as toxic

            for i in range(krum_client):
                current_maxium=0
                for j in range(no_models):
                    if dist_per_client[j]>dist_per_client[current_maxium]:
                        current_maxium=j
                dist_per_client[current_maxium]=0
                krum_poison_id.append(current_maxium)
            for i in krum_poison_id:
                del all_diff[i]

        for diff in all_diff:
            for name, params in server.global_model.state_dict().items():
                # diff[name].to(device)
                # 根据客户端的参数差值字典更新总体权重
                weight_accumulator[name] = weight_accumulator[name].cuda()
                # 问题problem：这个weight_accumulator变量不在gpu上，会导致报错
                diff[name]=diff[name].long()
                weight_accumulator[name].add_(diff[name])
        print("聚合完成 Done aggregate")
        total_time+=time_this_round_max
        print('此轮最长时间为'+str(int(time_this_round_max))
              +'秒, 客户端为'+clientid_with_max_time)

        time_aggregate_s=time.time()
        first_round = 0
        # 模型参数聚合
        server.model_aggregate(weight_accumulator)

        # 模型评估
        acc, loss = server.model_eval()

        time_aggregate_e=time.time()
        time_aggregate=time_aggregate_e-time_aggregate_s
        total_time+=time_aggregate
        print("Epoch %d, acc: %f, loss: %f, total time :%d\n" % (e, acc, loss,int(total_time)))
        #聚合的时间
        all_data=[acc,loss,int(total_time),clientid_with_max_time,int(time_this_round_max)]
        ws.append(all_data)
    if poison==0:
        wb_path='./result/solo-'+model_name  +'-'+current_al+"-"+str(global_epoches)+"epoches.xlsx"
    else:
        if krum==0:
            wb_path = ('./result/solo-' + model_name + '-' + current_al
                       + "-" + str(global_epoches) + "epoches-"+str(poison)+"poisonclient.xlsx")
        else:
            wb_path = ('./result/solo-' + model_name + '-' + current_al
                       + "-" + str(global_epoches) + "epoches-" + str(poison) + "poisonclient-krum"+str(krum_client)+".xlsx")
    wb.save(wb_path)
    model_path = './result/solo-'  +model_name  +'-'+current_al +'.pt'
    torch.save(server.global_model, model_path)

torch.cuda.empty_cache()
