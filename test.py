import argparse, json
import datetime
import os
import logging
import torch, random
import matplotlib.pyplot as plt
import xlwt
from server import *
from client import *
import models, datasets

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Federated Learning')
    parser.add_argument('-c', '--conf', dest='conf')
    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        conf = json.load(f)

    train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

    server = Server(conf, eval_datasets)
    clients = []

    speed = [20, 21, 22, 24, 70]
    # all_range = list(range(len(train_datasets)))
    # datalens = [all_range[0:6250],all_range[6250:25000],all_range[25000:31250],all_range[31250:50000]]
    ids = ['A', 'B', 'C', 'D', 'E']
    for c in range(conf["no_models"]):
        ###暂时setting 速度
        clients.append(Client(conf, server.global_model, train_datasets, eval_datasets, speed[c], ids[c], c))

    print("\n\n")

    blocknum = 0
    # 以满足预设acc为条件
    #g_acc = 0
    while blocknum in range(20):

        weight_accumulator = {}
        for name, params in server.global_model.state_dict().items():
            weight_accumulator[name] = torch.zeros_like(params)

        # 工人节点计时
        for c in clients:
            c.runLocalTime()
        # 判断满足速度后更新本地模型
        for c in clients:
            if c.localtime == c.sp:
                #diff = c.local_train()
                # 评价本地
                #acc, loss = c.model_eval()
                #print("Epoch %d client_%s, acc: %f, loss: %f\n" % (server.global_T, c.id, acc, loss))
                # 记入添加本地精确度 & 版本
                #c.up_acc.append(acc)
                c.up_v.append(c.local_v)
                # 将已完成的工人上传添加到block中
                server.block[server.global_T].append(c.id)
                # 将模型参数更入weight_accumulator
                #for name, params in server.global_model.state_dict().items():
                    #weight_accumulator[name].add_(diff[name])

            if len(server.block[blocknum]) == conf["blocksize"]:

                print(server.block)
                server.block.append([])

                # staleness
                for c in clients:
                    if c.id in server.block[blocknum]:
                        s = 1 / (server.global_T - c.local_v + 1)
                        c.curr_T.append(server.global_T)
                        c.staleness_set.append(s)
                        print("client %s staleness is %f" % (c.id, s))

                # 平均weight_accumulator中参数
                #server.model_aggregate(weight_accumulator)
                server.global_T += 1
                ###替换version
                for c in clients:
                    if c.id in server.block[blocknum]:
                        c.local_v = server.global_T
                        #c.copy_param(server.global_model)
                blocknum += 1

                # 评价全局
                #g_acc, g_loss = server.model_eval()
                #print("global_T %d,global_acc: %f, loss: %f\n" % (server.global_T, g_acc, g_loss))
                ##reset工人时间
                for c in clients:
                    if c.localtime >= c.sp:
                        c.resetLocalTime()
