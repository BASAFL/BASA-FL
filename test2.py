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

def upload(c,client,server,weight_accumulator,blocknum):
	if c.localtime == c.sp:
		if (server.global_T - c.local_v) <= 2:
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
		else:
			#c.copy_param(server.global_model)
			c.local_v = server.global_T
			c.resetLocalTime()
			c.malicious.append(server.global_T)
			#weight_accumulator, blocknum = upload(c, clients,server, weight_accumulator, blocknum)
	weight_accumulator_s, blocknum_s = update(blocknum, client, server,weight_accumulator)
	return weight_accumulator_s,blocknum_s

def update(blocknum,client,server,weight_accumulator):
	if len(server.block[blocknum]) == conf["blocksize"]:
		# staleness
		for c in client:
			if c.id in server.block[blocknum]:
				s = 1 / (server.global_T - c.local_v + 1)
				c.curr_T.append(server.global_T)
				c.staleness_set.append(s)
				print("client %s staleness is %f" % (c.id, s))
		print(server.block)
		server.block.append([])
		blocknum = fedavg(weight_accumulator,server,blocknum)
	return weight_accumulator,blocknum

def fedavg(weight_accumulator,server,blocknum):
	# 平均weight_accumulator中参数
	#server.model_aggregate(weight_accumulator)
	server.global_T += 1
	###替换version
	for c in clients:
		if c.id in server.block[blocknum]:
			c.local_v = server.global_T
			#c.copy_param(server.global_model)
	blocknum += 1
	#global g_acc
	# 评价全局
	#g_acc, g_loss = server.model_eval()
	#server.global_acc.append(g_acc)
	#server.g_T.append(server.global_T)
	#print("global_T %d,global_acc: %f, loss: %f\n" % (server.global_T, g_acc, g_loss))
	##reset工人时间
	for c in clients:
		if c.localtime >= c.sp:
			c.resetLocalTime()
	return blocknum

if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	

	with open(args.conf, 'r') as f:
		conf = json.load(f)	
	
	
	train_datasets, eval_datasets = datasets.get_dataset("./data/", conf["type"])

	server = Server(conf, eval_datasets)
	clients = []

	speed = [15, 25, 35, 50]
	#all_range = list(range(len(train_datasets)))
	#datalens = [all_range[0:6250],all_range[6250:25000],all_range[25000:31250],all_range[31250:50000]]
	ids = ['A','B','C','D']
	for c in range(conf["no_models"]):
		clients.append(Client(conf, server.global_model, train_datasets,eval_datasets, speed[c], ids[c],c))
	print("\n\n")

	blocknums = 0
	#以满足预设acc为条件
	g_acc = 0
	while server.global_T <= 20:

		weight_accumulators = {}
		for name, params in server.global_model.state_dict().items():
			weight_accumulators[name] = torch.zeros_like(params)

		#工人节点计时
		for c in clients:
			c.runLocalTime()
		#判断满足速度后更新本地模型
		for c in clients:
			weight_accumulators,blocknums = upload(c,clients,server, weight_accumulators,blocknums)

print("exceeds staleness threshold:")
for c in clients:
	print("client_ %s" % c.id, end='\t' )
	print(c.malicious)

#########################################
node1 = {"code": "A", "up_acc": [], "up_Local_v": [], "up_global_T": [], "staleness": []}
node2 = {"code": "B", "up_acc": [], "up_Local_v": [], "up_global_T": [], "staleness": []}
node3 = {"code": "C", "up_acc": [], "up_Local_v": [], "up_global_T": [], "staleness": []}
node4 = {"code": "D", "up_acc": [], "up_Local_v": [], "up_global_T": [], "staleness": []}
#node5 = {"code": "E", "up_acc": [], "up_Local_v": [], "up_global_T": [], "staleness": []}
nodeList = [node1,node2,node3,node4,node5]
i = 0
for c in clients:
	if c.client_id == i:
		nodeList[i]["up_acc"]=c.up_acc
		nodeList[i]["up_Local_v"] = c.up_v
		nodeList[i]["up_global_T"] = c.curr_T
		nodeList[i]["staleness"] = c.staleness_set
		i += 1

# graph
# 0.准备数据
x_node1 = node1["up_global_T"]
x_node2 = node2["up_global_T"]
x_node3 = node3["up_global_T"]
x_node4 = node4["up_global_T"]
#x_node5 = node5["up_global_T"]

y_node1 = node1["up_acc"]
y_node2 = node2["up_acc"]
y_node3 = node3["up_acc"]
y_node4 = node4["up_acc"]
#y_node5 = node5["up_acc"]
# 1.创建画布
plt.figure(figsize=(20, 8), dpi=100)  # 画布大小，dpi：清晰度

# 2.绘制图像（折线图）
plt.plot(x_node1, y_node1, label='worker_A', color='r', markeredgecolor='r', marker='^', markersize='10', markeredgewidth=5)
# 设置线的风格，颜色，添加图例
plt.plot(x_node2, y_node2, color='g', linestyle='--', label='worker_B', marker='o', markeredgecolor='g', markersize='10', markeredgewidth=5)
plt.plot(x_node3, y_node3, color='b', linestyle='-', label='worker_C',marker='x', markeredgecolor='b', markersize='10', markeredgewidth=5)
plt.plot(x_node4, y_node4, color='y', linestyle='-', label='worker_D',marker='s', markeredgecolor='y', markersize='10', markeredgewidth=5)
#plt.plot(x_node5, y_node5, color='m', linestyle='-', label='worker_E',marker='*', markeredgecolor='m', markersize='10', markeredgewidth=5)

# 2.2 添加网格显示
plt.grid(True, linestyle='--', alpha=0.5)

# 2.3 添加描述信息
plt.xlabel('Global Iteration', fontsize=30)
plt.ylabel('accuracy', fontsize=30)
#plt.title('1123', fontsize=20)

# 2.4 图像保存（放在show前面，show()会释放figure资源，如果显示图像之后保存图片只能保存空图片）
#plt.savefig('./test.png')

# 2.5 显示图例
plt.legend(loc="best",fontsize=25)  # 0
plt.title('The group—β',fontsize=25)

plt.savefig('./jubu_staleness.png')
# 3.图像显示
plt.show()
###################################################################
# graph
# 0.准备数据
x_global = server.g_T

y_global = server.global_acc

# 1.创建画布
plt.figure(figsize=(20, 8), dpi=100)  # 画布大小，dpi：清晰度

# 2.绘制图像（折线图）
plt.plot(x_global, y_global, label='global_accuracy', color='r', markeredgecolor='r', marker='^', markersize='10', markeredgewidth=5)
# 设置线的风格，颜色，添加图例

# 2.2 添加网格显示
plt.grid(True, linestyle='--', alpha=0.5)

# 2.3 添加描述信息
plt.xlabel('Global Iteration', fontsize=30)
plt.ylabel('accuracy', fontsize=30)
#plt.title('1123', fontsize=20)

# 2.4 图像保存（放在show前面，show()会释放figure资源，如果显示图像之后保存图片只能保存空图片）
#plt.savefig('./test.png')

# 2.5 显示图例
plt.legend(loc="best",fontsize=25)  # 0
plt.title('The global iteration ',fontsize=25)

plt.savefig('./quanju_staleness.png')
# 3.图像显示
plt.show()

#####################################################################
# re-Rank
for nodeName in nodeList:
    TT = 0
    for t in nodeName["up_Local_v"]:
        if t != TT:
            nodeName["up_Local_v"].insert(TT, 0)
            nodeName["up_acc"].insert(TT, 0)
        TT += 1
for nodeName in nodeList:
    # print(nodeName["up_Local_v"])
    print(nodeName["up_acc"])

# store acc_weight with version_v
workbook = xlwt.Workbook(encoding = 'utf-8')        #设置一个workbook，其编码是utf-8
worksheet = workbook.add_sheet("test_sheet")        #新增一个sheet

for i in range(len(nodeList)):
    worksheet.write(i, 0, label=nodeList[i]["code"])

j = 0
for nodeName in nodeList:
    for i in range(len(nodeName["up_acc"])):                    #循环将a和b列表的数据插入至excel
        worksheet.write(j, i+1, label=nodeName["up_acc"][i])
    j += 1

# store staleness with version_v
j = 0
for nodeName in nodeList:
    for i in range(len(nodeName["staleness"])):                    #循环将a和b列表的数据插入至excel
        worksheet.write(j, i + len(server.block) + 2, label=nodeName["staleness"][i])
    j += 1

workbook.save(r"C:\Users\user\Downloads\03\staleness.xls") #这里save需要特别注意，文件格式只能是xls，不能是xlsx，不然会报错
print(len(nodeList))