import torch.nn as nn
import torch
import dataset
import configs
import itertools
import numpy as np
from models import CNN
from dataset import get_dataloader2

class Trainer:
    def __init__(self,models,modelsY,optimizers,critrions,train_loader,test_loader):
        self.models = models
        self.modelsY = modelsY
        self.optimizers = optimizers
        self.critrions = critrions
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.pretrain = True
        self.models_grad,self.modelsY_grad = [[] for n in range(configs.NMODELS)],[[] for n in range(configs.NMODELS)]
        self.train_accuracy,self.test_accuracy = [],[]
        self.train_loss,self.test_loss = [],[]

    def accurate(self,pre_data, labels):
        pre, target = pre_data.cpu(), labels.cpu()
        pred = torch.max(pre.data, 1)[1]
        rights = torch.eq(pred, target.data.view_as(pred)).sum()
        return 100 * rights / len(target)

    def update_grad(self,models_grad: list):
        # u 均值为0 方差为1 的随机数
        noise = torch.Tensor([np.random.laplace(loc=0, scale=1)]).float().cuda()
        for model_grad in models_grad:
            # 将那两个f(x)分步计算，同时初始化
            model_layer_grad_1, model_layer_grad_2 = model_grad[0], model_grad[0]
            model_layer_grad_2s = []
            for model1_layer_grad in model_grad:
                model_layer_grad_2 = model1_layer_grad - configs.Y*noise
                model_layer_grad_2 *= noise
                model_layer_grad_2s.append(model_layer_grad_2)

            for i,model2_layer_grad in enumerate(model_grad):
                model2_layer_grad += configs.Y*noise
                model2_layer_grad += model_layer_grad_2s[i]

    def update_xy_grad(self,index: int, models_grad: list, modelsY_grad: list, configure):
        """
        对模型的梯度进行更新
        :param index: 更新梯度的模型的索引
        :param models_grad: 保存的所有模型的梯度
        :param modelsY_grad: 保存的辅助模型的梯度
        :param config: 调用权重矩阵的配置类
        :return:
        """
        # 保存的模型梯度Xi
        changed_grad = models_grad[index]
        ar = configure.flush_weight()['ar'][index]
        ac = configure.flush_weight()['ac'][index]
        index_models_grad = models_grad[index]
        index_modelYs_grad = modelsY_grad[index]
        # 直接对Xi做运算让其本身变为Xi+1
        for node, rate in ar[1:]:
            index_node_models_grad =models_grad[node]
            for model_grad,model_node_grad,modelY_node_grad in itertools.zip_longest(index_models_grad,
                                                                                     index_node_models_grad,
                                                                                      index_modelYs_grad ):
            # 与权重矩阵相乘求和
                model_grad += rate * model_node_grad
                model_grad -= configs.E *modelY_node_grad
                model_grad += configs.B * model_node_grad
        # 计算Yi+1
        for node, rate in ar:
            for model_gradi, modelY_grad,model_node_grad2 in itertools.zip_longest(changed_grad, modelsY_grad[index],models_grad[node]):

            # 暂时保存Yi
                modelY_grad_temp = modelY_grad
            # model_gradi 是Xi
                modelY_grad = model_gradi

                modelY_grad -= rate * model_node_grad2
                modelY_grad += configs.E * modelY_grad_temp
        for node, rate in ac:
            for modelY_grad,modelY_node_grad2 in itertools.zip_longest(modelsY_grad[index],models_grad[node]):
                modelY_grad += rate * modelY_node_grad2



    def get_grad(self,index):
        for name in self.models[index].parameters():
            self.models_grad[index].append(name.grad)
            if self.pretrain:
                self.modelsY_grad[index].append(name.grad)

    def put_grad(self,index):
        for name,new_grad in itertools.zip_longest(self.models[index].parameters(),self.models_grad[index]):
            name.grad = new_grad

    def _run_batch(self,index,data,label,batch):
        if batch%200 == 0:
            print("这是第{}个模型：{}批次训练".format(index,batch))
        if  self.pretrain:
            pre = self.models[index](data)
            loss = self.critrions[index](pre,label)
            self.optimizers[index].zero_grad()
            loss.backward()
            self.optimizers[index].step()
            self.get_grad(index)
            if index == 9:
                self.pretrain = False
                print("=======预训练结束=======")
        else:
            #self.get_grad(index)
            self.update_grad(self.models_grad)
            self.update_xy_grad(index,
                                models_grad=self.models_grad,
                                modelsY_grad=self.modelsY_grad,
                                configure=configs.configure)
            pre = self.models[index](data)
            loss = self.critrions[index](pre, label)
            self.optimizers[index].zero_grad()
            loss.backward()
            self.optimizers[index].step()
            self.put_grad(index)
            self.train_accuracy.append(self.accurate(pre,label))
            self.train_loss.append(loss.item())


    def _run_epoch(self,epoch):
        print("这是第{}轮训练".format(epoch))
        for batch,(data,label) in enumerate(self.train_loader):
            data,label = data.cuda(),label.cuda()
            for index in range(len(self.models)):
                self._run_batch(index,data,label,batch)
                # if len(self.train_accuracy) != 0:
                #     msg = "Epoch:{} Model:{} train_accuracy:{}% train_loss:{}"
                #     train_accuracy = sum(self.train_accuracy)/len(self.train_accuracy)
                #     train_loss = sum(self.train_loss) / len(self.train_loss)
                #     print(msg.format(epoch,index,train_accuracy,train_loss))


    def train(self):
        if self.pretrain:
            print("=======预训练开始=======")
        for epoch in range(configs.EPOECHS):
            self._run_epoch(epoch=epoch)
        for index,model in enumerate(self.models):
            self.show_update(index,model)
    def show_update(self,index, model):
        """
        展示最终优化结果
        :param index: 模型索引
        :param model: 模型
        :return:
        """
        criterion = self.critrions[index]
        with torch.no_grad():
            for data, label in self.test_loader:
                data, label = data.cuda(), label.cuda()
                res = model(data)
                loss = criterion(res, label)
                res, label = res.cpu(), label.cpu()
                pred = torch.max(res.data, 1)[1]
                rights = torch.eq(pred, label.data.view_as(pred)).sum()
                self.test_loss.append(loss.item())
                self.test_accuracy.append(100 * rights / len(label))
        test_loss = sum(self.test_loss) / len(self.test_loss)
        test_accuracy = sum(self.test_accuracy) / len(self.test_accuracy)
        print("经过参数更新后,##模型{}的 loss: {:.3f} accuracy: {:.3f}%".format(index, test_loss, test_accuracy))

nmodels, nmodelsY = [CNN().cuda() for n in range(configs.NMODELS)], \
                    [CNN().cuda() for n in range(configs.NMODELS)]
critrions = [nn.CrossEntropyLoss() for n in range(configs.NMODELS)]
optimizers = [torch.optim.Adam(model.parameters()) for model in nmodels]
train_loader,test_loader = get_dataloader2()

trainer = Trainer(models=nmodels,
                  modelsY=nmodelsY,
                  optimizers=optimizers,
                  critrions=critrions,
                  train_loader=train_loader,
                  test_loader=test_loader)

trainer.train()