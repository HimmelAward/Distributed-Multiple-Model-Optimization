import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch
import config

#自定义反向传播函数
class CustomFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,z, w, b):
        #ctx 反向传播可能用到的上下文
        # 保存输入张量和权重张量，以备反向传播时使用
        ctx.save_for_backward(z, w, b)
        # 计算输出张量
        #主要对这个修改！！！！！
        y = z.matmul(w) + b
        return y

    @staticmethod
    def backward(ctx, grad_output):
        # 从上下文中获取保存的张量
        x, w, b = ctx.saved_tensors
        # 计算反向传播的梯度
        #主要对这个修改！！！！
        grad_x = grad_output.mm(w.t())
        grad_w = x.t().mm(grad_output)
        grad_b = grad_output.sum(0)
        return grad_x, grad_w, grad_b
#神经网络模型
class MNN(nn.Module):
    def __init__(self):
        super(MNN, self).__init__()
        #self.conv1 =nn.Conv2d(1,16,5,1,2)
        #self.conv2 = nn.Conv2d(16,32,5,1,2)
        #self.linear = nn.Linear(32*7*7,10)
        self.linear = nn.Linear(28*28)
        self.custom = CustomFunction.apply

    def forward(self,x):
        #正向传播
        #自定义反向传播过程
        x = self.custom(x,self.conv1.weight,self.conv1.bias)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(2)
        x = self.custom(x, self.conv2.weight, self.conv2.bias)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(2)
        x = x.view(x.size(0),-1)
        out = self.linear(x)
        x = self.custom(x, self.linear.weight, self.linear.bias)
        return out

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2),        #(16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#(16,14,14)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.linear = nn.Linear(32*7*7,10)

    def forward(self,x):
        x = x.cuda()#电脑问题可以在其他电脑舍去
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        out = self.linear(x)
        return out


if "__name__" == "__main__":
    ...
    # mnn = MNN()
    # train_loader,test_loader = dataset.get_dataloader()
    # optimizer = torch.optim.Adam(mnn.parameters(),lr=1e-3)
    # critrion = torch.nn.CrossEntropyLoss()
    #
    # for train_data,train_label in train_loader:
    #     res = mnn(train_data)
    #     optimizer.zero_grad()
    #     loss = critrion(res,train_label)
    #     loss.backward()
    #     optimizer.step()
    #     print(loss.item())