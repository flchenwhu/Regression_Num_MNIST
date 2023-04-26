import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
import tensorwatch as tw
from torchvision import models
from torchsummary import summary  #输出网络模型结构的相关包
from matplotlib import pyplot as plt # 导入matplotlib库，用于绘图

from utils import plot_image,plot_curve,one_hot # 导入自定义的一些工具函数，如plot_image, plot_curve, one_hot

import os  # 导入os库，用于操作系统相关的功能

os.environ[
    'KMP_DUPLICATE_LIB_OK']='TRUE'   # 设置一个环境变量，防止出现重复的库的错误

batch_size=512  # 设置每个批次的数据量为512


# 创建一个训练数据加载器，使用torchvision.datasets.MNIST方法下载并加载MNIST手写数字数据集
# 使用torchvision.transforms.Compose方法对数据进行预处理，包括转换为张量和归一化 使用torch.utils.data.DataLoader方法将数据封装成批次，并打乱顺序
train_loader=torch.utils.data.DataLoader(
    
    torchvision.datasets.MNIST('mnist_data',train=True,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081,))
                               ])),
    batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/',train=False,download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,),(0.3081,))
                               ])),
    batch_size=batch_size,shuffle=False)

x,y=next(iter(train_loader))  # 使用next(iter())方法从训练数据加载器中获取下一个批次的数据，包括输入x和标签y
print(x.shape,y.shape,x.min(),x.max())  # 打印x和y的形状、x的最小值和最大值

plot_image(x,y,'image sample')

# 移动到GPU上
x=x.cuda()  # （自己添加）
y=y.cuda()  # （自己添加）

class Net(nn.Module):
# 定义神经网络模型类Net继承nn.Module类

    def __init__(self):
    # 定义模型的初始化方法

        super(Net,self).__init__()
        # 继承并执行父类（nn.Module）的初始化方法
        # xw+b
        self.fc1=nn.Linear(28 * 28,256)
        # 第一层全连接层，输入特征向量28*28，输出特征向量256
        self.fc2=nn.Linear(256,64)
        # 第二层全连接层，输入特征向量256，输出特征向量64
        self.fc3=nn.Linear(64,10)
        # 第三层全连接层，输入特征向量64，输出特征向量10

    def forward(self,x):
    # 定义前向传播计算过程的方法forward，输入参数x是模型输入

        # x: [b, 1, 28, 28]
        # h1 = relu(xw1+b1)
        x=F.relu(self.fc1(x))
        # 将第一层的输出传递到第二层，并应用ReLU激活函数 进行非线性变换
        # h2 = relu(h1w2+b2)
        x=F.relu(self.fc2(x))
        # h1作为输入通过第二层全连接层计算得到h2，并应用激活函数relu进行非线性变换
        # h3 = h2w3+b3
        x=self.fc3(x)
        # h2作为输入通过第三层全连接层计算得到输出结果h3
        return x

net=Net()
# 移动模型到GPU上    # print(next(net.parameters()).device)  # 返回cuda值
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # （自己添加）
net=net.to(device)  # 网络加载到GPU运行（自己添加）

################ 输出网络结构
summary(net, input_size=(28*28,)) # 调用summary函数，输入模型和输入大小


###################优化器
optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9) # 创建一个 梯度下降SGD 优化器，用来更新 net 模型的参数
# optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 基于自适应学习率的优化器，它可以根据每个参数的梯度历史动态调整学习率，从而加速收敛

#####################训练
train_loss=[]
for epoch in range(3):  # epoch是所有数据都放入模型训练的次数

    for batch_idx,(x,y) in enumerate(train_loader):  # batch_idx即iteration，表示迭代次数

        # x: [b, 1, 28, 28], y: [512]
        # [b, 1, 28, 28] => [b, 784]
        x=x.cuda()  
        y=y.cuda() 
        x=x.view(x.size(0),28 * 28)
        # => [b, 10]
        out=net(x)
        y_onehot=one_hot(y)
        # loss = mse(out, y_onehot)
        # loss=F.mse_loss(out,y_onehot)  # 均方误差
        loss_func=torch.nn.CrossEntropyLoss()  # 直接使用pytorch内置的交叉熵损失函数 
        loss=loss_func(out,y_onehot)  # 直接使用pytorch内置的交叉熵损失函数 

        optimizer.zero_grad()
        loss.backward()
        # w' = w - lr*grad
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx % 10==0:
            print('epoch:',epoch,'迭代次数:',batch_idx,'loss:',loss.item())
###############输出训练准确度###############
    train_correct = 0
    total = 0
    for data in train_loader:
        x, y = data
        x=x.cuda()  
        y=y.cuda()  
        x=x.view(x.size(0),28 * 28)
        out = net(x)
        pred=out.argmax(dim=1)
        correct=pred.eq(y).sum().float().item()
        train_correct +=correct
        total += y.size(0)
        train_acc=train_correct / total
    print('epoch:',epoch,'迭代次数:',batch_idx,'loss:',loss.item(),'Train accuracy:',train_acc)


plot_curve(train_loss)  #输出loss变化曲线图
#plt.show()   # 显示视图

######### 测试（利用测试数据）
total_correct=0
for x,y in test_loader:
    x=x.cuda() 
    y=y.cuda() 
    x=x.view(x.size(0),28 * 28)
    out=net(x)
    # out: [b, 10] => pred: [b]
    pred=out.argmax(dim=1)
    correct=pred.eq(y).sum().float().item()
    total_correct+=correct

total_num=len(test_loader.dataset)
acc=total_correct / total_num
print('Test accuracy:',acc)

x,y=next(iter(test_loader))
x=x.cuda()  # （自己添加）
# y=y.cuda()  # （自己添加）
out=net(x.view(x.size(0),28 * 28))
pred=out.argmax(dim=1)
pred=pred.cuda().data.cpu()  # 自己添加
x=x.cuda().data.cpu()  # 自己添加
plot_image(x,pred,'test')
plt.show()

####################模型保存 保存整个网络
# torch.save(net, './Regression_model_.pt')
torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Testacc': acc,
            'TrainAcc': train_acc,
            'Batchsize': batch_size,
            }, f"model_Batchsize{batch_size}_epoch{epoch}_TrainAcc{train_acc:.3f}_TestAcc{acc:.3f}.pth")
# #读取
# use_model = torch.load('./Train_model.pt')
