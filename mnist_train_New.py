import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision
from torchsummary import summary  # 输出网络模型结构的相关包
from matplotlib import pyplot as plt

from utils import plot_image, plot_curve, one_hot  # 导入自定义的一些工具函数，如plot_image, plot_curve, one_hot

import os  # 导入os库，用于操作系统相关的功能

os.environ[
    'KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 设置一个环境变量，防止出现重复的库的错误

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载MNIST数据集
batch_size = 512
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

x, y = next(iter(train_loader))  # 使用next(iter())方法从训练数据加载器中获取下一个批次的数据，包括输入x和标签y
print(x.shape, y.shape, x.min(), x.max())  # 打印x和y的形状、x的最小值和最大值
plot_image(x, y, 'image sample')


# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().to(device)
################ 输出网络结构
# summary(net, input_size=(28*28,)) # 调用summary函数，输入模型和输入大小
###################优化器
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)  # 创建一个 梯度下降SGD 优化器，用来更新 net 模型的参数
# optimizer = optim.Adam(net.parameters(), lr=0.0001)  # 基于自适应学习率的优化器，它可以根据每个参数的梯度历史动态调整学习率，从而加速收敛


# 初始化准确率和损失列表
train_acc_list = []
test_acc_list = []
train_loss_list = []

# 训练和测试模型
train_loss = []
for epoch in range(2):  # epoch设置为（）中的数
    net.train()
    train_correct = 0
    train_total = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        x = x.view(x.size(0), 28 * 28)
        out = net(x)

        # loss_func=torch.nn.MultiLabelSoftMarginLoss()
        # loss = loss_func(out, y_onehot)

        loss_func = torch.nn.CrossEntropyLoss()
        loss = loss_func(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        # if batch_idx % 10==0:
        #    print('epoch:',epoch,'迭代次数:',batch_idx,'loss:',loss.item())

        # 计算训练准确率
        pred = out.argmax(dim=1)
        train_correct += pred.eq(y).sum().item()
        train_total += y.size(0)

    train_acc = train_correct / train_total
    train_acc_list.append(train_acc)

    # 测试模型
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), 28 * 28)
            out = net(x)
            pred = out.argmax(dim=1)
            test_correct += pred.eq(y).sum().item()
            test_total += y.size(0)

    test_acc = test_correct / test_total
    test_acc_list.append(test_acc)

    print(
        f'Epoch: {epoch}, Train_loss:{loss.item():.4f},Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}')

# 输出loss变化曲线图  # plot_curve(train_loss)
plt.plot(train_loss, '.-', label='Train Accuracy')
plt.xlabel("Iterations")  # 迭代次数
plt.ylabel("Loss")
plt.title("Loss curve")
plt.ylim(0, 1.0)
plt.legend(loc='best')  #legend的位置 如 upper right
plt.show()
# 绘制训练和测试准确率曲线
plt.plot(train_acc_list, 'o-', label='Train Accuracy')
plt.plot(test_acc_list, 's--', label='Test Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Epochs")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

# 随机选择9个测试样本并绘制预测结果
x, y = next(iter(test_loader))
x, y = x.to(device), y.to(device)
x = x.view(x.size(0), 28 * 28)  # 展平图像以匹配网络输入
out = net(x)
pred = out.argmax(dim=1)
# 将x重新塑形为图像的原始形状 (batch_size, 1, 28, 28)
x = x.view(x.size(0), 1, 28, 28)
# 选择9个图像进行展示
x_to_show = x[:9].cpu()
pred_to_show = pred[:9].cpu()
# 调用plot_image函数绘制图像和预测
plot_image(x_to_show, pred_to_show, 'Test Sample')

# 保存模型
torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'Testacc': test_acc,
            'TrainAcc': train_acc,
            'Batchsize': batch_size,
            }, f"mnist_model_TrainAcc{train_acc:.1f}_TestAcc{test_acc:.1f}.pth")

# torch.save({
#     'epoch': epoch,
#     'model_state_dict': net.state_dict()
# }, 'mnist_model.pth')


