import torch
from torch import nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 定义之前训练的网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 加载训练好的模型
model = Net()
checkpoint = torch.load('mnist_model_TrainAcc0.9_TestAcc0.9.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # transforms.Lambda(lambda x: 1.0 - x),  # 反转颜色  因为我们的图片是白底黑字
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载图像并进行预处理
img = Image.open('手写数字_推理.jpg')

# 检查图像模式并转换
if img.mode == 'RGBA':
    img = img.convert('RGB')

img = ImageOps.invert(img)  # 反转图像颜色
img_transformed = transform(img)
img_transformed = img_transformed.view(1, 28 * 28)  # 展平图像以匹配模型输入

# 绘制预处理后的图像
plt.imshow(img_transformed.view(28, 28), cmap='gray')
plt.title('Preprocessed Image')
plt.show()

# 推理
with torch.no_grad():
    output = model(img_transformed)
    _, predicted = torch.max(output.data, 1)
    print(f'Predicted digit: {predicted.item()}')



#### 黑底白字的图片  就用下面这个   mnist数据集就是黑底白字
# import torch
# from torch import nn
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
#
# # 定义之前训练的网络结构
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 256)
#         self.fc2 = nn.Linear(256, 64)
#         self.fc3 = nn.Linear(64, 10)
#
#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# # 加载训练好的模型
# model = Net()
# checkpoint = torch.load('mnist_model_TrainAcc0.9_TestAcc0.9.pth', map_location=torch.device('cpu'))
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()
#
# # 图像预处理
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
#
# # 加载图像并进行预处理
# img = Image.open('num_01.jpg')
# img_transformed = transform(img)
# img_transformed = img_transformed.view(1, 28 * 28)  # 展平图像以匹配模型输入
#
# # 绘制预处理后的图像
# plt.imshow(img_transformed.view(28, 28), cmap='gray')
# plt.title('Preprocessed Image')
# plt.show()
#
# # 推理
# with torch.no_grad():
#     output = model(img_transformed)
#     _, predicted = torch.max(output.data, 1)
#     print(f'Predicted digit: {predicted.item()}')
