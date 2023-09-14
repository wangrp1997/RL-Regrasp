import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
import re


float_pattern = r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+"

# # 定义神经网络模型（与之前的模型定义相同）
import torch
import torch.nn as nn


class CustomModel(nn.Module):
    def __init__(self, input_size_tactile, input_size_actuator, input_size_ratio):
        super(CustomModel, self).__init__()

        # 处理触觉数据的密集层
        self.fc_tactile = nn.Sequential(
            nn.Linear(input_size_tactile, 16),
            nn.ReLU(),
        )

        # 处理末端执行器数据的LSTM和池化层
        self.lstm_actuator = nn.LSTM(input_size=input_size_actuator, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2)
        self.pooling = nn.AdaptiveMaxPool1d(1)

        # 处理抓取比的密集层
        self.fc_ratio = nn.Sequential(
            nn.Linear(input_size_ratio, 16),
            nn.ReLU(),
        )

        # 两个额外的密集层
        self.fc1 = nn.Sequential(
            nn.Linear(16 + 32 + 16, 64),
            nn.ReLU(),
        )
        # Dropout层在全连接层之后
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)  # 输出抓取鲁棒性

    def forward(self, tactile_data, actuator_data, ratio_data):
        # 处理触觉数据
        tactile_output = self.fc_tactile(tactile_data)

        # 处理末端执行器数据
        actuator_output, _ = self.lstm_actuator(actuator_data)
        actuator_output = self.pooling(actuator_output.permute(0, 2, 1)).squeeze(2)

        # 处理抓取比
        ratio_output = self.fc_ratio(ratio_data)

        # 连接三个部分的输出
        combined_output = torch.cat((tactile_output, actuator_output, ratio_output), dim=1)

        # 通过两个额外的密集层生成最终输出
        final_output = self.fc1(combined_output)
        # Dropout层在全连接层之后
        final_output = self.dropout(final_output)
        final_output = self.fc2(final_output)

        return final_output


# 读取数据文件并筛选有效数据
def load_data(filename):
    import random
    random.seed(42)  # 这里的42是随机数种子的值，你可以选择任何整数作为种子值
    data = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip().split()
            temp = []
            ft_list = []
            for i in line[2:-2]:
                matches = re.findall(float_pattern, i)
                temp.append([float(match) for match in matches][0])
                if len(temp) == 6:
                    ft_list.append(temp.copy())
                    temp = []
            # 检查数据长度是否为32（六维力扭矩数据列表）
            if len(ft_list) == 32:
                # 提取触觉数据、力扭矩数据、抓取比例和标签
                tactile_data = np.array([float(line[0]), float(line[1])])
                force_torque_data = np.array(ft_list)
                grasp_ratio = np.array(float(line[-2]))
                label = float(line[-1])
                data.append((tactile_data, force_torque_data, grasp_ratio, label))
            # if len(data) == 1347:
            #     break
    return random.sample(data, 1347)



# 将数据划分为训练集和测试集
def split_data(data, test_size=0.2, random_state=42):
    # 将数据拆分为输入和标签
    X = [item[:3] for item in data]
    y = [item[3] for item in data]

    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)



    # 转换为PyTorch张量
    X_train_tensor=[]
    for x in X:
        item = [torch.FloatTensor(d) for d in x]
        X_train_tensor.append(item.copy())
    y_train = torch.FloatTensor(y)
    # X_test_tensor = []
    # for x_t in X_test:
    #     item = [torch.FloatTensor(d_t) for d_t in x_t]
    #     X_test_tensor.append(item.copy())
    # y_test = torch.FloatTensor(y_test)

    return X_train_tensor,y_train,
           # X_test_tensor,  y_test


# 创建数据加载器
def create_data_loader(X, y, batch_size=64):
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 读取数据并筛选有效数据
    data = load_data("data_lstm.txt")
    # print(len(data))
    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = split_data(data)
    X_train, y_train = split_data(data)
    print(len(X_train))
    print(len(y_train))
    # print("*"*100)
    # print(y_test)
    # # 创建数据加载器
    # train_loader = create_data_loader(X_train, y_train)
    # test_loader = create_data_loader(X_test, y_test)
    #
    # # 创建模型
    input_size_tactile = 2
    input_size_actuator = 32  # 由于力扭矩数据有32维
    input_size_ratio = 1
    model = CustomModel(input_size_tactile, input_size_actuator, input_size_ratio)

    # # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 500
    losses = []  # 用来记录每个epoch的损失值
    batch_size = 16
    batch_inputs = []
    batch_labels = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0  # 用于累积每个epoch的损失
        for inputs, labels in zip(X_train, y_train):
            batch_inputs.append(inputs)
            batch_labels.append(labels)
            if len(batch_inputs) == batch_size:
                optimizer.zero_grad()
                tactile_input = torch.stack([bi[0] for bi in batch_inputs])
                ft_input = torch.stack([bi[1] for bi in batch_inputs])
                ratio_input = torch.stack([bi[2] for bi in batch_inputs])
                label_ouput = torch.stack([bi for bi in batch_labels])
                outputs = model(tactile_input.view(-1,2), ft_input.view(-1,6,32), ratio_input.view(-1,1))
                # print(outputs)
                loss = criterion(outputs, label_ouput.view(-1,1))
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # 清空批次数据
                batch_inputs = []
                batch_labels = []

        # 处理最后一个不完整的批次
        if batch_inputs:
            optimizer.zero_grad()
            tactile_input = torch.stack([bi[0] for bi in batch_inputs])
            ft_input = torch.stack([bi[1] for bi in batch_inputs])
            ratio_input = torch.stack([bi[2] for bi in batch_inputs])
            label_ouput = torch.stack([bi for bi in batch_labels])
            outputs = model(tactile_input.view(-1, 2), ft_input.view(-1, 6, 32), ratio_input.view(-1, 1))
            loss = criterion(outputs, label_ouput.view(-1,1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss:.4f}")
    epochs = list(range(1, num_epochs + 1))

    torch.save(model.state_dict(), 'model-lstm-16-1347.pt')

    # 绘制损失曲线
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
    # # 在测试集上评估模型
    # model.eval()
    # with torch.no_grad():
    #     total_loss = 0
    #     for inputs, labels in zip(X_test,y_test):
    #         outputs = model(inputs[:][0].view(-1,2), inputs[:][1].view(-1,6,32), inputs[:][2].view(-1,1))
    #         loss = criterion(outputs, labels.view(-1,1))
    #         total_loss += loss.item()
    # #
    # avg_loss = total_loss / len(y_test)
    # print(f"平均测试损失: {avg_loss}")
