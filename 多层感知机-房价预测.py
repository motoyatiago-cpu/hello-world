import os
import hashlib    #你下载下来的文件是不是完整、正确的文件
import zipfile
import tarfile

import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader


# 1. 数据下载工具

DATA_HUB = {}

DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"

def download(name, cache_dir="../data"):
    """
    下载 DATA_HUB 中指定的数据文件 如果本地已经存在且 SHA-1 校验正确，则直接使用缓存
    """

    assert name in DATA_HUB, f"{name} 不存在于 DATA_HUB"

    url, sha1_hash = DATA_HUB[name]

    os.makedirs(cache_dir, exist_ok=True)

    # 根据 URL 得到文件名
    fname = os.path.join(cache_dir, url.split("/")[-1])

    # 如果文件已经存在，检查 SHA-1
    if os.path.exists(fname):

        sha1 = hashlib.sha1()

        with open(fname, "rb") as f:

            while True:

                data = f.read(1024 * 1024)

                if not data:
                    break

                sha1.update(data)

        # 文件正确，直接使用 如果本地文件和官方文件一致，就直接使用，不需要重新下载
        if sha1.hexdigest() == sha1_hash:
            print("使用本地缓存：", fname)
            return fname

        print("本地文件校验失败，重新下载")

    # 下载文件

    print("正在下载：", url)

    response = requests.get( url, stream=True,verify=True)

    response.raise_for_status()

    with open(fname, "wb") as f:

        for chunk in response.iter_content(chunk_size=1024 * 1024):

            if chunk:
                f.write(chunk)

    return fname


def download_extract(name, folder=None):
    """
    下载并解压 zip / tar 文件
    """

    fname = download(name)

    base_dir = os.path.dirname(fname)

    data_dir, ext = os.path.splitext(fname)

    if ext == ".zip":

        with zipfile.ZipFile(fname, "r") as fp:
            fp.extractall(base_dir)

    elif ext in (".tar", ".gz"):

        with tarfile.open(fname, "r") as fp:
            fp.extractall(base_dir)

    else:

        raise ValueError("只有 zip / tar / gz 文件可以解压")

    if folder:
        return os.path.join(base_dir, folder)

    return data_dir

# 2. 注册数据集

DATA_HUB["kaggle_house_train"] = (
    DATA_URL + "kaggle_house_pred_train.csv",
    "585e9cc93e70b39160e7921475f9bcd7d31219ce"
)

DATA_HUB["kaggle_house_test"] = (
    DATA_URL + "kaggle_house_pred_test.csv",
    "fa19780a7b011d9b009e8bff8e99922a8ee2eb90"
)

# 3. 读取数据

train_path = download("kaggle_house_train")
test_path = download("kaggle_house_test")

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)


print("\n================ 数据基本信息 ================")

print("训练集大小：", train_data.shape)
print("测试集大小：", test_data.shape)

print("\n训练集前4行：")
print(train_data.iloc[0:4,[0, 1, 2, 3, -3, -2, -1]])

# 4. 合并训练集和测试集特征

# train：Id | 特征1 | 特征2 | ... | 特征n | SalePrice
# test：Id | 特征1 | 特征2 | ... | 特征n
# train_data.iloc[:, 1:-1]去掉：第0列 Id 最后一列 SalePrice
# test_data.iloc[:, 1:] 去掉 Id 保留所有特征

all_features = pd.concat(
    (train_data.iloc[:, 1:-1],test_data.iloc[:, 1:]),axis=0)

print("\n合并后的特征数量：", all_features.shape)

# 5. 数值特征标准化

numeric_features = all_features.dtypes[ all_features.dtypes != "object"].index


all_features[numeric_features] = (
    all_features[numeric_features]
    .apply(
        lambda x: (x - x.mean()) / x.std()
    )
)

# 6. 填充缺失值

all_features[numeric_features] = (
    all_features[numeric_features]
    .fillna(0)
)

# 7. 类别特征独热编码

all_features = pd.get_dummies(all_features,dummy_na=True)

# True / False 转成 0 / 1
all_features = all_features.astype(np.float32)

print("预处理后的特征维度：", all_features.shape)

# 8. 转换为 PyTorch Tensor

n_train = train_data.shape[0]

train_features = torch.tensor(
    all_features.iloc[:n_train].values,
    dtype=torch.float32
)

test_features = torch.tensor(
    all_features.iloc[n_train:].values,
    dtype=torch.float32
)


# 房价标签
train_labels = torch.tensor(
    train_data["SalePrice"].values.reshape(-1, 1),
    dtype=torch.float32
)


print("\n================ Tensor 信息 ================")
print("train_features:", train_features.shape)
print("test_features :", test_features.shape)
print("train_labels  :", train_labels.shape)

# 9. 定义损失函数

loss = nn.MSELoss()

# 输入特征数量
in_features = train_features.shape[1]

# 10. 定义网络

def get_net():

    net = nn.Sequential(nn.Linear(in_features, 1))

    return net

# 11. 定义 Log RMSE

def log_rmse(net, features, labels):

    with torch.no_grad():

        # 预测值不能小于1
        clipped_preds = torch.clamp(net(features),min=1)

        rmse = torch.sqrt(
            loss(torch.log(clipped_preds),torch.log(labels))
        )

    return rmse.item()

# 12. 训练函数

def train(
        net,
        train_features,
        train_labels,
        test_features,
        test_labels,
        num_epochs,
        learning_rate,
        weight_decay,
        batch_size
):

    train_ls = []
    test_ls = []

    # 创建数据集
    dataset = TensorDataset(train_features,train_labels)

    # 创建 DataLoader
    train_iter = DataLoader(dataset,batch_size=batch_size,shuffle=True)

    # Adam 优化器
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    # 开始训练

    for epoch in range(num_epochs):

        net.train()

        for X, y in train_iter:

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            predictions = net(X)

            # 计算损失
            l = loss(predictions, y)

            # 反向传播
            l.backward()

            # 更新参数
            optimizer.step()

        # 每个 epoch 结束后计算训练误差

        train_score = log_rmse( net,train_features,train_labels)

        train_ls.append(train_score)

        # 如果存在验证集

        if test_labels is not None:

            test_score = log_rmse(net,test_features,test_labels)

            test_ls.append(test_score)

        # 每10轮输出一次
        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch [{epoch + 1:3d}/{num_epochs}] "
                f"Train Log RMSE: {train_score:.4f}"
            )

    return train_ls, test_ls

# 13. K折交叉验证

def get_k_fold_data(k, i, X, y):

    assert k > 1

    fold_size = X.shape[0] // k

    X_train = None
    y_train = None

    for j in range(k):

        start = j * fold_size
        end = (j + 1) * fold_size

        X_part = X[start:end]
        y_part = y[start:end]

        # 当前部分作为验证集
        if j == i:

            X_valid = X_part
            y_valid = y_part

        # 第一次生成训练集
        elif X_train is None:

            X_train = X_part
            y_train = y_part

        # 后面不断拼接
        else:

            X_train = torch.cat([X_train, X_part],dim=0)

            y_train = torch.cat([y_train, y_part], dim=0)

    return (X_train,y_train,X_valid,y_valid)


# 14. K折交叉验证训练

def k_fold(
        k,
        X_train,
        y_train,
        num_epochs,
        learning_rate,
        weight_decay,
        batch_size
):

    train_l_sum = 0
    valid_l_sum = 0

    for i in range(k):

        print("\n")

        print(f"第 {i + 1}/{k} 折")

        # 获取当前折的数据
        # ----------------------------------------------------

        data = get_k_fold_data(k,i,X_train,y_train)

        # 创建新的网络

        net = get_net()

        # 训练

        train_ls, valid_ls = train(
            net,
            *data,
            num_epochs,
            learning_rate,
            weight_decay,
            batch_size
        )

        # 保存最后一轮的误差

        train_l_sum += train_ls[-1]

        valid_l_sum += valid_ls[-1]

        print(f"\n第 {i + 1} 折结果：")
        print(f"训练 Log RMSE：{train_ls[-1]:.4f}")
        print(f"验证 Log RMSE：{valid_ls[-1]:.4f}")

    # 平均
    return (train_l_sum / k,valid_l_sum / k)


# 15. 设置超参数

k = 5

num_epochs = 100

learning_rate = 5

weight_decay = 0

batch_size = 64


# 16. 开始 K 折交叉验证

print("\n\n开始 K 折交叉验证")

train_l, valid_l = k_fold(
    k,
    train_features,
    train_labels,
    num_epochs,
    learning_rate,
    weight_decay,
    batch_size
)


# 17. 最终结果

print("\n")
print("=" * 60)

print( f"{k}-折交叉验证结果")

print(f"平均训练 Log RMSE：{train_l:.4f}")

print(f"平均验证 Log RMSE：{valid_l:.4f}")

print("=" * 60)