import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import preprocessing
from torch.utils.data import TensorDataset
from tqdm import tqdm
import json
import os
import warnings
from sklearn.neural_network import MLPRegressor

from coding.llh.analysis.shap_model import shap_calculate
from coding.llh.static.process import grid_search, bayes_search
from coding.llh.visualization.draw_line_graph import draw_line_graph
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import learning_curve
import numpy as np

from coding.llh.static.config import Config
from coding.llh.static.process import grid_search, bayes_search
from coding.llh.visualization.draw_learning_curve import draw_learning_curve
from coding.llh.visualization.draw_line_graph import draw_line_graph
from coding.llh.visualization.draw_scatter_line_graph import draw_scatter_line_graph
from coding.llh.metrics.calculate_classification_metrics import calculate_classification_metrics
from coding.llh.metrics.calculate_regression_metrics import calculate_regression_metrics
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


def mlp_regression(feature_names, x, y, x_train_and_validate, y_train_and_validate, x_test, y_test, train_and_validate_data_list=None, hyper_params_optimize=None):
    info = {}
    model_name = "mlp regression model"

    model = MLPRegressor()
    params = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [100, 200, 300]
    }

    if hyper_params_optimize == "grid_search":
        best_model = grid_search(params, model, x_train_and_validate, y_train_and_validate)
    elif hyper_params_optimize == "bayes_search":
        best_model = bayes_search(params, model, x_train_and_validate, y_train_and_validate)
    else:
        best_model = model
        best_model.fit(x, y)

    info["{} Params".format(model_name)] = best_model.get_params()

    y_pred = best_model.predict(x_test).reshape(-1, 1)

    # 0202:

    train_sizes, train_scores, test_scores = learning_curve(best_model, x[:500], y[:500], cv=5, scoring="r2")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # draw_learning_curve(train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)

    # draw_scatter_line_graph(x_test, y_pred, y_test, lr_coef, lr_intercept, ["pred", "real"], "logistic regression model residual plot")

    info.update(calculate_regression_metrics(y_pred, y_test, model_name))
    # info.update(calculate_classification_metrics(y_pred, y_test, "logistic regression"))
    # mae, mse, rsme, r2, ar2 = calculate_regression_metrics(y_pred, y_test, model_name)

    # shap_calculate(best_model, x_test, feature_names)

    return info, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std


def ann(df):
    # 参数初始化
    lr = 0.0001
    batch_size = 32
    input_dim = 10
    output_dim = 4
    epochs = 40
    best_acc = 0
    save_path = "./model/model.pth"

    # 硬件定义
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device loaded for training: [{}]".format(device))

    # 数据集分割
    def split_data(data: pd.DataFrame):
        data = np.array(data)

        dataX = data[:, 1:]
        dataY = data[:, :1]

        dataX = np.array(dataX)
        dataY = np.array(dataY)

        total_size = dataX.shape[0]
        train_size = int(np.round(0.8 * total_size))

        x_train = dataX[: train_size, :]
        y_train = dataY[: train_size]

        x_test = dataX[train_size:, :]
        y_test = dataY[train_size:]

        return x_train, y_train, x_test, y_test, total_size, train_size

    x_train, y_train, x_test, y_test, total_size, train_size = split_data(df)

    # 数据预处理
    x_train = preprocessing.scale(x_train)
    x_test = preprocessing.scale(x_test)

    y_train = y_train - 1
    y_test = y_test - 1

    # 数据格式转换
    x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).to(torch.float32)

    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size, False)

    print("Data loaded for training: [{}]".format(len(train_data)))
    print("Data loaded for testing: [{}]".format(len(test_data)))

    # 模型定义
    class ANN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(ANN, self).__init__()

            self.hidden1 = nn.Sequential(
                nn.Linear(input_dim, 16, bias=True),
                nn.ReLU()
            )
            self.hidden2 = nn.Sequential(
                nn.Linear(16, 32, bias=True),
                nn.ReLU()
            )
            self.hidden3 = nn.Sequential(
                nn.Linear(32, 64, bias=True),
                nn.ReLU()
            )
            self.hidden4 = nn.Sequential(
                nn.Linear(64, 128, bias=True),
                nn.ReLU()
            )
            self.hidden5 = nn.Sequential(
                nn.Linear(128, 256, bias=True),
                nn.ReLU()
            )
            self.hidden6 = nn.Sequential(
                nn.Linear(256, 512, bias=True),
                nn.ReLU()
            )
            self.hidden7 = nn.Sequential(
                nn.Linear(512, 1024, bias=True),
                nn.ReLU()
            )
            self.hidden8 = nn.Sequential(
                nn.Linear(1024, output_dim, bias=True),
                nn.Softmax()
            )

        def forward(self, x):
            x = self.hidden1(x)
            x = self.hidden2(x)
            x = self.hidden3(x)
            x = self.hidden4(x)
            x = self.hidden5(x)
            x = self.hidden6(x)
            x = self.hidden7(x)
            x = self.hidden8(x)

            return x

    model = ANN(input_dim, output_dim).to(device)
    print("Model set: [{}]".format(model))

    # 损失函数定义
    criterion = nn.CrossEntropyLoss()
    print("Criterion set: [{}]".format(type(criterion)))

    # 优化器定义
    optimizer = torch.optim.Adam(model.parameters(), lr)
    print("Optimizer set: [{}]".format(type(optimizer)))
    print()

    if os.path.isfile(save_path):
        # 模型加载
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict, strict=False)
        print("!Model loaded")

        with open("./model/best_acc.json", "r") as f:
            print("Best accuracy of current model: [{}]".format(json.load(f)))

    else:
        print("!Training starting\n")

        train_loss_list = []
        train_acc_list = []
        test_loss_list = []
        test_acc_list = []

        y_pred_list = []
        y_real_list = []

        for epoch in range(epochs):
            # 模型训练
            model.train()

            train_loss = 0
            train_acc = 0
            train_acc_count = 0
            train_count = 0
            train_bar = tqdm(train_loader)
            for data in train_bar:
                x_train, y_train = data
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                # 优化器重置
                optimizer.zero_grad()
                # 前向传播
                output = model(x_train)
                # 计算误差
                loss = criterion(output, y_train.reshape(-1).long())
                # 反向传播：更新梯度
                loss.backward()
                # 反向传播：更新参数
                optimizer.step()

                train_loss += loss.item()
                train_bar.desc = "Train epoch[{}/{}] loss: {:.3f}".format(epoch + 1, epochs, loss)
                train_acc_count += (output.argmax(axis=1) == y_train.view(-1).int()).sum().item()
                train_count += len(x_train)

            train_acc = train_acc_count / train_count

            # 模型测试
            model.eval()

            test_loss = 0
            test_acc = 0
            test_acc_count = 0
            test_count = 0
            with torch.no_grad():
                test_bar = tqdm(test_loader)
                for data in test_bar:
                    x_test, y_test = data
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    # 前向传播
                    output = model(x_test)

                    y_pred_list.append(output.tolist())
                    y_real_list.append(y_test.tolist())

                    # 计算误差
                    loss = criterion(output, y_test.reshape(-1).long())

                    test_loss += loss.item()
                    test_bar.desc = "Test epoch[{}/{}] loss: {:.3f}".format(epoch + 1, epochs, loss)
                    test_acc_count += (output.argmax(axis=1) == y_test.view(-1).int()).sum().item()
                    test_count += len(x_test)

                test_acc = test_acc_count / test_count

            print("\nEpoch: {}".format(epoch + 1))
            print("Train_loss: {:.4f}".format(train_loss))
            print("Train_accuracy: {:.4f}".format(train_acc))
            print("Test_loss: {:.4f}".format(test_loss))
            print("Test_accuracy: {:.4f}".format(test_acc))
            print("\n")

            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)

            # 保存当前最优模型和最优准确率值
            if test_acc > best_acc:
                best_acc = test_acc
                with open("./model/info.json", "w") as f:
                    json.dump({
                        "best_acc": [best_acc],
                        "train_loss_list": train_loss_list,
                        "train_acc_list": train_acc_list,
                        "test_loss_list": test_loss_list,
                        "test_acc_list": test_acc_list,
                        "y_pred_list": y_pred_list,
                        "y_real_list": y_real_list
                    }, f)

                torch.save(model.state_dict(), save_path)

        print("\n!Training finished")
        print("Best accuracy: {:.4f}".format(best_acc))

        # 数据可视化
        draw_line_graph(
            range(len(y_pred_list)),
            [y_pred_list, y_real_list],
            "ANN prediction",
            ["predict, real"]
        )

