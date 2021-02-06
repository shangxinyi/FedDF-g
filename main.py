from options import args_parser
import numpy as np
from torch import tensor, add, mul, div, max, eq
from torch.optim import SGD
from torch.nn import CrossEntropyLoss, KLDivLoss
from torch.nn.functional import softmax, log_softmax
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataset import dataset_training_teaching, ToDataset, DataClients
from update import Global_FedDF, Local_FedDF
from tqdm import tqdm
import matplotlib.pyplot as plt
from methods import fed_ag, fedavg, fed_ag1

if __name__ == '__main__':
    args = args_parser()
    dataset = datasets.CIFAR10(args.path_cifar10, download=True)
    # 将数据集划分为训练集和蒸馏集
    dataset_training, dataset_teaching = dataset_training_teaching(dataset=list(dataset),
                                                                   num_classes=args.num_classes,
                                                                   num_data_training=args.num_data_training)
    data_global_test = datasets.CIFAR10(args.path_cifar10, train=False, transform=ToTensor(), download=True)
    # 给每个client分数据
    data_clients = DataClients(dataset=dataset_training,
                               num_classes=args.num_classes,
                               num_clients=args.num_clients)
    #all_acces1, all_losses1, all_acces2, all_losses2 = fed_ag1(data_clients, dataset_teaching, data_global_test)
    all_acces1, all_losses1 = fed_ag(data_clients, dataset_teaching, data_global_test)

    all_acces2, all_losses2 = fedavg(data_clients, data_global_test)

    X = range(0, args.num_rounds)

    # 绘制acc图
    plt.figure()
    plt.plot(list(X), all_acces1, label='FedAG')
    plt.plot(list(X), all_acces2, label='FedAvg')
    plt.xlabel("Global rounds")
    plt.ylabel("Testing Accuracy")
    plt.grid()  # 生成网格
    plt.legend()

    # 绘制loss图
    plt.figure()
    plt.plot(list(X), all_losses1, label='FedAG')
    plt.plot(list(X), all_losses2, label='FedAvg')
    plt.xlabel("Global rounds")
    plt.ylabel("Training Loss")
    plt.grid()  # 生成网格
    plt.legend()

    plt.show()
