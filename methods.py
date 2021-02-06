from torch.utils.data.dataloader import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from dataset import dataset_training_teaching, ToDataset, DataClients
from update import Global_FedDF, Local_FedDF, Global_FedAvg
from tqdm import tqdm
from options import args_parser
import numpy as np


def fed_ag(data_clients, dataset_teaching, data_global_test):
    args = args_parser()

    global_model1 = Global_FedDF(dataset_global_teaching=ToDataset(dataset_teaching, ToTensor()),
                                num_classes = args.num_classes,
                                num_epochs_global_teaching=args.num_epochs_global_teaching,
                                batch_size_global_teaching=args.batch_size_global_teaching,
                                lr_global_teaching=args.lr_global_teaching,
                                ld=args.ld,
                                temperature=args.temperature,
                                device=args.device)

    local_model1 = Local_FedDF(num_classes=args.num_classes,
                        num_epochs_local_training=args.num_epochs_local_training,
                        batch_size_local_training=args.batch_size_local_training,
                        lr_local_training=args.lr_local_training,
                        device=args.device)



    # 开始训练
    total_clients = list(range(args.num_clients))
    all_acc1 = []
    all_loss1 = []
    for r in range(args.num_rounds):
        print('EPOCH: [%03d/%03d]' % (r+1, args.num_rounds))

        dict_global_params1 = global_model1.download_params()

        online_clients = np.random.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params1 = []

        list_nums_local_data = []

        #local training
        for client in tqdm(online_clients, desc='local training'):
            data_client = ToDataset(data_clients[client], ToTensor())
            list_nums_local_data.append(len(data_client))
            local_model1.train(dict_global_params1, data_client)
            dict_local_params1 = local_model1.upload_params()
            list_dicts_local_params1.append(dict_local_params1)
        #global update
        global_model1.update(list_dicts_local_params1, list_nums_local_data)
        #global valiation
        acc1, loss1 = global_model1.validation(data_global_test, args.batch_size_test)
        all_acc1.append(acc1)
        all_loss1.append(loss1)
        print('-' * 21)
    return all_acc1, all_loss1


def fedavg(data_clients, data_global_test):
    args = args_parser()
    global_model2 = Global_FedAvg(device=args.device, num_classes=args.num_classes)
    local_model2 = Local_FedDF(num_classes=args.num_classes,
                              num_epochs_local_training=args.num_epochs_local_training,
                              batch_size_local_training=args.batch_size_local_training,
                              lr_local_training=args.lr_local_training,
                              device=args.device)
    # 开始训练
    total_clients = list(range(args.num_clients))
    all_acc = []
    all_loss = []
    for r in range(args.num_rounds):
        print('EPOCH: [%03d/%03d]' % (r + 1, args.num_rounds))
        dict_global_params = global_model2.download_params()
        online_clients = np.random.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params = []
        list_nums_local_data = []

        # local training
        for client in tqdm(online_clients, desc='local training'):
            data_client = ToDataset(data_clients[client], ToTensor())
            list_nums_local_data.append(len(data_client))
            local_model2.train(dict_global_params, data_client)
            dict_local_params = local_model2.upload_params()
            list_dicts_local_params.append(dict_local_params)

        # global update
        global_model2.update(list_dicts_local_params, list_nums_local_data)
        # global valiation
        acc, loss = global_model2.validation(data_global_test, args.batch_size_test)
        all_acc.append(acc)
        all_loss.append(loss)
        print('-' * 21)
    return all_acc, all_loss


#两个方法每一轮选择相同的clients
def fed_ag1(data_clients, dataset_teaching, data_global_test):
    args = args_parser()

    global_model1 = Global_FedDF(dataset_global_teaching=ToDataset(dataset_teaching, ToTensor()),
                                 num_classes=args.num_classes,
                                 num_epochs_global_teaching=args.num_epochs_global_teaching,
                                 batch_size_global_teaching=args.batch_size_global_teaching,
                                 lr_global_teaching=args.lr_global_teaching,
                                 ld=args.ld,
                                 temperature=args.temperature,
                                 device=args.device)
    global_model2 = Global_FedAvg(device=args.device, num_classes=args.num_classes)
    local_model1 = Local_FedDF(num_classes=args.num_classes,
                               num_epochs_local_training=args.num_epochs_local_training,
                               batch_size_local_training=args.batch_size_local_training,
                               lr_local_training=args.lr_local_training,
                               device=args.device)

    local_model2 = Local_FedDF(num_classes=args.num_classes,
                               num_epochs_local_training=args.num_epochs_local_training,
                               batch_size_local_training=args.batch_size_local_training,
                               lr_local_training=args.lr_local_training,
                               device=args.device)

    # 开始训练
    total_clients = list(range(args.num_clients))
    all_acc1 = []
    all_loss1 = []
    all_acc2 = []
    all_loss2 = []
    for r in range(args.num_rounds):
        print('EPOCH: [%03d/%03d]' % (r + 1, args.num_rounds))

        dict_global_params1 = global_model1.download_params()
        dict_global_params2 = global_model2.download_params()

        online_clients = np.random.choice(total_clients, args.num_online_clients, replace=False)
        list_dicts_local_params1 = []
        list_dicts_local_params2 = []
        list_nums_local_data = []

        # local training
        for client in tqdm(online_clients, desc='local training'):
            data_client = ToDataset(data_clients[client], ToTensor())
            list_nums_local_data.append(len(data_client))

            local_model1.train(dict_global_params1, data_client)
            dict_local_params1 = local_model1.upload_params()
            list_dicts_local_params1.append(dict_local_params1)
            #
            local_model2.train(dict_global_params2, data_client)
            dict_local_params2 = local_model2.upload_params()
            list_dicts_local_params2.append(dict_local_params2)

        # global update
        global_model1.update(list_dicts_local_params1, list_nums_local_data)
        global_model2.update(list_dicts_local_params2, list_nums_local_data)
        # global valiation
        acc1, loss1 = global_model1.validation(data_global_test, args.batch_size_test)
        acc2, loss2 = global_model2.validation(data_global_test, args.batch_size_test)
        all_acc1.append(acc1)
        all_loss1.append(loss1)
        all_acc2.append(acc2)
        all_loss2.append(loss2)
        print('-' * 21)
    return all_acc1, all_loss1, all_acc2, all_loss2
