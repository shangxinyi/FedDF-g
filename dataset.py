import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from global_imbalance import split_data_raelwd
from sampling_dirichlet import clients_indices

#将数据集划分为训练集和蒸馏集，
def dataset_training_teaching(dataset, num_classes, num_data_training):
    list_data_same_label = [[] for _ in range(num_classes)]
    for datum in dataset:
        _, label = datum
        list_data_same_label[label].append(datum)
    data_training = []
    data_teaching = []

    for data_same_label in list_data_same_label:
        np.random.shuffle(data_same_label)
        data_training.extend(data_same_label[:num_data_training // 10])
        data_teaching.extend(data_same_label[num_data_training // 10:])
    return data_training, data_teaching


def cifariid(dict_indexs: dict, num_classes: int, num_clients: int):
    list_classes = list(range(num_classes))
    class_partition_split = {}
    for ind, class_ in enumerate(list_classes):
        class_partition_split[class_] = [list(i) for i in np.array_split(dict_indexs[ind], num_clients)]
    clients_split = {client: [] for client in range(num_clients)}
    for client in range(num_clients):
        indcs = []
        for class_ in list_classes:
            if len(class_partition_split[class_]) > 0:
                indcs.extend(class_partition_split[class_][-1])
                class_partition_split[class_].pop()
        clients_split[client] = indcs
    for client in range(num_clients):
        print(len(clients_split[client]))
    return clients_split

class DataClients(object):
    def __init__(self, dataset, num_classes: int, num_clients: int):
        self.dataset = dataset
        self.dict_labels_idxs = self.classify_label(num_classes)
        self.dict_client_idxs = split_data_raelwd(self.dict_labels_idxs, num_classes, num_clients)
        #self.dict_client_idxs = cifariid(self.dict_labels_idxs, num_classes, num_clients)
        #self.dict_client_idxs = clients_indices(self.dict_labels_idxs, num_classes, num_clients, 1)

    def classify_label(self, num_classes):
        dict_label_idxs = {label: [] for label in range(num_classes)}
        for idx, data in enumerate(self.dataset):
            _, label = data
            dict_label_idxs[label].append(idx)

        return dict_label_idxs

    def __getitem__(self, client):
        idxs = self.dict_client_idxs[client]
        data = []
        for idx in idxs:
            data.append(self.dataset[idx])
        return data

    def __len__(self):
        return len(self.dict_client_idxs)


class ToDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)



