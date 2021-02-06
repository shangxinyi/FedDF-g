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
from light_resnet import light_resnet14

from tqdm import tqdm
from global_imbalance import split_data_raelwd


#聚合本地模型+知识蒸馏+全局测试+上传模型参数
class Global_FedDF(object):
    def __init__(self,
                 dataset_global_teaching,
                 num_classes: int,
                 num_epochs_global_teaching: int,
                 batch_size_global_teaching: int,
                 lr_global_teaching: int,
                 ld:float,
                 temperature: float,
                 device:str):
        #用resnet网络
        self.device = device
        self.dataset_teaching = dataset_global_teaching
        self.model = light_resnet14(num_classes)
        self.model.to(self.device)
        self.dict_global_params = self.model.state_dict()
        self.num_classes = num_classes
        self.num_epoch_teaching = num_epochs_global_teaching
        self.batch_size_global_teaching = batch_size_global_teaching
        self.lr_global_teaching = lr_global_teaching
        self.ld = ld
        self.temperature = temperature
        #损失函数
        self.ce_loss = CrossEntropyLoss()
        self.kld_loss = KLDivLoss(reduction='batchmean')
        self.optimizer = SGD(self.model.parameters(), lr=lr_global_teaching)

    #将本地更新聚合
    def aggregation(self, list_dicts_local_params: list, list_nums_local_data: list):
        for name_param in self.dict_global_params:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            self.dict_global_params[name_param] = value_global_param


    def update(self, list_dicts_local_params: list, list_nums_local_data: list):
        self.aggregation(list_dicts_local_params, list_nums_local_data)
        #计算g
        g = self.compute_g(list_dicts_local_params)
        #teaching
        for epoch in tqdm(range(self.num_epoch_teaching), desc='global teaching'):
            self.teach_one_epoch(g, list_dicts_local_params)

    # 计算g
    def compute_g(self, list_dicts_local_params):
        g = [0 for _ in range(self.num_classes)]
        for image, label in tqdm(self.dataset_teaching, desc='computing g'):
            idx = label
            image, label = image.reshape([1, 3, 32, 32]).to(self.device), tensor(label).reshape(1).to(self.device)
            avg_logits = self.avg_logits(image, list_dicts_local_params)
            g[idx] += self.ce_loss(avg_logits, label).item()
        g = div(tensor(g), len(self.dataset_teaching) / self.num_classes)
        print(g)
        return g

    def avg_logits(self, image, list_dicts_local_params: list):
        list_logits = []
        for dict_local_params in list_dicts_local_params:
            self.model.load_state_dict(dict_local_params)
            self.model.eval()
            list_logits.append(self.model(image))
        return sum(list_logits) / len(list_logits)

    #teaching
    def teach_one_epoch(self, g, list_dicts_local_params: list):
        data_loader = DataLoader(self.dataset_teaching, self.batch_size_global_teaching, shuffle=True)
        for batch_data in data_loader:
            images, labels = batch_data
            images, labels = images.to(self.device), labels.to(self.device)
            logits_teacher = add(self.avg_logits(images, list_dicts_local_params), g.to(self.device))
            self.model.load_state_dict(self.dict_global_params)
            self.model.train()
            logits_student = self.model(images)
            x = log_softmax(div(logits_student, self.temperature), -1)
            y = softmax(div(logits_teacher, self.temperature), -1)
            soft_loss = self.kld_loss(x, y.detach())
            hard_loss = self.ce_loss(logits_student, labels)
            total_loss = add(mul(soft_loss, self.ld), mul(hard_loss, 1-self.ld))
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.dict_global_params = self.model.state_dict()

    #validation
    def validation(self, data_test, batch_size_test: int):
        self.model.load_state_dict(self.dict_global_params)
        self.model.eval()
        test_loader = DataLoader(data_test, batch_size_test)
        num_corrects = 0
        list_loss = []
        for data_batch in tqdm(test_loader, desc='global testing'):
            images, labels = data_batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicts = max(outputs, -1)
            num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            loss_batch = self.ce_loss(outputs, labels)
            list_loss.append(loss_batch.cpu().item())
        accuracy = num_corrects / len(data_test)
        loss = sum(list_loss) / len(list_loss)
        print('acc-feddf: %04f' % accuracy)
        print('loss-feddf: %04f' % loss)
        return accuracy, loss

    #上传聚合后参数
    def download_params(self):
        return self.model.state_dict()






#用本地数据更新模型+上传参数
class Local_FedDF(object):
    def __init__(self,
                 num_classes: int,
                 num_epochs_local_training: int,
                 batch_size_local_training: int,
                 lr_local_training: float,
                 device: str):
        self.device = device
        self.model = light_resnet14(num_classes)
        self.model.to(device)
        self.num_epochs = num_epochs_local_training
        self.batch_size = batch_size_local_training
        self.ce_loss = CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=lr_local_training)

    #training
    def train(self, dict_global_params, data_client):
        self.model.load_state_dict(dict_global_params)
        self.model.train()
        for epoch in range(self.num_epochs):
            data_loader = DataLoader(dataset=data_client,
                                     batch_size=self.batch_size,
                                     shuffle=True)
            for data_batch in data_loader:
                images, labels = data_batch
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.ce_loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def upload_params(self):
        return self.model.state_dict()


class Global_FedAvg(object):
    def __init__(self, device: str, num_classes):
        self.device = device
        self.model = light_resnet14(num_classes)
        self.model.to(self.device)
        self.loss_fc = CrossEntropyLoss()
        self.num_classes = num_classes
        self.dict_global_params = self.model.state_dict()
        self.ce_loss = CrossEntropyLoss()

    def update(self, list_dicts_local_params: list, list_nums_local_data: list):
        for name_param in self.dict_global_params:
            list_values_param = []
            for dict_local_params, num_local_data in zip(list_dicts_local_params, list_nums_local_data):
                list_values_param.append(dict_local_params[name_param] * num_local_data)
            value_global_param = sum(list_values_param) / sum(list_nums_local_data)
            self.dict_global_params[name_param] = value_global_param

    def validation(self, data_test, batch_size_test: int):
        self.model.load_state_dict(self.dict_global_params)
        self.model.eval()
        test_loader = DataLoader(data_test, batch_size_test)
        num_corrects = 0
        list_loss = []
        for data_batch in tqdm(test_loader, desc='global testing'):
            images, labels = data_batch
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicts = max(outputs, -1)
            num_corrects += sum(eq(predicts.cpu(), labels.cpu())).item()
            loss_batch = self.ce_loss(outputs, labels)
            list_loss.append(loss_batch.cpu().item())
        accuracy = num_corrects / len(data_test)
        loss = sum(list_loss) / len(list_loss)
        print('acc-fedavg: %04f' % accuracy)
        print('loss-fedavg: %04f' % loss)
        return accuracy, loss

        # 上传聚合后参数
    def download_params(self):
        return self.model.state_dict()


