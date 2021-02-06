import random
import numpy as np



#把总的份数分给每一个class，以此确定每个class需要分成几份
def break_into(total_partition, num_classes):
    to_ret = [1 for _ in range(num_classes)]
    for i in range(total_partition - num_classes):
        ind = random.randint(0, num_classes - 1)
        to_ret[ind] += 1
    return to_ret


#给列表
def split_data_raelwd(dict_indexs: dict, num_classes: int, num_clients: int):

    #class列表
    list_classes = list(range(num_classes))
    np.random.shuffle(list_classes)

    # 给每一个client随机分几份
    temp = [np.random.randint(1, 11) for i in range(num_clients)]
    total_partition = sum(temp)

    # 为了凑够总的total_partition，每个class要出几份
    class_partition = break_into(total_partition, num_classes)

    #降序排列？为什么需要降序排列
    #class_partition = sorted(class_partition, reverse=True)
    class_partition_split = {}

    #每个class按照上面的份数进行划分，用array_split函数
    for ind, class_ in enumerate(list_classes):
        class_partition_split[class_] = [list(i) for i in np.array_split(dict_indexs[ind], class_partition[ind])]

    #给每个client划分数据
    clients_split = {client: [] for client in range(num_clients)}
    count = 0
    for client in range(num_clients):
        n = temp[client]  #client可以分到的份数
        j = 0
        indcs = []

        while n > 0:
            class_ = list_classes[j]
            if len(class_partition_split[class_]) > 0:
                if class_ in {1, 5, 7} and len(class_partition_split[class_]) > 3:
                    count += len(class_partition_split[class_][-1])
                    class_partition_split[class_].pop()
                    n -= 1
                else:
                    indcs.extend(class_partition_split[class_][-1])
                    count += len(class_partition_split[class_][-1])
                    class_partition_split[class_].pop()
                    n -= 1
            j += 1

        list_classes = sorted(list_classes, key=lambda x: len(class_partition_split[x]), reverse=True)
        clients_split[client] = indcs
    for client in range(num_clients):
        print(len(clients_split[client]))
    return clients_split
