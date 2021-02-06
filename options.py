import os
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--path_cifar10', type=str, default='/home/a1/桌面/global_imbalance_distillation/data/CIFAR10')
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--num_classes', type=int, default=10)
    # FL设置
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=5)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--num_epochs_local_training', type=int, default=20)
    parser.add_argument('--num_data_training', type=int, default=49000)
    parser.add_argument('--batch_size_local_training', type=int, default=5)
    parser.add_argument('--batch_size_test', type=int, default=100)

    # 蒸馏设置
    parser.add_argument('--num_epochs_global_teaching', type=int, default=50)
    parser.add_argument('--batch_size_global_teaching', type=int, default=10)
    parser.add_argument('--lr_global_teaching', type=float, default=0.01)
    parser.add_argument('--lr_local_training', type=float, default=0.01)
    parser.add_argument('--temperature', type=float, default=1.)

   # 两个loss的trade-off
    parser.add_argument('--ld', type=float, default=0.5)

    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    return args
