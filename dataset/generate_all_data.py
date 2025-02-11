from generate_Cifar100 import generate_dataset as cifar_100_gen
from generate_Cifar10 import generate_dataset as cifar_10_gen
from generate_svhn import generate_dataset as svhn_gen
import random
import numpy as np

niid = True 
balance = False
partition = "dir"


random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "SVHN_20_01/"

svhn_gen(dir_path, num_clients, niid, balance, partition)

random.seed(1)
np.random.seed(1)
num_clients = 100
dir_path = "SVHN_100_01/"

svhn_gen(dir_path, num_clients, niid, balance, partition)

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "Cifar10_20_01/"

cifar_10_gen(dir_path, num_clients, niid, balance, partition)

random.seed(1)
np.random.seed(1)
num_clients = 20
dir_path = "Cifar100_20_01/"

cifar_100_gen(dir_path, num_clients, niid, balance, partition)

random.seed(1)
np.random.seed(1)
num_clients = 100
dir_path = "Cifar100_100_01/"

cifar_100_gen(dir_path, num_clients, niid, balance, partition)






