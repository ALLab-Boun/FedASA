from flcore.clients.client_alp_2nd_algo import clientalp_v2
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from threading import Thread
import random
import time
import numpy as np
from collections import defaultdict




class Fed_alp_v2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientalp_v2)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]



    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_protos()
            #self.global_protos = proto_aggregation(self.uploaded_protos,self.uploaded_label_weights)
            self.global_protos = proto_aggregation_old(self.uploaded_protos)
           
            self.send_protos()

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_label_weights = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            self.uploaded_label_weights.append(client.label_weights)

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
            

    
    #baseden geleni kullanalaım
    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

   # https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list,local_label_weights_list):
    agg_protos_label = defaultdict(list)
    weight_dict = defaultdict(list)
    for id,local_protos in enumerate(local_protos_list):
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])
            weight_dict[label].append(local_label_weights_list[id][label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            weight_list = weight_dict[label]
            total_sample = 0
            for i,z in zip(proto_list,weight_list):
                proto += i.data * z
                total_sample  += z
            agg_protos_label[label] = proto  / total_sample
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label


def proto_aggregation_old(local_protos_list):
    agg_protos_label = defaultdict(list)
    for local_protos in local_protos_list:
        for label in local_protos.keys():
            agg_protos_label[label].append(local_protos[label])

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = proto / len(proto_list)
        else:
            agg_protos_label[label] = proto_list[0].data

    return agg_protos_label

