from flcore.clients.client_alp_4th_algo_attention import clientalp_v4
from flcore.servers.serverbase import Server
from utils.data_utils import read_client_data
from utils.federated_attention import SelfAttentionLayer
from threading import Thread
import time
import numpy as np
import copy
import random
from collections import defaultdict
from numpy import dot
from numpy.linalg import norm
import torch
import torch.optim as optim
import torch.nn.functional as F




class Fed_alp_v4(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientalp_v4)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = [None for _ in range(args.num_classes)]
        self.uploaded_heads = []


    def train(self):

        first_iter_flag = 1
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
            self.global_protos = proto_aggregation(self.uploaded_protos)
            #self.send_protos()

            self.receive_models()
            self.aggregate_parameters()

            #embed_size = 512  # Example: Adjust based on your prototype size
            #num_classes = 10  # Example: Adjust based on your total number of classes
            #heads = 8  # Number of attention heads in the self-attention mechanism

            num_classes = self.num_classes
            feature_size = 512
            models = {i: SelfAttentionLayer(feature_size) for i in range(num_classes)}

            label_client_dict = {}

            label_client_dict_attention = {}

            for el in range(0,num_classes):
                label_client_dict[el] = {}
                label_client_dict_attention[el] = {}


            for idx,value in enumerate(self.uploaded_protos):

                for lab,prot in value.items():
                    label_client_dict[lab][self.uploaded_ids[idx]] = prot
                    label_client_dict_attention[lab][self.uploaded_ids[idx]] = prot

            # Train each model
            for label in models:
                # Example training data for this class
               # data = label_client_dict[label]  # 100 examples, 10 sequence length #data = data.values()

                all_prototypes = []

                # Iterate through each client and each prototype
                for client_id, prototypes in label_client_dict[label].items():
                    all_prototypes.append(prototypes.squeeze()) 
                data = torch.stack(all_prototypes, dim=0)
                
                model = models[label]
                optimizer = optim.Adam(model.parameters(), lr=0.001)

                for epoch in range(100):
                    optimizer.zero_grad()
                    output, attention_weights = model(data)
                    loss = F.mse_loss(output, data) 
                    loss.backward()
                    optimizer.step()
                    print(f"Epoch {epoch + 1}, Loss: {loss.item()}, Class: {label}")
                
                count = 0
                for client_id,prototypes in label_client_dict[label].items():

                    label_client_dict_attention[label][client_id] = output[count]
                    count+= 1

                #after 10 epochs save the calculated outputs for all label , clients

            client_label_proto_dict = {}



            
            for key,value in label_client_dict_attention.items():
                for clt,prt in value.items():
                    if clt in client_label_proto_dict:
                        client_label_proto_dict[clt][key] = prt.detach()
                    elif clt not in client_label_proto_dict:
                        client_label_proto_dict[clt] = {}
                        client_label_proto_dict[clt][key] = prt.detach()

                    else:
                        print("Error")

            #o client o ana kadar görmediyse o classı global proto gödnerilir
            for label in range(0,num_classes):
                for clt,lab_prot in client_label_proto_dict.items():
                    if label not in client_label_proto_dict[clt]:
                        client_label_proto_dict[clt][label] = self.global_protos[label].detach()
                

            #aynı şekilde üretelim dictioanryi


            self.send_protos_attention(client_label_proto_dict)
            #her clienta gidecek olan global protos bu hesaplananlar olsun, 
            #o clienttan gelmeyen proto yerine aggregated protos kullanalım
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
        

    def send_protos_attention(self,client_protos_w_attention):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(client_protos_w_attention[client.id])

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

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
        
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)
            

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])

        print("client_based_accuraccies")
        count = 0
        for el1,el2 in zip(stats[2],stats[1]):
            print(stats[0][count],":" ,round(el1/el2,2)*100)
            count += 1

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


    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.uploaded_heads = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
                self.uploaded_heads.append(client.model.head)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_and_send_heads(self):
        head_weights = prototype_distances(self.uploaded_ids,self.uploaded_protos,self.num_classes)

        for idx, cid in enumerate(self.uploaded_ids):
            print('(Client {}) Weights of Classifier Head'.format(cid))
            print(head_weights[idx],'\n')

            if head_weights[idx] is not None:
                new_head = self.add_heads(head_weights[idx])
            else:
                new_head = self.uploaded_heads[cid]

            self.clients[cid].set_head(new_head)


    def add_heads(self, weights):
        new_head = copy.deepcopy(self.uploaded_heads[0])
        for param in new_head.parameters():
            param.data.zero_()
                    
        for w, head in zip(weights, self.uploaded_heads):
            for server_param, client_param in zip(new_head.parameters(), head.parameters()):
                server_param.data += client_param.data.clone() * w
        return new_head
            

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L221
def proto_aggregation(local_protos_list):
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




def proto_aggregation_with_self_attention(client_ids, client_prototypes, model, num_classes, epochs=5, lr=0.001):
    """
    Train the model on client prototypes for several epochs to refine attention weights, 
    then aggregate prototypes for each class using an attention-based weighted average.

    :param client_ids: A list of client identifiers.
    :param client_prototypes: A dictionary where keys are client_ids and values are dictionaries with
                              labels as keys and prototypes as values.
    :param model: The self-attention model.
    :param num_classes: The total number of unique classes across all clients.
    :param epochs: Number of epochs to train the model on prototypes.
    :param lr: Learning rate for the optimizer.
    :return: A dictionary with labels as keys and aggregated prototypes as values.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()

    # Combine all prototypes into a single dataset for training
    all_prototypes = []
    all_labels = []
    for client_id in client_ids:
        for label, proto in client_prototypes[client_id].items():
            all_prototypes.append(proto)
            all_labels.append(label)
    all_prototypes_tensor = torch.stack(all_prototypes).unsqueeze(1)  # Shape: [total_prototypes, 1, embed_size]
    all_labels_tensor = torch.tensor(all_labels)  # Shape: [total_prototypes]

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Assuming the model's forward method is adjusted to handle batches of prototypes
        reconstructed_protos, attention_weights = model(all_prototypes_tensor, all_labels_tensor)
        loss = F.mse_loss(reconstructed_protos, all_prototypes_tensor)
        loss.backward()
        optimizer.step()

    # After training, aggregate prototypes using the refined attention weights
    aggregated_prototypes = {label: torch.zeros(model.embed_size) for label in range(num_classes)}
    total_attention_weights = {label: 0 for label in range(num_classes)}

    # Use the attention weights to aggregate prototypes
    for i, label in enumerate(all_labels):
        proto = all_prototypes[i]
        attention_weight = attention_weights[i, i].item()  # Assuming diagonal attention weight represents self-attention
        aggregated_prototypes[label] += proto * attention_weight
        total_attention_weights[label] += attention_weight

    # Normalize the aggregated prototypes
    for label in aggregated_prototypes:
        if total_attention_weights[label] > 0:
            aggregated_prototypes[label] /= total_attention_weights[label]

    return aggregated_prototypes






def prototype_distances(client_ids, prototypes, num_classes):
    weights = []

    # Assuming prototype vector size is 512
    vector_size = 512

    # Iterate over each client
    for clt in client_ids:
        
        weights_for_client = []
        # Iterate over each class to calculate total cosine similarity
        for other_client in client_ids:
        
            
            total_cos_sim = 0
            # Compare with other clients
            for cls in range(num_classes):
                # Fetch the prototype tensors
                prototype_a = prototypes[clt][cls]
                
                prototype_b = prototypes[other_client][cls]

                # Check if either prototype tensor is empty, and replace with a zero vector if so
                try:
                    prototype_a = prototype_a if prototype_a != [] else torch.zeros(vector_size)
                    prototype_b = prototype_b if prototype_b != [] else torch.zeros(vector_size)
                except:
                    return prototype_b

                # Normalize the prototypes to unit vectors, avoid division by zero
                prototype_a_norm = prototype_a / prototype_a.norm(dim=0) if prototype_a.norm(dim=0) > 0 else prototype_a
                prototype_b_norm = prototype_b / prototype_b.norm(dim=0) if prototype_b.norm(dim=0) > 0 else prototype_b

                # Calculate cosine similarity as the dot product of normalized vectors
                # Adding a small epsilon to the denominator for numerical stability
                cos_sim = torch.dot(prototype_a_norm, prototype_b_norm) / (prototype_a_norm.norm() * prototype_b_norm.norm() + 1e-8)
                total_cos_sim += cos_sim.item()  # Convert to Python scalar with .item()
                
            weights_for_client.append(total_cos_sim)

        #TODO yakınlık thresholdu koymak lazım boşuna noise geliyor gibi
        weights_for_client = [el / sum(weights_for_client) for el in weights_for_client] 
        weights.append(weights_for_client)
        
        # Normalize total cosine similarity by the number of classes and clients to prevent scale issues
        #normalized_total_cos_sim = total_cos_sim / (num_classes * len(client_ids))
        #weights.append(normalized_total_cos_sim)

    return weights






        
