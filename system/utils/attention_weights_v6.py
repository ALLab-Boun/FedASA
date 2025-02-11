import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSelfAttentionLayer(nn.Module):
    def __init__(self, feature_size, intermediate_size, num_attention_weights,device):
        super(EnhancedSelfAttentionLayer, self).__init__()
        self.feature_size = feature_size
        self.intermediate_size = intermediate_size
        self.num_attention_weights = num_attention_weights
        self.device = device
        # Reduce dimensionality
        self.key_reduce = nn.Linear(512, 512)
        self.query_reduce = nn.Linear(512, 512)

        self.value_reduce = nn.Linear(512, 512)

        
        
        ## Optional non-linear activation
        self.activation = nn.ReLU()
        
        # Expand dimensionality to control the number of attention weights
        self.key_expand = nn.Linear(intermediate_size, 1)
        self.query_expand = nn.Linear(intermediate_size, 1)
        
        # Value transformation remains the same size
        self.value = nn.Linear(feature_size,feature_size)

    def forward(self, x_to_query,x_to_key):
        # Dimensionality reduction

        reduced_keys = self.key_reduce(x_to_key.to(self.device))

        reduced_queries = self.query_reduce(x_to_query.to(self.device))

        reduced_value = self.value_reduce(x_to_key.to(self.device))


        weights = F.softmax(torch.matmul(reduced_keys, reduced_queries.T),dim = 0).view(self.num_attention_weights, 1)  # Change shape to [20, 1]



        resulting_prototype = torch.matmul(reduced_value.T,weights)

        resulting_prototype_wa_weights = torch.matmul(x_to_key.T.to(self.device),weights)
        

        client_weights = weights

        output = resulting_prototype
        
        
        

        
        return  client_weights, output
    

