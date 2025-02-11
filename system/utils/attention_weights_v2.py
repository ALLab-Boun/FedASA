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
        self.key_reduce = nn.Linear(feature_size, intermediate_size)
        self.query_reduce = nn.Linear(feature_size, intermediate_size)
        
        ## Optional non-linear activation
        self.activation = nn.ReLU()
        
        # Expand dimensionality to control the number of attention weights
        self.key_expand = nn.Linear(intermediate_size, 1)
        self.query_expand = nn.Linear(intermediate_size, 1)
        
        # Value transformation remains the same size
        self.value = nn.Linear(feature_size,feature_size)

    def forward(self, x_to_query,x_to_key):
        # Dimensionality reduction
        reduced_keys = self.activation(self.key_reduce(x_to_key.to(self.device)))
        reduced_queries = self.activation(self.query_reduce(x_to_query.to(self.device)))
        
        # Dimensionality expansion
        keys = self.key_expand(reduced_keys)
        queries = self.query_expand(reduced_queries)
        values = self.value(x_to_key.to(self.device))
        
        # Compute scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.num_attention_weights, dtype=torch.float32))
        

        scores_aggregated = scores.sum(dim=2)  # Now shape [20, 10]

        # Optionally, sum or average across the remaining label dimension to get [20]
        scores_final = scores_aggregated.mean(dim=1)  # Summing could also be used depending on the desired influence

        # Apply softmax across the client dimension to normalize these into scalar weights
        client_weights = F.softmax(scores_final, dim=0)
        
        # Softmax to obtain attention weights
        
        attention_weights = F.softmax(scores, dim=-1)
        
        values = values.permute(1, 0, 2)
        weights = client_weights.view(1, len(x_to_query), 1)
        weighted_data = values * weights
        output = weighted_data.sum(dim=1)

        all_clt = x_to_key.to(self.device).permute(1, 0, 2)
        weighted_proto = all_clt * weights
        output_proto = weighted_proto.sum(dim=1)
        # Apply attention weights to the values
        #output = torch.matmul(client_weights, values)
        
        return output, client_weights, output_proto
    

