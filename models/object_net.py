import torch
import torch.nn as nn

"""
ObjectEvent is a neural network designed to process object level features into a object representation. That is later feed into classifier along with event level representation
This is a transformer based model that uses nn.TransformerEncoderLayer to process the object level features
We can modify this to use a simple feedforward network or a more complex architecture like a CNN or RNN if needed
"""


class ObjectEvent(nn.Module):
    def __init__(self,input_dim, model_dim, num_heads, num_encoder_layers, ff_dim, batch_first=True, pooltype='GlobalAttentionPooling'):
        super(ObjectEvent, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.positional_embedding = nn.Embedding(input_dim, model_dim)  # assuming a maximum of 1000 objects, can be adjusted
        self.attn_mlp = nn.Linear(model_dim, 1)  # global attention pooling layer
        self.pooltype = pooltype  # type of pooling to use, can be 'GlobalAttentionPooling' or 'MeanPooling'


    def forward(self, x):
        """
        Forward pass through the network
        :param x: Input tensor of shape (batch_size, num_objects, input_dim)
        :return: Output tensor of shape (batch_size, num_objects, model_dim)
        """
        # Project input features to model dimension
        x = self.input_projection(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0),-1) #adding batch dimension for positional encoding
        x = x + self.positional_embedding(positions)  # add positional encoding to the embbedded object features
        x = self.transformer_encoder(x)
        ##### Object Pooling  i.e. aggregating across all objects #####
        if self.pooltype == 'GlobalAttentionPooling':
            # Global attention pooling
            scores = self.attn_mlp(x)  # (batch, n_objects, 1)
            weights = torch.softmax(scores, dim=1)  # (batch, n_objects, 1)
            x = (weights * x).sum(dim=1)  # (batch, model_dim)
        
        else:
            x = x.mean(dim=1)  #just mean pooling if not specified otherwise 
      
        return x  # final object representation that gets passed to the classifier