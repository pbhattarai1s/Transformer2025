import torch 
import torch.nn as nn
from models.event_net import EventNet
from models.object_net import ObjectEvent
from torch.autograd import Function


"""Classifier is a neural network designed to process event and object level features into a final classification output.
This is a simple feedforward network that takes the concatenated event and object representations as input.
"""

"""
Adding an adverserial component to it using gradient reversal 
"""

# this is the gradient reversal function for adversary that takes an arbitary value of lambda for penlaizing correct prediction of mbb
class GradientReversal(Function):
    @staticmethod # doesn't need to acess any other class properties
    def forward(ctx, x, adv_lambda):
        ctx.adv_lambda = adv_lambda
        return x.view_as(x)
    
    def backward(ctx, grad_output):
        return -ctx.adv_lambda * grad_output, None

class EventClassifier(nn.Module):
    def __init__(self, event_input_dim, object_input_dim, hidden_dim, output_dim, num_heads, num_encoder_layers, ff_dim, batch_first=True, adv_lambda=None, adv_method = None, mbb_bins = None ):
        super(EventClassifier, self).__init__()
        self.event_net = EventNet(event_input_dim, hidden_dim, output_dim)
        self.object_net = ObjectEvent(object_input_dim, hidden_dim, num_heads=num_heads, num_encoder_layers=num_encoder_layers, ff_dim=ff_dim, batch_first=batch_first)
        self.adv_lambda = adv_lambda
        self.adv_method = adv_method
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1), #should update this to match the output dimension of your classifier
            #nn.Sigmoid()
        )
        self.adversary = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(mbb_bins)), #should update this to match the output dimension of your classifier
            #nn.Sigmoid()
        )
       
    def forward(self, event_features, object_features,valid_mask):
        event_representation = self.event_net(event_features)
        object_representation = self.object_net(object_features,src_key_padding_mask=~valid_mask)
        combined_representation = torch.cat((event_representation, object_representation), dim=-1)
        classif_output = self.classifier(combined_representation)

        adv_output = None
        if self.adv_method == "adversary_on_representation":
            flipped_features = GradientReversal.apply(combined_representation, self.adv_lambda  )
            adv_output = self.adversary(flipped_features)
        else:
            combined = torch.cat([combined_representation, classif_output], dim=-1) 
            flipped_features = GradientReversal.apply(combined, self.adv_lambda  )
            adv_output = self.adversary(flipped_features)

        return classif_output, adv_output
    