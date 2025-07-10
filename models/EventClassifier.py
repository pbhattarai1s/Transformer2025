import torch 
import torch.nn as nn
from models.event_net import EventNet
from models.object_net import ObjectEvent


"""Classifier is a neural network designed to process event and object level features into a final classification output.
This is a simple feedforward network that takes the concatenated event and object representations as input.
"""

class EventClassifier(nn.Module):
    def __init__(self, event_input_dim, object_input_dim, hidden_dim, output_dim, num_heads, num_encoder_layers, ff_dim, batch_first=True ):
        super(EventClassifier, self).__init__()
        self.event_net = EventNet(event_input_dim, hidden_dim, output_dim)
        self.object_net = ObjectEvent(object_input_dim, hidden_dim, num_heads=num_heads, num_encoder_layers=num_encoder_layers, ff_dim=ff_dim, batch_first=batch_first)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1), #should update this to match the output dimension of your classifier
            nn.Sigmoid()
        )

    def forward(self, event_features, object_features):
        event_representation = self.event_net(event_features)
        object_representation = self.object_net(object_features)
        
        combined_representation = torch.cat((event_representation, object_representation), dim=-1)
        output = self.classifier(combined_representation)

        return output
    