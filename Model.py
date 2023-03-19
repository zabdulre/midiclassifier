from torch import nn
import torch.nn.functional as F


class MLPmodel(nn.Module):

    def __init__(self, number_of_classes, length_of_feature_vector, argv):
        super(MLPmodel, self).__init__()
        inputDimension = length_of_feature_vector
        hidden_dim = argv.hidden
        dropout_prob = argv.dropout
        self.inputLayer: nn.Linear = nn.Linear(inputDimension, hidden_dim)
        self.nonLinearity = F.relu
        self.dropout = nn.Dropout(dropout_prob)
        self.hiddenLayer: nn.Linear = nn.Linear(hidden_dim, number_of_classes)
        return

    def forward(self, features):
        layer1PreSigmoid = self.inputLayer(features)
        layer1PreDropout = self.nonLinearity(layer1PreSigmoid)
        layer1Output = self.dropout(layer1PreDropout)
        layer2PreSigmoid = self.hiddenLayer(layer1Output)
        output = layer2PreSigmoid
        return output
