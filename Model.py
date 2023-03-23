from torch import nn
import torch.nn.functional as F


class MLPDoubleModel(nn.Module):

    def __init__(self, number_of_classes, length_of_feature_vector, argv):
        super(MLPDoubleModel, self).__init__()
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

class MLPTripleModel(nn.Module):

    def __init__(self, number_of_classes, length_of_feature_vector, argv):
        super(MLPTripleModel, self).__init__()
        inputDimension = length_of_feature_vector
        hidden_dim = argv.hidden
        dropout_prob = argv.dropout
        self.inputLayer: nn.Linear = nn.Linear(length_of_feature_vector, hidden_dim)
        self.nonLinearity = F.relu
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.hiddenLayer1: nn.Linear = nn.Linear(hidden_dim, hidden_dim)
        self.hiddenLayer2: nn.Linear = nn.Linear(hidden_dim, number_of_classes)
        return

    def forward(self, x):
        layer1PreSigmoid = self.inputLayer(x)
        layer1PreDropout = self.nonLinearity(layer1PreSigmoid)
        layer1Output = self.dropout1(layer1PreDropout)
        layer2PreSigmoid = self.hiddenLayer1(layer1Output)
        layer2PreDropout = self.nonLinearity(layer2PreSigmoid)
        layer2Output = self.dropout2(layer2PreDropout)
        layer3PreSigmoid = self.hiddenLayer2(layer2Output)
        output = layer3PreSigmoid
        return output
