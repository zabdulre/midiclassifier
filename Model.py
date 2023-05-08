from torch import nn
import torch.nn.functional as F
from torch import flatten

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
        self.isAudioModel = False
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
        self.isAudioModel = False
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

class CNNModel(nn.Module):
    def __init__(self, number_of_classes, length_of_feature_matrix, argv):
        super(CNNModel, self).__init__()
        #num freuqencies should be 2048
        inputDimensionFreq, inputDimensionTimestep = length_of_feature_matrix
        dropout_prob = argv.dropout
        self.isAudioModel = True

        #replace kernel of 5x5 with axb (make it look at decent number of frequencies, not too much timestep)
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv3d(16,20,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        return

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        output = x
        return output