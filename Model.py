from torch import nn
import torch.nn.functional as F
from torch import flatten, unsqueeze

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
    def __init__(self, number_of_classes, dims_of_feature_matrix, argv):
        super(CNNModel, self).__init__()
        #num freuqencies should be 2048
        inputDimensionFreq, inputDimensionTimestep = dims_of_feature_matrix #TODO use these
        dropout_prob = argv.dropout
        self.isAudioModel = True
        #replace kernel of 5x5 with axb (make it look at decent number of frequencies, not too much timestep)
        self.conv1 = nn.Conv2d(1, 16, 32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.pool3 = nn.MaxPool2d(2)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.pool4 = nn.MaxPool2d(2)
        self.relu4 = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(832, number_of_classes) #TODO calculate this
        #self.dropout = nn.Dropout(dropout_prob)
        #self.fc2 = nn.Linear(50000, number_of_classes)
        #self.fc1 = nn.Linear(20 * ((((((inputDimensionFreq-k1+1)/2)-k2+1)/2)-k3+1)/2)
       #                      * ((((((inputDimensionTimestep-k1+1)/2)-k2+1)/2)-k3+1)/2), 120)
        return

    def forward(self, x):
        x = unsqueeze(x, 1)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        #x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flat(x) # flatten all dimensions except batch
        x = self.fc1(x)
        #x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        #x = self.fc2(x)
        output = x
        return output