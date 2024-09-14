import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import *
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
import torchvision
from torchvision import transforms

# define the 3D CNN
num_classes = 12

# Create CNN Model
class CNNModel(nn.Module):
    def __init__(self, in_c_conv, hidden_c_conv, out_c_conv, in_c_linear, hidden_c2=12):
        super(CNNModel, self).__init__()
        
        self.conv_layer1 = self._conv_layer_set(in_c_conv, hidden_c_conv)
        self.conv_layer2 = self._conv_layer_set(hidden_c_conv, out_c_conv)
        self.drop=nn.Dropout(p=0.50)
        self.fc1 = nn.Linear(in_c_linear, hidden_c2) 
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_c2, num_classes)
        
    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=0),
        nn.ReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.drop(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.classifier(out)
        
        return out
    
def create_model(num_channels=3, hidden_c_conv=7, out_c_conv=9, in_c_linear=36):
    model = (
        CNNModel(
            in_c_conv=num_channels, 
            hidden_c_conv=hidden_c_conv, 
            out_c_conv=out_c_conv, 
            in_c_linear=in_c_linear, 
            hidden_c2=12)
    )
    return model

def load_in_model(model_path):
    newmodel = create_model()
    newmodel.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return newmodel
