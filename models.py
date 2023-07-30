import torch.nn as nn

class SimpleModel(nn.Module):
    """
    simple network to classify ducks and geese
    input is an RGB 224x224 image
    """
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 30, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(30))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels = 30, out_channels = 40, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            nn.BatchNorm2d(40))
      
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels = 40, out_channels = 60, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.layer4 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=25*25*60, out_features = 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.Linear(100, out_features = 2))

    def forward(self, x):
        # use operator fusion to save the time of storing values in memory
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))