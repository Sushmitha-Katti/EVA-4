import torch.nn as nn
import torch.nn.functional as F

# ****************************************************S11 MODEL(New Resnet)************************************************************


class NewResnet(nn.Module):
    def __init__(self):
        super(NewResnet, self).__init__()

        self.preplayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(),  
            )

        self.x1 = nn.Sequential(

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            )
        self.R1 = nn.Sequential(

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            )
        self.layer2 = nn.Sequential(

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1, bias=False), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        
            )
        self.x2 = nn.Sequential(

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        
            )

        self.R2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=1, bias=False), 
            nn.BatchNorm2d(512),
            nn.ReLU(),
            )

        
        self.pool = nn.MaxPool2d(4, 4)

        self.fc = nn.Linear(in_features = 512, out_features = 10, bias=False)

      


    def forward(self, x):

        preplayer = self.preplayer(x) 
        x1 = self.x1(preplayer)
        R1 = self.R1(x1)
        layer1 = x1+R1
        layer2 = self.layer2(layer1)
        x2 = self.x2(layer2)
        R2 = self.R2(x2)
        layer3 = R2+x2
        maxpool = self.pool(layer3)
        x = maxpool.view(maxpool.size(0),-1)
        fc = self.fc(x)

        return F.log_softmax(fc.view(-1,10), dim=-1)
      