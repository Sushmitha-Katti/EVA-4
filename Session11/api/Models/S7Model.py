import torch.nn as nn
import torch.nn.functional as F

# ********************************************S7 MODEL***************************************************************
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False), #Rf = 3, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p = 0.1),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), #Rf = 5, j = 1
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p = 0.1),

           
            
            )
        self.convblock2 = nn.Sequential(

            #Dilated Network
           nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), padding=1,dilation=2, bias=False), #jout=2, kernel_size = 5, rf = 6+(4)*2 = 14, o/p = 14
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p = 0.1),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False), #jout=2, kernel_size = 3, rf = 14+(2)*2 = 18, o/p =14
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p = 0.1),

            
            )

        self.convblock3 = nn.Sequential(
            #DepthWise Seperable Network
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False, groups = 32), # jout = 4, rf = 20+(2)*4 = 28, o/p = 7
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1, 1), padding=1, bias=False), #jout = 4, rf = 28+(0)*4 = 28, o/p = 9
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p = 0.1),

            #DepthWise Seperable Network

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1,groups=64, bias=False),#jout = 4, rf = 28+(2)*4 = 36, o/p = 9
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p = 0.1),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1, 1), padding=1, bias=False), #jin = jout = 4, rf = 36+(0)*4 = 36, o/p = 11
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Dropout(p = 0.1),
            
            )

        self.convblock4 = nn.Sequential(

            #Dialated Network
          nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1,dilation=2, bias=False), #jout = 8, rf = 40+(4)*8 = 72, o/p = 3
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(p=0.15),


            #AVG Pool

            nn.AdaptiveAvgPool2d(1), #op = 1



           nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)#Op_size = 1, 
            

            
            )



        self.pool = nn.MaxPool2d(2, 2)

      


    def forward(self, x):

        x = self.convblock1(x) # i/p= 32 o/p=32 Rf = 6
        x = self.pool(x) # jout = 2, Rf = 6, O/p = 16 
        x = self.convblock2(x) # Rf = 18 jout = 2, o/p =14
        x = self.pool(x) # jout = 4, s = 2, Rf = 18+1*2 = 20, o/p = 7
        x = self.convblock3(x) # jout = 4, Rf = 36, o/p = 11
        x = self.pool(x) # jout = 8, s = 2, Rf = 36+1*4 = 40 o/p = 5
        x = self.convblock4(x)  # o/p = 1
     
        x = x.view(-1, 10)
      
        return x

