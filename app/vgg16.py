from torch.nn import Module, Sequential, Linear, Dropout, ReLU, Flatten
from torch.nn import Conv2d, BatchNorm2d, MaxPool2d
from torch.nn import init

class VGG16_V1(Module):
    def __init__(self, input_channels: int, output_classes: int):
        super().__init__()
        #size = (224, 224)
        self.conv_block_1 = Sequential(
            Conv2d(in_channels=input_channels,
                   out_channels=64,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(64),
            ReLU(),
            Conv2d(in_channels=64,
                   out_channels=64,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2),
            Dropout(0.2)      
        )
        #size = (112, 112)
        self.conv_block_2 = Sequential(
            Conv2d(in_channels=64,
                   out_channels=128,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(128),
            ReLU(),
            Conv2d(in_channels=128,
                   out_channels=128,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2),     
            Dropout(0.2) 
        )
         #size = (56, 56)
        self.conv_block_3 = Sequential(
            Conv2d(in_channels=128,
                   out_channels=256,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256,
                   out_channels=256,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(256),
            ReLU(),
            Conv2d(in_channels=256,
                   out_channels=256,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(256),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2),   
            Dropout(0.2)    
        )
        #size = (28, 28)
        self.conv_block_4 = Sequential(
            Conv2d(in_channels=256,
                   out_channels=512,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512,
                   out_channels=512,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512,
                   out_channels=512,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2),   
            Dropout(0.2)    
        )
        #size = (14, 14)
        self.conv_block_5 = Sequential(
            Conv2d(in_channels=512,
                   out_channels=512,
                   kernel_size=3,
                   padding=1,
                   stride=1), 
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512,
                   out_channels=512,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(512),
            ReLU(),
            Conv2d(in_channels=512,
                   out_channels=512,
                   kernel_size=3,
                   padding=1,
                   stride=1),
            BatchNorm2d(512),
            ReLU(),
            MaxPool2d(kernel_size=2,
                      stride=2), 
            Dropout(0.2)  
        )
        #size = (7, 7)
        self.classifier = Sequential(
            Flatten(),
            Linear(in_features=512*7*7,
                    out_features=4096),
            ReLU(),
            Dropout(p=0.5),
            Linear(in_features=4096,
                    out_features=4096),
            ReLU(),
            Dropout(p=0.5),
            Linear(in_features=4096,
                    out_features=output_classes)
        )
        self.apply(self.init_weights)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.conv_block_4(x)
        x = self.conv_block_5(x)
        x = self.classifier(x)
        return x
    
    def init_weights(self, m):
        if type(m) == Conv2d:
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
        elif type(m) == Linear:
            init.xavier_normal_(m.weight)