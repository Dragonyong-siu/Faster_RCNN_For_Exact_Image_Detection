# 0.Fast_RCNN_VGG16 : Feature Extraction
device = 'cuda'
import torch.nn as nn
from collections import OrderedDict
class Faster_RCNN_VGG16(nn.Module):
  def __init__(self):
    super(Faster_RCNN_VGG16, self).__init__()
    self.Convolution_Layer = nn.Sequential(
        OrderedDict([('Conv_1', nn.Conv2d(3, 64, 3, stride = 1, padding = 1)),
                     ('Batch_Norm_1', nn.BatchNorm2d(64)),
                     ('ReLU_1', nn.ReLU(inplace = True)),
                     ('Conv_2', nn.Conv2d(64, 64, 3, stride = 1, padding = 1)),
                     ('Batch_Norm_2', nn.BatchNorm2d(64)),
                     ('ReLU_2', nn.ReLU(inplace = True)),
                     ('MaxPool_1', nn.MaxPool2d(2, stride = 2)),

                     ('Conv_3', nn.Conv2d(64, 128, 3, stride = 1, padding = 1)),
                     ('Batch_Norm_3', nn.BatchNorm2d(128)),
                     ('ReLU_3', nn.ReLU(inplace = True)),
                     ('Conv_4', nn.Conv2d(128, 128, 3, stride = 1, padding = 1)),
                     ('Batch_Norm_4', nn.BatchNorm2d(128)),
                     ('ReLU_4', nn.ReLU(inplace = True)),
                     ('MaxPool_2', nn.MaxPool2d(2, stride = 2)),

                     ('Conv_5', nn.Conv2d(128, 256, 3, stride = 1, padding = 1)),
                     ('Batch_Norm_5', nn.BatchNorm2d(256)),
                     ('ReLU_5', nn.ReLU(inplace = True)),
                     ('Conv_6', nn.Conv2d(256, 256, 3, stride = 1, padding = 1)),
                     ('Batch_Norm_6', nn.BatchNorm2d(256)),
                     ('ReLU_6', nn.ReLU(inplace = True)),
                     ('Conv_7', nn.Conv2d(256, 256, 3, stride = 1, padding = 1)),
                     ('Batch_Norm_7', nn.BatchNorm2d(256)),
                     ('ReLU_7', nn.ReLU(inplace = True)),
                     ('MaxPool_3', nn.MaxPool2d(2, stride = 2)),

                     ('Conv_8', nn.Conv2d(256, 512, 3, stride = 1, padding = 1)),
                     ('Batch_Norm_8', nn.BatchNorm2d(512)),
                     ('ReLU_8', nn.ReLU(inplace = True)),
                     ('Conv_9', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
                     ('Batch_Norm_9', nn.BatchNorm2d(512)),
                     ('ReLU_9', nn.ReLU(inplace = True)),
                     ('Conv_10', nn.Conv2d(512, 512, 3, stride = 1, padding = 1))]))
                     #('Batch_Norm_10', nn.BatchNorm2d(512)),
                     #('ReLU_10', nn.ReLU(inplace = True))]))
                     #('Maxpool_4', nn.MaxPool2d(2, stride = 2))]))
    nn.init.normal_(self.Convolution_Layer._modules['Conv_1'].weight, std=0.01)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_1'].bias, 0)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_2'].weight, std=0.001)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_2'].bias, 0)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_3'].weight, std=0.01)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_3'].bias, 0)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_4'].weight, std=0.001)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_4'].bias, 0)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_5'].weight, std=0.01)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_5'].bias, 0)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_6'].weight, std=0.001)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_6'].bias, 0)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_7'].weight, std=0.01)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_7'].bias, 0)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_8'].weight, std=0.001)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_8'].bias, 0)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_9'].weight, std=0.01)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_9'].bias, 0)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_10'].weight, std=0.001)
    nn.init.normal_(self.Convolution_Layer._modules['Conv_10'].bias, 0)
  def forward(self, Image):
    Conv_Image = self.Convolution_Layer(Image)
    return Conv_Image

Faster_RCNN_Extractor = Faster_RCNN_VGG16().to(device)
