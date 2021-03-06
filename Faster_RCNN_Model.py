# 3.Faster_RCNN_Main_Model : Fully_Conection_Model
class Fully_Conection_Model(nn.Module):
  def __init__(self):
    super(Fully_Conection_Model, self).__init__()
    self.Classification_Layer = nn.Sequential(
        OrderedDict([('Linear_1', nn.Linear(512 * 7 * 7, 4096)),
                     ('ReLU_1', nn.ReLU(inplace = True)),
                     #('Dropout_1', nn.Dropout(0.1)),
  
                     #('Linear_2', nn.Linear(4096, 4096)),
                     #('ReLU_2', nn.ReLU(inplace = True)),
                     #('Dropout_2', nn.Dropout(0.1)),

                     ('Linear_3', nn.Linear(4096, 4096)),
                     ('ReLU_3', nn.ReLU(inplace = True)),
                     ('Dropout_3', nn.Dropout(0.1)),
    
                     ('Linear_4', nn.Linear(4096, 21))]))
    
    self.Regression_Layer = nn.Sequential(
        OrderedDict([('Linear_1', nn.Linear(512 * 7 * 7, 4096)),
                     ('ReLU_1', nn.ReLU(inplace = True)),
                     #('Dropout_1', nn.Dropout(0.1)),
    
                     #('Linear_2', nn.Linear(4096, 4096)),
                     #('ReLU_2', nn.ReLU(inplace = True)),
                     #('Dropout_2', nn.Dropout(0.1)),

                     ('Linear_3', nn.Linear(4096, 4096)),
                     ('ReLU_3', nn.ReLU(inplace = True)),
                     ('Dropout_3', nn.Dropout(0.1)),
    
                     ('Linear_4', nn.Linear(4096, 4))]))
    
    nn.init.normal_(self.Classification_Layer._modules['Linear_1'].weight, std = 0.01)
    nn.init.normal_(self.Classification_Layer._modules['Linear_1'].bias, 0)
    nn.init.normal_(self.Classification_Layer._modules['Linear_2'].weight, std = 0.001)
    nn.init.normal_(self.Classification_Layer._modules['Linear_2'].bias, 0)
    nn.init.normal_(self.Classification_Layer._modules['Linear_3'].weight, std = 0.01)
    nn.init.normal_(self.Classification_Layer._modules['Linear_3'].bias, 0)
    nn.init.normal_(self.Classification_Layer._modules['Linear_4'].weight, std = 0.001)
    nn.init.normal_(self.Classification_Layer._modules['Linear_4'].bias, 0)
    nn.init.normal_(self.Regression_Layer._modules['Linear_1'].weight, std = 0.01)
    nn.init.normal_(self.Regression_Layer._modules['Linear_1'].bias, 0)
    nn.init.normal_(self.Regression_Layer._modules['Linear_2'].weight, std = 0.01)
    nn.init.normal_(self.Regression_Layer._modules['Linear_2'].bias, 0)
    nn.init.normal_(self.Regression_Layer._modules['Linear_3'].weight, std = 0.01)
    nn.init.normal_(self.Regression_Layer._modules['Linear_3'].bias, 0)
    nn.init.normal_(self.Regression_Layer._modules['Linear_4'].weight, std = 0.01)
    nn.init.normal_(self.Regression_Layer._modules['Linear_4'].bias, 0)
    
    #self.Linear_1 = nn.Linear(512 * 7 * 7, 4096, bias = True)
    #self.Linear_2 = nn.Linear(4096, 4096, bias = True)
    #self.Linear_3 = nn.Linear(4096, 4096, bias = True)
    #self.Linear_4 = nn.Linear(4096, 4096, bias = True)
    #self.Linear_5 = nn.Linear(4096, 4096, bias = True)
    
    #self.Linear_Cls = nn.Linear(4096, 21, bias = True)
    #self.Linear_Reg = nn.Linear(4096, 4, bias = True)
    
    #self.ReLU = nn.ReLU(inplace = True)
    #self.LeakyReLU = nn.LeakyReLU(inplace = True) 
    #self.Dropout = nn.Dropout(0.1, inplace = False)
    
    #nn.init.normal_(self.Linear_Cls.weight, std = 0.01)
    #nn.init.normal_(self.Linear_Cls.bias, 0)
    #nn.init.normal_(self.Linear_Reg.weight, std = 0.01)
    #nn.init.normal_(self.Linear_Reg.bias, 0)
    #nn.init.normal_(self.Linear_1.weight, std = 0.01)
    #nn.init.normal_(self.Linear_1.bias, 0)
    #nn.init.normal_(self.Linear_2.weight, std = 0.01)
    #nn.init.normal_(self.Linear_2.bias, 0)
    #nn.init.normal_(self.Linear_3.weight, std = 0.01)
    #nn.init.normal_(self.Linear_3.bias, 0)
    #nn.init.normal_(self.Linear_4.weight, std = 0.01)
    #nn.init.normal_(self.Linear_4.bias, 0)
    #nn.init.normal_(self.Linear_5.weight, std = 0.01)
    #nn.init.normal_(self.Linear_5.bias, 0)
    
  def forward(self, Model_INPUT):
    Flatten_INPUT = Model_INPUT.view(-1, 512 * 7 * 7)
    Classification_Output = self.Classification_Layer(Flatten_INPUT)
    Bbox_Regression_Output = self.Regression_Layer(Flatten_INPUT)
    
    #FC_1 = self.Linear_1(Flatten_INPUT)
    #FC_2 = self.ReLU(FC_1)
    #FC_3 = self.Linear_2(FC_2)
    #FC_4 = self.ReLU(FC_3)
    #FC_5 = self.Linear_3(FC_4)
    #FC_6 = self.ReLU(FC_5)
    #FC_7 = self.Linear_4(FC_6)
    #FC_8 = self.ReLU(FC_7)
    #FC_9 = self.Linear_5(FC_8)
    #FC_10 = self.ReLU(FC_9)
    #FC_11 = self.Dropout(FC_10)
    #Classification_Output = self.Linear_Cls(FC_6)
    #Bbox_Regression_Output = self.Linear_Reg(FC_6)
    
    return Classification_Output, Bbox_Regression_Output

Faster_RCNN_Model = Fully_Conection_Model().to(device) 
