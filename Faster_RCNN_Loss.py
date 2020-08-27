def Faster_RCNN_Loss(Cls_Logit, Cls_Label, Reg_Logit, Reg_Label, 
                     Guide_Label, Ncls = 1, Nreg = 1, lamda = 10):
  CLS_Function = nn.CrossEntropyLoss()
  REG_Function = nn.SmoothL1Loss(reduction = 'none')
  CLS_Loss = CLS_Function(Cls_Logit, Cls_Label) 
  REG_Out =  REG_Function(Reg_Logit, Reg_Label)
  REG_Loss = lamda * Zero_background(REG_Out, Guide_Label)
  return CLS_Loss, REG_Loss
