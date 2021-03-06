def Zero_background(Reg_Logit, Guide_Label):
  Guide_Label = Guide_Label.to(device)
  Part_Mean_Logit = torch.mean(Reg_Logit, dim = 1)
  Zero_Mean_Logit = Part_Mean_Logit * Guide_Label
  Mean_Logit = torch.mean(Zero_Mean_Logit)
  return Mean_Logit

from tqdm import tqdm
import random
def Train_Epoch(dataloader, model, optimizer, device):
  model.train()
  Book = tqdm(dataloader, total = len(dataloader))
  total_CLS_loss = 0.0
  total_REG_loss = 0.0
  for bi, Dictionary in enumerate(Book):
    Model_ids = Dictionary['Model_ids']
    Cls_label = Dictionary['Cls_label']
    Reg_label = Dictionary['Reg_label']
    Guide_label = Dictionary['Guide_label']
    model_ids = Model_ids.squeeze(0).to(device)
    cls_label = Cls_label.long().squeeze(0).to(device)
    reg_label = Reg_label.squeeze(0).to(device)
    guide_label = Guide_label.squeeze(0).to(device)
      
    model.zero_grad()
    Logits = model(model_ids) 
    Loss = Faster_RCNN_Loss(Logits[0],
                            cls_label,
                            Logits[1],
                            reg_label,
                            guide_label)
    CLS_Loss, REG_Loss = Loss[0], Loss[1]
    CLS_Loss.backward(retain_graph = True)
    REG_Loss.backward()#retain_graph = True)
    optimizer.step()
    #optimizer.zero_grad()
    total_CLS_loss += CLS_Loss.item()
    total_REG_loss += REG_Loss.item()
    
    #del model_ids
    #del cls_label
    #del reg_label
    #del guide_label
    #del Logits
  
  Average_CLS_Loss = total_CLS_loss / len(dataloader)
  Average_REG_Loss = total_REG_loss / len(dataloader)
  print(" Average CLS Loss: {0:.2f}".format(Average_CLS_Loss))
  print(" Average REG Loss: {0:.2f}".format(Average_REG_Loss))

def FIT(model, Epochs, Learning_Rate):
  #optimizer = torch.optim.SGD(model.parameters(),
  #                            lr = Learning_Rate,
  #                            momentum = 0.9,
  #                            dampening = 0,
  #                            weight_decay = 0.00005,
  #                            nesterov = False)
  optimizer = torch.optim.AdamW(model.parameters(), lr = Learning_Rate)
  for i in range(Epochs):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    Train_Epoch(Train_dataloader, model, optimizer, device)
    torch.save(model, '/content/gdrive/My Drive/' + f'Faster_RCNN_Model:{i+1}')
    
FIT(Faster_RCNN_Model, Epochs = 20, Learning_Rate = 0.00003)
