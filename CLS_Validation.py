def Evaluation_Index(dataloader, model, device):
    #model.eval()
    Labels = []
    Objects = []
    with torch.no_grad():
      for bi, Dictionary in enumerate(dataloader):
        Model_ids = Dictionary['Model_ids']
        Cls_label = Dictionary['Cls_label']
        Image_IDS = Model_ids.squeeze(0).to(device)
        Image_LABEL = Cls_label.squeeze(0).to(device)
        Logits_Prob = model(Image_IDS)[0].cpu()
        print(Logits_Prob)
        Object = np.argmax(Logits_Prob, axis = 1)
        Labels.extend(Image_LABEL.cpu().detach().numpy().tolist())
        Objects.extend(Object.cpu().detach().numpy().tolist())
        
    return Labels, Objects
##Get output of function##
from torch.utils.data import DataLoader
Valid_dataloader = DataLoader(Faster_RCNN_Dataset(train_data_4),
                              batch_size = 1,
                              shuffle = True,
                              drop_last = True)
Labels, Objects = Evaluation_Index(Valid_dataloader, Trained_Faster_RCNN, device)
print(Labels)
print(Objects)
##Calculating the accuarcy##
from sklearn.metrics import accuracy_score
accuracy_score(Labels, Objects)
