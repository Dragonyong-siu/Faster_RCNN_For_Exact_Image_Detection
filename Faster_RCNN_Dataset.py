import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import Normalize
class Faster_RCNN_Dataset(torch.utils.data.Dataset):
  def __init__(self, Data):
    self.Data = Data
  
  def __len__(self):
    return len(self.Data)

  def __getitem__(self, index):
    Dictionary = {}
    # PIL_Image
    PIL_Image = self.Data[index][0]
    Annotation = self.Data[index][1]['annotation']
    Normalization = torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                     (0.229, 0.224, 0.225))

    # Original_size
    # Full_Image_Array
    Objects = Annotation['object']
    Size_Tuple = Annotation['size']
    Original_Size = (int(Size_Tuple['depth']),
                     int(Size_Tuple['height']),
                     int(Size_Tuple['width']))
    Image_Array = np.asarray(PIL_Image)
    Full_Image_Array = Image_Array.copy()
    Full_Image_Tensor = torch.Tensor(Full_Image_Array).reshape(3,
                                                               Original_Size[1],
                                                               Original_Size[2])
    Full_Image_Tensor = Normalization(Full_Image_Tensor)
    Full_Image_Tensor = Full_Image_Tensor.reshape(Original_Size[1],
                                                  Original_Size[2],
                                                  3)

    # Ground_Truths
    Ground_Truths = []
    Names = []
    for i in range(len(Objects)):
      bndbox = Objects[i]['bndbox']
      xmax = int(bndbox['xmax'])
      xmin = int(bndbox['xmin'])
      ymax = int(bndbox['ymax'])
      ymin = int(bndbox['ymin'])
      ground_truth = (xmax, xmin, ymax, ymin)
      name = Objects[i]['name']
      Ground_Truths.append(ground_truth)
      Names.append(name)

    # Feature_Map
    Input_Tensor = Full_Image_Tensor.to(device)
    Input_View = Input_Tensor.view(-1, 3, Original_Size[1], Original_Size[2])
    Feature_Map = Faster_RCNN_Extractor(Input_View)

    # Regions_of_Interests(RoIs)
    _, Orig_Height, Orig_Width = Original_Size
    _, Conv_Height, Conv_Width = Feature_Map.squeeze(0).shape
    rois = RPN(Feature_Map.to(device), (Orig_Width, Orig_Height))
    RoIs = rois[2]
    Regions_of_Interests = []
    for candidate in RoIs:
      bottom = (candidate[0] * Conv_Height) / Orig_Height
      left = (candidate[1] * Conv_Width) / Orig_Width
      top = (candidate[2] * Conv_Height) / Orig_Height
      right = (candidate[3] * Conv_Width) / Orig_Width
      roi = (int(right), int(left), int(top), int(bottom))
      if int(right) > int(left) and int(top) > int(bottom):
        Regions_of_Interests.append(roi)
  
    # Model_ids
    Key_Number = 8
    Model_Ids = RoI_Pooling(Feature_Map, Regions_of_Interests, (7, 7))
    
    T_ids = []
    T_label = []
    T_GT = []
    T_Box = []
    T_target = []
    T_guide = []
    
    F_ids = []
    F_label = []
    F_GT = []
    F_Box = []
    F_target = []
    F_guide = []
    for i in range(len(Ground_Truths)):
      Xmax, Xmin, Ymax, Ymin = Ground_Truths[i]
      Resized_GT = (int((Xmax * Conv_Width) / Orig_Width),
                    int((Xmin * Conv_Width) / Orig_Width),
                    int((Ymax * Conv_Height) / Orig_Height),
                    int((Ymin * Conv_Height) / Orig_Height))
      for bi, RoI in enumerate(Regions_of_Interests):
        if Compute_IoU(RoI, Resized_GT) > 0.5:
          model_label = Encoding(Names[i])
          T_ids.append(Model_Ids[bi].unsqueeze(0).to(device))
          T_label.append(model_label)
          T_target.append((Make_targets(RoI, Resized_GT)).tolist())
          T_guide.append(1)
          if len(T_ids) >= Key_Number * 7 / 8:
            break
          
        elif Compute_IoU(RoI, Resized_GT) >= 0.0 and Compute_IoU(RoI, Resized_GT) < 0.1:
          model_label = Encoding('background')
          F_ids.append(Model_Ids[bi].unsqueeze(0).to(device))
          F_label.append(model_label)
          F_target.append((Make_targets(RoI, Resized_GT)).tolist())
          F_guide.append(0)

    # Cls_label
    # Reg_label
    if len(T_ids) > Key_Number:
      Model_ids = T_ids[:Key_Number]
      Cls_label = T_label[:Key_Number]
      Reg_label = T_target[:Key_Number]
      Guide_label = T_guide[:Key_Number]

    else:
      F_num = Key_Number - len(T_ids)
      F_ids = F_ids[:F_num]
      F_label = F_label[:F_num]
      F_target = F_target[:F_num] 
      F_guide = F_guide[:F_num]
    
      Model_ids = T_ids + F_ids
      Cls_label = T_label + F_label
      Reg_label = T_target + F_target
      Guide_label = T_guide + F_guide
    
    while len(Model_ids) < Key_Number:
      Model_ids.append(Model_ids[0])
      Cls_label.append(Cls_label[0])
      Reg_label.append(Reg_label[0])
      Guide_label.append(Guide_label[0])
    # Shuffle Input
    Model_INPUT = []
    for i in range(Key_Number):
      INPUT_Tuple = (Model_ids[i], Cls_label[i], Reg_label[i], Guide_label[i])
      Model_INPUT.append(INPUT_Tuple)

    import random
    random.shuffle(Model_INPUT)
    Model_ids = []
    Cls_label = []
    Reg_label = []
    Guide_label = []
    for i in range(Key_Number):
      Model_ids.append(Model_INPUT[i][0])
      Cls_label.append(Model_INPUT[i][1])
      Reg_label.append(Model_INPUT[i][2])
      Guide_label.append(Model_INPUT[i][3])

    # Dictionary['Full_Image_Tensor '] = Full_Image_Tensor 
    # Dictionary['Original_size'] = Original_size 
    # Dictionary['Ground_Truths'] = Ground_Truths
    # Dictionary['Feature_Map'] = Feature_Map
    # Dictionary['Regions_of_Interests(RoIs)'] = Regions_of_Interests(RoIs)
    Dictionary['Full_Image_Tensor'] = Full_Image_Tensor
    Dictionary['Model_ids'] = torch.cat(Model_ids)
    Dictionary['Cls_label'] = torch.Tensor(Cls_label)
    Dictionary['Reg_label'] = torch.Tensor(Reg_label)
    Dictionary['Guide_label'] = torch.Tensor(Guide_label)
    
    return Dictionary

def RoI_Pooling(feature_map, rois, size):
  Pooled_RoI = []
  rois_num = len(rois)
  for i in range(rois_num):
    roi = rois[i]
    Right, Left, Top, Bottom = roi
    Cut_Feature_Map = feature_map[:, :, Left:Right, Bottom:Top]
    Fixed_Feature_Map = F.adaptive_max_pool2d(Cut_Feature_Map, size)
    Pooled_RoI.append(Fixed_Feature_Map)

  return torch.cat(Pooled_RoI)

def Compute_IoU(CD_box, GT_box):
  X_1 = np.maximum(CD_box[1], GT_box[1])
  X_2 = np.minimum(CD_box[0], GT_box[0])
  Y_1 = np.maximum(CD_box[3], GT_box[3])
  Y_2 = np.minimum(CD_box[2], GT_box[2])

  Intersection = np.maximum(X_2 - X_1, 0) * np.maximum(Y_2 - Y_1, 0)
  CD_area = (CD_box[0] - CD_box[1]) * (CD_box[2] - CD_box[3])
  GT_area = (GT_box[0] - GT_box[1]) * (GT_box[2] - GT_box[3])
  Union = CD_area + GT_area - Intersection
  IoU = Intersection / Union

  return IoU

def Make_targets(P, G):
  p_r, p_l, p_t, p_b = torch.Tensor(P).to(device)
  g_r, g_l, g_t, g_b = torch.Tensor(G).to(device)
  t_x = ((g_l + g_r) * 0.5 + (p_l + p_r) * 0.5) / (p_r - p_l)
  t_y = ((g_b + g_t) * 0.5 + (p_b + p_t) * 0.5) / (p_t - p_b)
  t_w = torch.log(g_r - g_l) - torch.log(p_r - p_l)
  t_h = torch.log(g_t - g_b) - torch.log(p_t - p_b)
  return torch.Tensor([t_x, t_y, t_w, t_h])

from torch.utils.data import DataLoader
Train_dataloader = DataLoader(Faster_RCNN_Dataset(train_data_4),
                              batch_size = 1,
                              shuffle = True,
                              drop_last = True)
