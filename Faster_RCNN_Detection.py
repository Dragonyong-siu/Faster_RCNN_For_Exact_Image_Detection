def Resize(roi, ratio_value):
  xmax = int(roi[0] * ratio_value)
  xmin = int(roi[1] * ratio_value)
  ymax = int(roi[2] * ratio_value)
  ymin = int(roi[3] * ratio_value)
  return (xmax, xmin, ymax, ymin)

def Make_Forms(Tuple):
  x_2, x_1, y_2, y_1 = Tuple
  T_x = (x_1 + x_2) * 0.5
  T_y = (y_1 + y_2) * 0.5
  T_w = (x_2 - x_1)
  T_h = (y_2 - y_1)
  return (T_x, T_y, T_w, T_h)

def Make_Logits(P, D):
  (p_x, p_y, p_w, p_h) = P
  (d_x, d_y, d_w, d_h) = D
  D_u = (p_w * d_x + p_x,
         p_h * d_y + p_y,
         p_w * torch.exp(d_w),
         p_h * torch.exp(d_h))
  return D_u

import cv2
from PIL import Image
import matplotlib.pyplot as plt
class Faster_RCNN_Detection(torch.utils.data.Dataset):
  def __init__(self, data):
    self.data = data
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, index):
    Dictionary = {}
    
    Full_Image_Array = Faster_RCNN_Dataset(self.data)[index]['Full_Image_Array']
    Model_ids = Faster_RCNN_Dataset(self.data)[index]['Model_ids']
    Guide = Faster_RCNN_Dataset(self.data)[index]['Guide_label']
    Roi_box = Faster_RCNN_Dataset(self.data)[index]['Roi_box']
    Roi_ratio = Faster_RCNN_Dataset(self.data)[index]['Roi_ratio']

    # Trained_Faster_RCNN : Model_output
    Reg_logits = Trained_Faster_RCNN(Model_ids)[1]
    Reg_logits = Reg_logits.cpu().detach()

    # True_value
    True_roi = []
    for i in range(len(Roi_box)):
      if Roi_box[i][0] == 1:
        True_roi.append((Roi_box[i][1], Reg_logits[i]))

    # Trained_box
    Trained_box = []
    for j in range(len(True_roi)):
      T = torch.Tensor(Make_Forms(True_roi[j][0]))
      Logit = True_roi[j][1]
      D_u = Make_Logits(T, Logit)
      Trained_box.append(D_u)

    # Ratio_value
    # Resized_box
    Ratio_value = Roi_ratio[0] / Roi_ratio[2]
    Resized_box = []
    for k in range(len(Trained_box)):
      D_roi = Trained_box[k]
      Resized_roi = Resize(D_roi, Ratio_value)
      Resized_box.append(Resized_roi)

    Copy_image = Full_Image_Array.copy()
    Green_line = (125, 255, 51)
    for n in range(len(Resized_box)):
      xmax, xmin, ymax, ymin = Resized_box[n]
      Copy_image = cv2.rectangle(Copy_image,
                                (xmin, ymax),
                                (xmax, ymin),
                                color = Green_line,
                                thickness = 2)
    
    Detection_Image = Copy_image
    Dictionary['Detection_Image'] = Detection_Image
    
    return Dictionary
