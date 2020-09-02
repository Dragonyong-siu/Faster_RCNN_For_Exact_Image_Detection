# Faster_RCNN_For_Exact_-Image-Detection

 Faster R-CNN : Towards Real-Time Object Detection with Region Proposal Networks

 Base
  Data : PASCAL_VOC_2012

 0.Faster_RCNN_VGG16 : Feature_Map Extraction

 1.RPN (Region Proposal Network) 
  It has 2 modules : a classifier/ a regressor.
   Classifier - Getting the probability of a proposal having the target object.
   Regressor - Regressing the coordinates of the proposals.
 
  For all image, scale and aspect-ratio are two important parameters.
   The developer chose 3 scale and 3 aspect-ratio (k = 9)
   For Whole Image, there are K * H * W anchors.(if stride == 1)

  Anchor is the central point of the sliding window.
 
  The structure of RPN is a small convolutional neural network. 
   Input the feature map area(M, N)
   Output 2 layer : Cls_layer, Reg_layer(Extracted with 1 * 1 convolution kernal)

  Classification Loss uses Cross-Entropy Function
  Regression Loss uses Distance Functuon
  
  Get Proposals which has 5 values : label & proposal coordinates  

 2.Faster_RCNN_Dataset

 3.Faster_RCNN_Main_Model : Fully_Conection_Model

 4.Faster_RCNN_Loss : L_cls(p, u) + lamda * [u >= 1] * L_loc(t_u, v)

 5.Train(Classification + Bbox_Regression)

**************************************************************************************************
RPN nms : IoU threshold

Region-based CNN nms (class-based nms) : probability threshold

After getting the final objects and ignoring those predicted as background, we apply class-based NMS.
This is done by grouping the objects by class, sorting them by probability and then applying NMS to each independent group before joining them again.
