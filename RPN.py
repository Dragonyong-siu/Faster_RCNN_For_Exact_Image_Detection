from torchvision.ops import nms
class Region_Proposal_Network(nn.Module):
  def __init__(self, In_Channels = 512, Out_Channels = 512,
               Ratios = [0.5, 1, 2], Anchor_Scales = [8, 16, 32],
               Feat_Stride = 16, Proposal_Creator_Params = {}):
    super(Region_Proposal_Network, self).__init__()
    self.Anchor_Base = Generate_Anchor_Base()
    self.Feat_Stride = Feat_Stride
    self.Proposal_Layer = ProposalCreator(self, Proposal_Creator_Params)
    self.K = self.Anchor_Base.shape[0]
    self.Convolution = nn.Conv2d(In_Channels, Out_Channels,
                                 3, stride = 1, padding = 1).to(device)
    self.Classification = nn.Conv2d(Out_Channels, self.K * 2,
                                    1, stride = 1, padding = 0).to(device)
    self.Regression = nn.Conv2d(Out_Channels, self.K * 4,
                                1, stride = 1, padding = 0).to(device)
   
  def forward(self, Image, Image_size, scale = 1):
    N, C, H, W = Image.shape
    Anchor = Get_Shifted_Anchor(np.array(self.Anchor_Base),
                                      self.Feat_Stride, H, W)
    N_Anchor = Anchor.shape[0] // (H * W)
    Feature_Map = F.relu(self.Convolution(Image))
    
    RPN_Locs = self.Regression(Feature_Map)
    RPN_Locs = RPN_Locs.permute(0, 2, 3, 1).contiguous().view(N, -1, 4)
    RPN_Scores = self.Classification(Feature_Map)
    RPN_Scores = RPN_Scores.permute(0, 2, 3, 1).contiguous()

    RPN_Softmax_Scores = F.softmax(RPN_Scores.view(N, H, W, N_Anchor, 2), dim = 4)
    RPN_FG_Scores = RPN_Softmax_Scores[:, :, :, :, 1].contiguous()
    RPN_FG_Scores = RPN_FG_Scores.view(N, -1)
    RPN_Scores = RPN_Scores.view(N, -1, 2)
    RoIs = []
    RoI_Indices = []
    for i in range(N):
      RoI = self.Proposal_Layer(RPN_Locs[i].cpu().data.numpy(),
                                RPN_FG_Scores[i].cpu().data.numpy(),
                                Anchor,
                                Image_size,
                                scale = scale)
      batch_index = i * np.ones((len(RoI)), dtype = np.int32)
      RoIs.append(RoI)
      RoI_Indices.append(batch_index)
    RoIs = np.concatenate(RoIs, axis = 0)
    RoI_Indices = np.concatenate(RoI_Indices, axis = 0)
    return RPN_Locs, RPN_Scores, RoIs, RoI_Indices, Anchor

import numpy as np
def Get_Shifted_Anchor(anchor_base, feat_stride, height, width):
  Shift_Y = np.arange(0, height * feat_stride, feat_stride)
  Shift_X = np.arange(0, width * feat_stride, feat_stride)
  Shift_X, Shift_Y = np.meshgrid(Shift_X, Shift_Y)
  Shift = np.stack((Shift_Y.ravel(), Shift_X.ravel(),
                    Shift_Y.ravel(), Shift_X.ravel()),
                   axis = 1)
  A = anchor_base.shape[0]
  K = Shift.shape[0]
  anchor = anchor_base.reshape((1, A, 4)) + \
           Shift.reshape((1, K, 4)).transpose((1, 0, 2))
  anchor = anchor.reshape((A * K , 4)).astype(np.float32)                    
  return anchor

class ProposalCreator:
  def __init__(self, 
               parent_model, 
               nms_thresh = 0.7, 
               n_train_pre_nms = 6000,
               n_train_post_nms = 3000, 
               n_test_pre_nms = 3000,
               n_test_post_nms = 150, 
               min_size = 13):
    self.parent_model = parent_model
    self.nms_thresh = nms_thresh
    self.n_train_pre_nms = n_train_pre_nms
    self.n_train_post_nms = n_train_post_nms
    self.n_test_pre_nms = n_test_pre_nms
    self.n_test_post_nms = n_test_post_nms
    self.min_size = min_size

  def __call__(self, loc, score,
               anchor, img_size, scale = 1):
    if self.parent_model.training:
      n_pre_nms = self.n_train_pre_nms
      n_post_nms = self.n_train_post_nms
    else:
      n_pre_nms = self.n_test_pre_nms
      n_post_nms = self.n_test_post_nms
    roi = Loc2Bbox(anchor, loc)
    roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
    roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])

    min_size = self.min_size * scale
    hs = roi[:, 2] - roi[:, 0]
    ws = roi[:, 3] - roi[:, 1]
    keep = np.where((hs >= min_size) & (ws >= min_size))[0]
    roi = roi[keep, :]
    score = score[keep]   
    
    order = score.ravel().argsort()[::-1]
    order = order[:n_pre_nms]
    score = score[order]
    roi = roi[order, :]
    keep = nms(torch.from_numpy(roi).to(device),
               torch.from_numpy(score).to(device),
               0.7)
    if n_post_nms > 0:
      keep = keep[:n_post_nms]
    
    roi = roi[keep.cpu().numpy()]
    return roi

def Loc2Bbox(src_bbox, loc):
  if src_bbox.shape[0] == 0:
    return np.zeros((0, 4), dtype = loc.dtype)
  
  src_bbox = src_bbox.astype(src_bbox.dtype, copy = False)
  src_height = src_bbox[:, 2] - src_bbox[:, 0]
  src_width = src_bbox[:, 3] - src_bbox[:, 1]
  src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
  src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

  dy = loc[:, 0::4]
  dx = loc[:, 1::4]
  dh = loc[:, 2::4]
  dw = loc[:, 3::4]

  ctr_y = dy * src_height[:, np.newaxis] + src_ctr_y[:, np.newaxis]
  ctr_x = dx * src_width[:, np.newaxis] + src_ctr_x[:, np.newaxis]
  h = np.exp(dh) * src_height[:, np.newaxis]
  w = np.exp(dw) * src_width[:, np.newaxis]
  
  dst_bbox = np.zeros(loc.shape, dtype = loc.dtype)
  #(ymin, xmin, ymax, xmax)
  dst_bbox[:, 0::4] = ctr_y - 0.5 * h
  dst_bbox[:, 1::4] = ctr_x - 0.5 * w
  dst_bbox[:, 2::4] = ctr_y + 0.5 * h
  dst_bbox[:, 3::4] = ctr_x + 0.5 * w

  return dst_bbox

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
def Generate_Anchor_Base(base_size = 16, ratios = [0.5, 1, 2], 
                         anchor_scales = [8, 16, 32]):
  py = base_size / 2
  px = base_size / 2

  anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                         dtype = np.float32)
  for i in range(len(ratios)):
    for j in range(len(anchor_scales)):
      h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
      w = base_size * anchor_scales[j] * np.sqrt(1 / ratios[i])

      index = i * len(anchor_scales) + j
      anchor_base[index, 0] = py - h / 2
      anchor_base[index, 1] = px - w / 2
      anchor_base[index, 2] = py + h / 2
      anchor_base[index, 3] = px + w / 2
  return anchor_base

RPN = Region_Proposal_Network().to(device)
