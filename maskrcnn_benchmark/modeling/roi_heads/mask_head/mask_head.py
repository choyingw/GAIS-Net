import torch
from torch import nn
from torch.nn import functional as F
from maskrcnn_benchmark.structures.bounding_box import BoxList
import math
import time

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator
from .loss import Smoothness
from .loss_miou import make_roi_mask_loss_evaluator_miou

from ._3D2D_utils import *

def disps_mask_pc(boxList,disps,num_pts):
    """
    Convert to pseudo-lidar repr.
    """
    batch_num = len(boxList)
    pcs, coords = [], []
    for i in range(batch_num):
        boxes = boxList[i]
        disp_image = torch.squeeze(disps[i],dim=0)
        pc, coord = box_extractor(boxes, disp_image, num_pts)
        pcs.append(pc)
        coords.append(coord)
    return pcs, coords

def disps_mask(boxList,disps,num_pts):
    """
    Crop out the masked areas from disparity maps.
    """
    batch_num = len(boxList)
    side = int(math.sqrt(num_pts))
    n = 0
    for i in range(batch_num):
        n += len(boxList[i])
    if n == 0:
        return False
    disp_coll = torch.cuda.FloatTensor(n, 1, side, side).fill_(0)
    count = 0
    for j in range(batch_num):
        disp_image = torch.squeeze(disps[j],dim=0) 
        x1,y1,x2,y2 = boxList[j]._split_into_xyxy()
        x1,y1,x2,y2 = x1.int(), y1.int(),\
                     x2.int(), y2.int()

        if len(x1.shape) == 0: ## For the no-box handle: a dummy box
            x1, x2, y1, y2 = np.array([0]), np.array([0]), np.array([0]), np.array([0])
        
        for k in range(y1.size()[0]):
            if ((x2[k]-x1[k])==0) or ((y2[k]-y1[k])==0):
                a = disp_image[y1[k]:y1[k]+1, x1[k]:x1[k]+1].unsqueeze(0).unsqueeze(0)
                print("catch a dummy box.")
            else:
                a = disp_image[y1[k]:y2[k],x1[k]:x2[k]].unsqueeze(0).unsqueeze(0)
            a = F.interpolate(a, (side, side), mode='bilinear')
            disp_coll[count] = a.squeeze(0)
            count += 1 
    return disp_coll

def keep_only_positive_boxes(boxes):
    """
    Given a set of BoxList containing the `labels` field,
    return a set of BoxList for which `labels > 0`.

    Arguments:
        boxes (list of BoxList)
    """
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    positive_inds = []
    num_boxes = 0
    for boxes_per_image in boxes:
        labels = boxes_per_image.get_field("labels")
        inds_mask = labels > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
        positive_inds.append(inds_mask)
    return positive_boxes, positive_inds

class ROIMaskHead_miou(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead_miou, self).__init__()
        self.cfg = cfg.clone()
        self.backbone_ch = in_channels
        self.feature_extractor = make_roi_mask_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)
        self.resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        self.num_pts = 1024
        self.segNet_p25 = p25DSeg(self.num_pts)
        self.segNet_3D = InstanceSeg(self.num_pts)
        self.predictor_25D = make_roi_mask_predictor(cfg, self.feature_extractor.out_channels)
        self.predictor_3D = make_roi_mask_predictor(cfg, self.feature_extractor.out_channels)
        self.smoothness = Smoothness()
        self.loss_evaluator_miou = make_roi_mask_loss_evaluator_miou(cfg)


    def forward(self, features, proposals, targets=None, disps=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals
            proposals, positive_inds = keep_only_positive_boxes(proposals)

        ## 2.5D part
        disp_coll = disps_mask(proposals, disps, self.num_pts)
        if disp_coll is False: # dummy box and return false.
            return False, (), () ,() ,(), ()
        out = self.segNet_p25(disp_coll)
        feat25ds = F.interpolate(out, (self.resolution, self.resolution), mode='bilinear')

        ## 3D part
        pcs, coords = disps_mask_pc(proposals, disps, self.num_pts)
        batch_num = len(pcs)
        pc_number_total = 0
        for pc in pcs:
            pc_number_total += len(pc)
        count = 0
        input_prepare = torch.cuda.FloatTensor(pc_number_total, 3, self.num_pts).fill_(0)
        for i, (pc, proposal, coord) in enumerate(zip(pcs, proposals, coords)):
            for n in range(pc.size()[0]): # Num of boxes
                ppc = pc[n]
                ppc = ppc.unsqueeze(0)
                input_prepare[count] = ppc            
                count += 1
        out = self.segNet_3D(input_prepare)
        batch, channel = out.size()[0], out.size()[1]
        feat3ds = torch.reshape(out,(batch, channel, int(math.sqrt(self.num_pts)), int(math.sqrt(self.num_pts))))
        feat3ds = feat3ds.permute(0,1,3,2)
        feat3ds = F.interpolate(feat3ds, (self.resolution, self.resolution), mode='bilinear')

        # features: 5 (FPN)x size(batch, 256, H/16, W/16)
        # proposals(list): batchsize -> BoxLists

        ## 2D part
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x, roi_feature = self.feature_extractor(features, proposals)

        mask_logits = self.predictor(x)
        mask_logits_25D = self.predictor_25D(feat25ds)
        mask_logits_3D = self.predictor(feat3ds)

        if not self.training:
            result = self.post_processor(mask_logits, proposals, x2=mask_logits_25D, x3=mask_logits_3D)
            return x, result, {}, roi_feature, feat25ds, feat3ds 

        positives, labels = self.loss_evaluator.only_return_ind(proposals, targets)
        if positives.size()[0] == 0:
            loss_corr_25D3D = 0
        else:
            loss_corr_25D3D = 0.1*F.binary_cross_entropy_with_logits(mask_logits_25D[positives, labels],
                     mask_logits_3D[positives, labels])

        loss_mask, selected_mask, labels, maskiou_targets = self.loss_evaluator_miou(proposals, mask_logits, targets)
        loss_mask_25D, selected_mask_25D, labels_25D, maskiou_targets_25D = self.loss_evaluator_miou(proposals, mask_logits_25D, targets)
        loss_mask_3D, selected_mask_3D, labels_3D, maskiou_targets_3D = self.loss_evaluator_miou(proposals, mask_logits_3D, targets)
        
        loss_smooth_3D = 1.2*self.smoothness(mask_logits_3D)

        p1 = (roi_feature, selected_mask, labels, maskiou_targets, mask_logits)
        p2 = (feat25ds, selected_mask_25D, labels_25D, maskiou_targets_25D, mask_logits_25D)
        p3 = (feat3ds, selected_mask_3D, labels_3D, maskiou_targets_3D, mask_logits_3D)

        return x, all_proposals, dict(loss_mask=1*loss_mask, loss_mask_3D=0.1*loss_mask_3D,
                 loss_mask_25D=0.1*loss_mask_25D, loss_mask_25D3D=0.025*loss_corr_25D3D,
                 loss_smooth_3D=1*loss_smooth_3D), p1, p2, p3


def build_roi_mask_head(cfg, in_channels):
    if cfg.MM == "miou" and cfg.MODEL.MASKIOU_ON:
        return ROIMaskHead_miou(cfg, in_channels)
    else:
        raise NotImplementedError("The default mode (MM) is miou. If you're not using the 2D/2.5D/3D repr with mask scoring, please write your customized class.")
        