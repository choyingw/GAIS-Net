# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN_2B(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN_2B, self).__init__()

        self.backbone_rgb = build_backbone(cfg, in_ch = 3)
        self.backbone_d = build_backbone(cfg, in_ch = 1)

        self.rpn = build_rpn(cfg, self.backbone_rgb.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone_rgb.out_channels)
        self.sml_trans = nn.Conv2d(512,256,3)

    def forward(self, images, targets=None, disps=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        ## two image sizes

        #print(disps.size())
        images = to_image_list(images)
        #image_sizes

        s0l, s1l = 0, 0
        for m in range(len(images.image_sizes)):
            s0,s1 = list(images.image_sizes[m])[0], list(images.image_sizes[m])[1]
            if s0 > s0l:
                s0l = s0
            if s1 > s1l:
                s1l = s1

        if disps is not None:
            disps_tensors = torch.zeros(len(disps),1,s0l,s1l).cuda()
            for k in range(len(disps)):
                #print("@@@", list(disps[k].size())[1])
                #print("###", list(disps[k].size())[2])
                disps_tensors[k,:,:list(disps[k].size())[1],:list(disps[k].size())[2]] = disps[k]

        features_rgb = self.backbone_rgb(images.tensors)
        features_d = self.backbone_d(disps_tensors)
        #print(features_rgb[0].type())
        #print(features_d[0].type())

        #sml_trans = nn.Conv2d(512,256,3).cuda()
        #nn.init.kaiming_uniform_(sml_trans.weight, a=1)


        #print(len(features_rgb))
        #print(features_rgb[2].size())

        #print(features_d.size())


        f_trans = []
        for p in range(len(features_rgb)):
            f = self.sml_trans(torch.cat((features_rgb[p],features_d[p]), dim=1))
            f_trans.append(f)        
        features = tuple(f_trans)

        #print(features[0].size())
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result
