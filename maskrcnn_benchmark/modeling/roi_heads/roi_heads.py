import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .maskiou_head.maskiou_head import build_roi_maskiou_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, proposals, targets=None, disps=None):
        losses = {}
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON:
            mask_features = features
            if (
                self.training
                and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR
            ):
                mask_features = x

            ### Implement the maskhead and maskiou head here
            if not self.cfg.MODEL.MASKIOU_ON:
                x, detections, loss_mask = self.mask(mask_features, detections, targets, disps)
                losses.update(loss_mask)
            else:
                if self.training:
                    x, detections, loss_mask, p1, p2, p3 = self.mask(mask_features, detections, targets, disps)
                    losses.update(loss_mask)

                    loss_maskiou, detections = self.maskiou(detections, p1, p2, p3)
                    losses.update(loss_maskiou)
                else:
                    x, detections, _, f1, f2, f3 = self.mask(mask_features, detections, targets, disps)
                    if x is False: # dummy box and return false.
                        return False, (), ()
                    if f2 is not None:
                        _, detections = self.maskiou.forward_eval(detections, f1, f2, f3)
                    else:
                        _, detections = self.maskiou.forward_eval_single(detections, f1, f2, f3)

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
        if cfg.MODEL.MASKIOU_ON:
            roi_heads.append(("maskiou", build_roi_maskiou_head(cfg)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)
    return roi_heads
