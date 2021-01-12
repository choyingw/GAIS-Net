import torch
from torch import nn

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .roi_maskiou_feature_extractors import make_roi_maskiou_feature_extractor
from .roi_maskiou_predictors import make_roi_maskiou_predictor
from .inference import make_roi_maskiou_post_processor
from .loss import make_roi_maskiou_loss_evaluator

from .inference import make_roi_mask_post_processor

class ROIMaskIoUHead(torch.nn.Module):
    def __init__(self, cfg):
        super(ROIMaskIoUHead, self).__init__()
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_maskiou_feature_extractor(cfg)
        self.predictor = make_roi_maskiou_predictor(cfg)
        self.post_processor = make_roi_maskiou_post_processor(cfg)
        self.loss_evaluator = make_roi_maskiou_loss_evaluator(cfg)
        self.post_processor_mask = make_roi_mask_post_processor(cfg)

    def forward(self, proposals, p1, p2, p3):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            selected_mask (list[Tensor]): targeted mask
            labels (list[Tensor]): class label of mask
            maskiou_targets (list[Tensor], optional): the ground-truth maskiou targets.

        Returns:
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
            results (list[BoxList]): during training, returns None. During testing, the predicted boxlists are returned.
                with the `mask` field set
        """
        if p1[0].shape[0] == 0 and not self.training:
            return {}, proposals

        # Only using the 2D part to train the maskscoring module.

        # 2D
        f_2D = self.feature_extractor(p1[0], p1[1])
        pred_maskiou_2D = self.predictor(f_2D)

        # 25D
        # f_25D = self.feature_extractor(p2[0], p2[1])
        # pred_maskiou_25D = self.predictor(f_25D)

        # 3D
        # f_3D = self.feature_extractor(p3[0], p3[1])
        # pred_maskiou_3D = self.predictor(f_3D)

        loss_maskiou_2D = self.loss_evaluator(p1[2], pred_maskiou_2D, p1[3])
        return dict(loss_maskiou=0.01*(loss_maskiou_2D)), None

    def forward_eval(self, proposals, f1, f2, f3):

        if f1[0].shape[0] == 0 and not self.training:
            return {}, proposals

        # Maskscore prediction of multiple representations.

        # 2D
        f_2D = self.feature_extractor(f1, proposals[0].get_field("mask1"))
        pred_maskiou_2D = self.predictor(f_2D)

        # 25D
        f_25D = self.feature_extractor(f2,  proposals[0].get_field("mask2"))
        pred_maskiou_25D = self.predictor(f_25D)

        # 3D
        f_3D = self.feature_extractor(f3, proposals[0].get_field("mask3"))
        pred_maskiou_3D = self.predictor(f_3D)

        result = self.post_processor_mask(proposals[0].get_field("mask1"), proposals, x2=proposals[0].get_field("mask2"),
                 x3=proposals[0].get_field("mask3"), m1=pred_maskiou_2D, m2=pred_maskiou_25D, m3=pred_maskiou_3D)

        return {}, result

        def forward_eval_single(self, proposals, f1):

            if f1[0].shape[0] == 0 and not self.training:
                return {}, proposals

            # 2D
            f_2D = self.feature_extractor(f1, proposals[0].get_field("mask"))
            pred_maskiou_2D = self.predictor(f_2D)
            result = self.post_processor_mask.forward_single(proposals[0].get_field("mask"), proposals, m1=pred_maskiou_2D)
            return {}, result

def build_roi_maskiou_head(cfg):
    return ROIMaskIoUHead(cfg)
