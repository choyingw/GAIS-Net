# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN
from .generalized_rcnn_disp import GeneralizedRCNN_DISP
from .generalized_rcnn_2B import GeneralizedRCNN_2B
from .generalized_rcnn_3D2D import GeneralizedRCNN_3D2D


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN,\
						 		 "GeneralizedRCNN_DISP": GeneralizedRCNN_DISP,\
						 		 "GeneralizedRCNN_2B": GeneralizedRCNN_2B,\
						 		 "GeneralizedRCNN_3D2D": GeneralizedRCNN_3D2D}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
