# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from maskrcnn_benchmark import _C
import sys
try:
    sys.path.append('/usr/local/lib/python3.5/dist-packages/apex-0.1-py3.5-linux-x86_64.egg')
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')

# Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
