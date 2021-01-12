import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from maskrcnn_benchmark.layers.misc import interpolate

from maskrcnn_benchmark.structures.bounding_box import BoxList

class MaskIoUPostProcessor(nn.Module):
    """
    Getting the maskiou according to the targeted label, and computing the mask score according to maskiou.
    """

    def __init__(self):
        super(MaskIoUPostProcessor, self).__init__()

    def forward(self, boxes, pred_maskiou, labels):
        num_masks = pred_maskiou.shape[0]
        index = torch.arange(num_masks, device=labels.device)
        maskious = pred_maskiou[index, labels]
        maskious = [maskious]
        results = []
        for maskiou, box in zip(maskious, boxes):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                bbox.add_field(field, box.get_field(field))
            bbox_scores = bbox.get_field("scores")
            mask_scores = bbox_scores * maskiou
            bbox.add_field("mask_scores", mask_scores)
            results.append(bbox)

        return results

def make_roi_maskiou_post_processor(cfg):
    maskiou_post_processor = MaskIoUPostProcessor()
    return maskiou_post_processor

def softmax(x,y):
    """Compute softmax values for each sets of scores in x."""
    sx = x.shape[0]
    a = torch.cuda.FloatTensor(sx).fill_(0)
    b = torch.cuda.FloatTensor(sx).fill_(0)
    m = torch.stack((a,b),1)
    for i in range(sx):
        tmp = torch.exp((m[i] - torch.max(m[i]))*20)
        e_m = tmp/ torch.sum(tmp)
        a[i], b[i] = e_m[0], e_m[1]
    return a, b


class MaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, masker=None):
        super(MaskPostProcessor, self).__init__()
        self.masker = masker

    def forward(self, x, boxes, x2=None, x3=None, m1=None, m2=None, m3=None):
        """
        Maskscoring fusion of different modalities.
        """
        mask_prob1 = [x]
        mask_prob2 = [x2]
        mask_prob3 = [x3]

        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        a = [i for i in range(len(labels))]

        ## Maskscores. Using the following zip() to extract them.
        m1 = [m1[a,labels]]
        m2 = [m2[a,labels]]
        m3 = [m3[a,labels]]

        results = []

        for prob1, prob2, prob3, box, ms1, ms2, ms3 in zip(mask_prob1, mask_prob2, mask_prob3, boxes, m1, m2, m3):
            # 1: image, 2: disparity, 3: pc
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                if (field == "mask1") or (field == "mask2") or (field == "mask3"):
                    continue
                bbox.add_field(field, box.get_field(field))

            # 2.5D/3D combine first
            w2, w3 = ms2/(ms2+ms3), ms3/(ms2+ms3)
            
            # then combine with 2D
            ms23 = ms2*w2+ms3*w3
            k1, k23 = ms1/(ms23+ms1), ms23/(ms1+ms23)
            
            w2 = w2.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            w3 = w3.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            k1_ = k1.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            k23_ = k23.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            bbox.add_field("mask", ((prob3*w3 + prob2*w2)*k23_ + prob1*k1_))
            bbox_scores = bbox.get_field("scores")
            factor = (ms1*k1+ms23*k23)
            mask_scores = bbox_scores * factor
            bbox.add_field("mask_scores", mask_scores)

            results.append(bbox)

        return results

    def forward_single(self, x, boxes, m1=None):
        """
        Single forward of mask scores.
        """
        mask_prob = [x]

        labels = [bbox.get_field("labels") for bbox in boxes]
        labels = torch.cat(labels)
        a = [i for i in range(len(labels))]

        ## Maskscores !!!
        m1 = [m1[a,labels]]

        results = []

        for prob, box, ms1 in zip(mask_prob, boxes, m1):
            bbox = BoxList(box.bbox, box.size, mode="xyxy")
            for field in box.fields():
                if (field == "mask1") or (field == "mask2") or (field == "mask3"):
                    continue
                bbox.add_field(field, box.get_field(field))

            bbox_scores = bbox.get_field("scores")
            factor = ms1
            mask_scores = bbox_scores * factor
            bbox.add_field("mask_scores", mask_scores)

            results.append(bbox)

        return results

class MaskPostProcessorCOCOFormat(MaskPostProcessor):
    """
    From the results of the CNN, post process the results
    so that the masks are pasted in the image, and
    additionally convert the results to COCO format.
    """

    def forward(self, x, boxes):
        import pycocotools.mask as mask_util
        import numpy as np

        results = super(MaskPostProcessorCOCOFormat, self).forward(x, boxes)
        for result in results:
            masks = result.get_field("mask").cpu()
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
            result.add_field("mask", rles)
        return results


# the next two functions should be merged inside Masker
# but are kept here for the moment while we need them
# temporarily gor paste_mask_in_image
def expand_boxes(boxes, scale):
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = torch.zeros_like(boxes)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half
    return boxes_exp


def expand_masks(mask, padding):
    N = mask.shape[0]
    M = mask.shape[-1]
    pad2 = 2 * padding
    scale = float(M + pad2) / M
    padded_mask = mask.new_zeros((N, 1, M + pad2, M + pad2))

    padded_mask[:, :, padding:-padding, padding:-padding] = mask
    return padded_mask, scale


def paste_mask_in_image(mask, box, im_h, im_w, thresh=0.5, padding=1):
    # Need to work on the CPU, where fp16 isn't supported - cast to float to avoid this
    mask = mask.float()
    box = box.float()

    padded_mask, scale = expand_masks(mask[None], padding=padding)
    mask = padded_mask[0, 0]
    box = expand_boxes(box[None], scale)[0]
    box = box.to(dtype=torch.int32)

    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    mask = mask.to(torch.float32)
    mask = interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
    mask = mask[0][0]

    if thresh >= 0:
        mask = mask > thresh
    else:
        # for visualization and debugging, we also
        # allow it to return an unmodified mask
        mask = (mask * 255).to(torch.uint8)

    im_mask = torch.zeros((im_h, im_w), dtype=torch.uint8)
    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]) : (y_1 - box[1]), (x_0 - box[0]) : (x_1 - box[0])
    ]
    return im_mask


class Masker(object):
    """
    Projects a set of masks in an image on the locations
    specified by the bounding boxes
    """

    def __init__(self, threshold=0.5, padding=1):
        self.threshold = threshold
        self.padding = padding

    def forward_single_image(self, masks, boxes):
        boxes = boxes.convert("xyxy")
        im_w, im_h = boxes.size
        res = [
            paste_mask_in_image(mask[0], box, im_h, im_w, self.threshold, self.padding)
            for mask, box in zip(masks, boxes.bbox)
        ]
        if len(res) > 0:
            res = torch.stack(res, dim=0)[:, None]
        else:
            res = masks.new_empty((0, 1, masks.shape[-2], masks.shape[-1]))
        return res

    def __call__(self, masks, boxes):
        if isinstance(boxes, BoxList):
            boxes = [boxes]

        # Make some sanity check
        assert len(boxes) == len(masks), "Masks and boxes should have the same length."

        # TODO:  Is this JIT compatible?
        # If not we should make it compatible.
        results = []
        for mask, box in zip(masks, boxes):
            assert mask.shape[0] == len(box), "Number of objects should be the same."
            result = self.forward_single_image(mask, box)
            results.append(result)
        return results


def make_roi_mask_post_processor(cfg):
    if cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS:
        mask_threshold = cfg.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD
        masker = Masker(threshold=mask_threshold, padding=1)
    else:
        masker = None
    mask_post_processor = MaskPostProcessor(masker)
    return mask_post_processor
