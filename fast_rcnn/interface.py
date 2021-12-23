import numpy as np
import tensorflow as tf

from .model import FastRCNN

""" 
Interface with the Fast R-CNN model
"""


def vec_get_adjusted_bbox(bbox, correction):
    """
    bbox: [batch_size, 4] -> (x, y, w, h)
    corr: [batch_size, 4] -> (tx, ty, tw, th)
    Returns:
    bbox: [batch_size, 4] -> (y_start, x_start, y_end, x_end); where *_start < *_end and (0,0) is top-left corner 
    """
    x, y, w, h = np.transpose(bbox)
    tx, ty, tw, th = np.transpose(correction)

    cx = tx * w + x + w * 0.5
    cy = ty * h + y + h * 0.5

    nw = np.exp(tw) * w
    nh = np.exp(th) * h

    nx = cx - nw * 0.5
    ny = cy - nh * 0.5

    nbbox = np.stack((ny, nx, ny + nh, nx + nw))
    nbbox = np.transpose(nbbox)

    return nbbox


# TODO allow non-square images
# TODO return score:float
# TODO assert (or not, since used internally) imgs are of the same size with 3 channels matching model's input 
# TODO if roi_batch is None -> compute it
# TODO cache backbone output from img batches -> iterate sub-batches of RoIs (pad batch and extract if needed) 
def inference_batch(model: FastRCNN, img_batch: np.ndarray, rois_batch: np.ndarray = None, max_output_size=10, score_threshold=0.8, IoU_threshold=0.5):
    """
    Processes a batch of images of the same resolution matching model's input resolution

    Returns: [batch_size, n_bboxes, 2] -> (class_id:int, bbox:[4]float), where bbox is list [x, y, w, h] normalized from 0 to 1
    """                                             

    bbox_batch = []

    for img, rois in zip(img_batch, rois_batch):
        outputs = model((tf.expand_dims(img, 0), tf.expand_dims(rois, 0)))

        labels = outputs['labels'][0]
        targets = outputs['targets'][0].numpy()
        
        labels = tf.math.softmax(np.squeeze(labels), axis=1).numpy()
        classes = np.argmax(labels, axis=-1)

        objects = [(i, class_idx) for i, class_idx in enumerate(classes) if class_idx != 80 and labels[i, class_idx] > score_threshold]

        if len(objects) == 0:
            # print("No objects found for")
            bbox_batch.append([])
            continue

        rois *= model.img_dims[:2] * 2
        
        rois = [rois[i] for i, class_idx in objects]
        targets = [targets[i, class_idx] for i, class_idx in objects]
        scores = [labels[i, class_idx] for i, class_idx in objects]

        rois = vec_get_adjusted_bbox(rois, targets) # returns [y,x,y,x]

        selected = tf.image.non_max_suppression(rois, scores, max_output_size=max_output_size, iou_threshold=IoU_threshold).numpy()
        bboxes = [(rois[i], objects[i][1]) for i in selected]

        bbox_batch.append(bboxes)
    
    return bbox_batch
