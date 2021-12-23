import tensorflow as tf
import tensorflow.keras as nn


class RoiPooling(nn.layers.Layer):
    def __init__(self, patch_width, patch_height, **kwargs):
        self.patch_width = patch_width
        self.patch_height = patch_height
        super().__init__(**kwargs)
    

    def call(self, inputs, training=None):
        """
        inputs: (imgs, rois)
        imgs: [batch, height, width, channels]
        rois: [batch, num_rois, 4]
        """
        
        imgs = inputs[0]
        rois = inputs[1]
        roi_batch = tf.shape(rois)[1]

        if rois.dtype == tf.float16:  # presumably crop_and_resize fails with float16
            rois = tf.cast(rois, tf.float32)

        rois = tf.reshape(rois, (-1, 4))
        
        box_indicies = tf.range(0, tf.shape(rois)[0], dtype=tf.int32)
        box_indicies = tf.math.floordiv(box_indicies, roi_batch)

        bbox = tf.image.crop_and_resize(
            image=imgs,
            boxes=rois,
            box_indices=box_indicies,
            crop_size=(self.patch_height, self.patch_width),
            method='nearest',
        )

        return bbox
