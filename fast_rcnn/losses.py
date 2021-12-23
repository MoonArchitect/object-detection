import tensorflow as tf


class SmoothL1Loss(tf.keras.losses.Loss):
    """
    call(y_true, y_pred):
        y_true: [batch_size, n_rois, 1, 5] | float | [:, :, 0, 0] - class id in [0,80] | [:, :, 0, 1:5] - regression target
        y_pred: [batch_size, n_rois, 81, 4] | float
    """
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        labels = tf.cast(y_true[:, :, 0, 0], tf.int32)
        labels = tf.reshape(labels, [-1, 1])
        targets = y_true[:, :, 0, 1:5]
        targets = tf.reshape(targets, [-1, 4])

        regressions = tf.reshape(y_pred, [-1, 81, 4])
        regressions = tf.gather_nd(regressions, labels, batch_dims=1)
        
        diff = regressions - targets
        
        smooth_l1 = tf.where(tf.abs(diff) < 1, 0.5 * tf.square(diff), tf.abs(diff) - 0.5)
        smooth_l1 = tf.where(labels == 80, smooth_l1 * 0.0, smooth_l1)

        return tf.reduce_mean(smooth_l1)



class cocoClassLoss(tf.keras.losses.Loss):
    """
    COCO Class Weighted Categorical Crossentropy
    """
    def __init__(self):
        super().__init__()

        # TODO recompute and cache distribution
        self.class_weights = tf.constant([
            1.91648865e-04, 6.48112199e-03, 1.57583735e-03, 3.86938738e-03,
            2.49354714e-03, 3.31384001e-03, 2.07259137e-03, 2.51549187e-03,
            3.36149946e-03, 5.59265706e-03, 5.73046880e-03, 6.68817609e-03,
            1.65989901e-02, 2.91489858e-03, 3.29950977e-03, 2.85672327e-03,
            3.57496932e-03, 3.28893102e-03, 4.16773126e-03, 3.31884329e-03,
            3.13184242e-03, 6.11171606e-03, 3.04527436e-03, 2.44306914e-03,
            7.23029076e-03, 4.46438988e-03, 7.16353063e-03, 7.75307212e-03,
            4.79005825e-03, 1.02026282e-02, 8.15291264e-03, 1.35351976e-02,
            1.35530803e-02, 5.02933937e-03, 2.65373909e-02, 2.15911092e-02,
            5.98183120e-03, 6.67946627e-03, 6.20954955e-03, 5.35209175e-03,
            1.95435188e-02, 4.84129957e-03, 1.49511875e-02, 1.34152256e-02,
            2.00268075e-02, 4.29312173e-03, 6.51634548e-03, 1.10434908e-02,
            6.09195531e-03, 1.01165442e-02, 6.49708535e-03, 1.12397836e-02,
            1.14801769e-02, 3.56833698e-03, 1.06769496e-02, 6.99792184e-03,
            1.79036172e-03, 4.05514924e-03, 5.46973060e-03, 3.15420508e-03,
            2.34353554e-03, 2.25647276e-03, 3.66348886e-03, 6.00400486e-03,
            3.94376688e-02, 1.59757022e-02, 1.21863230e-02, 1.14801769e-02,
            1.28262827e-02, 6.05599100e-03, 1.78230855e-01, 3.48154396e-03,
            7.05460249e-03, 5.96636828e-03, 2.82758821e-03, 4.94237595e-03,
            1.90908980e-02, 5.14911226e-03, 2.02908973e-01, 2.94681081e-02,
            1.60262602e-05
        ], dtype=tf.float32)
        # TODO make ^ data type adaptive to input data

    
    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1, 81])
        y_pred = tf.reshape(y_pred, [-1, 81])
        
        weights = tf.gather(self.class_weights, tf.math.argmax(y_true, -1))
        weights = tf.cast(weights, y_true.dtype)
        
        losses = tf.keras.metrics.categorical_crossentropy(y_true, y_pred, from_logits=True, label_smoothing=0.0) * weights
        return tf.reduce_mean(losses)

