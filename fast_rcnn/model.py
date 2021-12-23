import tensorflow as tf
import tensorflow.keras as nn

from .RoiPooling import RoiPooling
from .losses import cocoClassLoss, SmoothL1Loss
from tensorflow.keras.applications import ResNet50V2, VGG16


class FastRCNN(nn.Model):
    """
    FastRCNN model
    """

    def __init__(self, img_dims, *args, **kwargs):
        assert isinstance(img_dims, (list, tuple))
        
        super().__init__(*args, **kwargs)

        self.img_dims = img_dims
        if len(img_dims) == 2:
            self.img_dims = [*img_dims, 3]

        # VGG16 backbone
        # self.backbone = VGG16(include_top=False, input_shape=(512, 512, 3), weights='imagenet')  # 
        # self.backbone = nn.Model(inputs=self.backbone.input, outputs=self.backbone.layers[17].output, name="backbone")  # 13 - 64x64, 17 - 32x32

        # ResNet50V2 backbone
        self.backbone = ResNet50V2(include_top=False, input_shape=self.img_dims)
        self.backbone = nn.Model(inputs=self.backbone.input, outputs=self.backbone.layers[141].output, name="backbone")
        
        self.RoiPooling = RoiPooling(7, 7)
        
        self.common_features = nn.layers.Dense(512, name="stem_features")
        self.features_bn = nn.layers.BatchNormalization(name="stem_bn")

        self.regression_layer = nn.layers.Dense(81 * 4, name="roi_regressor")
        self.classification_layer = nn.layers.Dense(81, name="roi_classification")


    def compile(self,
                classification_loss_weight=1.0,
                bbox_regression_loss_weight=0.001):
        """
        Configures the model for training with Adam optimizer. 
            COCO class weighted CategoricalCrossentropy used for classification loss. 
            SmoothL1Loss is used for bbox regression loss. 
        """
        super().compile(
            optimizer = nn.optimizers.Adam(),
            loss = {
                'labels': cocoClassLoss(),
                'targets': SmoothL1Loss(),
            },
            loss_weights={
                'labels': classification_loss_weight, 
                'targets': bbox_regression_loss_weight
            },
            metrics = {
                'labels': 'accuracy'
            },
        )

    # def build
    # TODO assert input is correct

    def call(self, inputs, training=None):
        images = inputs[0]
        rois = inputs[1]

        roi_shape = tf.shape(rois)
        batch = roi_shape[0]
        roi_batch = roi_shape[1]

        x = self.backbone(images)
        
        x = self.RoiPooling((x, rois))
        
        x = self.features_bn(x)
        x = tf.nn.swish(x)
        x = self.common_features(x)
        # flatten
        x = tf.reshape(x, [batch * roi_batch, -1])


        classes = self.classification_layer(x)
        regressions = self.regression_layer(x)

        classes = tf.reshape(classes, (batch, roi_batch, 81, 1))

        regressions = tf.reshape(regressions, (batch, roi_batch, 81, 4))
        
        return { "labels": classes, "targets": regressions }



