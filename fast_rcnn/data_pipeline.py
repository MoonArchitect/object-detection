import os
import numpy as np
import tensorflow as tf

from functools import partial
from time import perf_counter



def parse_example(ex):
    """
    Parses example from tfrecord into usable data
    """
    # Shards' features spec
    ex_feature_spec = {
        'id': tf.io.FixedLenFeature(shape=[], dtype=tf.int64),                      # int | [] | img id (ex. 342460)
        'filename': tf.io.FixedLenFeature(shape=[], dtype=tf.string),               # string | [] | img filename (ex. '000000342460.jpg')
        'dims': tf.io.FixedLenFeature(shape=[2], dtype=tf.int64),                   # int | [2] | img size [w, h] (ex. [426, 640])
        'labels': tf.io.VarLenFeature(dtype=tf.int64),                              # int | [n] | labels in range [0, 79]  TODO check if it is 79 or 80
        'bboxes': tf.io.FixedLenFeature(shape=[], dtype=tf.string),                 # str -> float | [n, 4] | bounding boxes [x,y,w,h], normalized from 0 to 1, w/ IoU > 0.5
        'targets': tf.io.FixedLenFeature(shape=[], dtype=tf.string),                # str -> float | [n, 4] | bbox regression targets [tx,ty,tw,th]
        'false_bboxes': tf.io.FixedLenFeature(shape=[], dtype=tf.string)            # str -> float | [n, 4] | background bounding boxes [x,y,w,h], normalized from 0 to 1, w/ 0.1 < IoU < 0.5
    }

    data = tf.io.parse_single_example(serialized = ex, features=ex_feature_spec)


    labels = tf.sparse.to_dense(data['labels'])
    labels = tf.cast(labels, dtype=tf.int32)

    filename = data['filename']
    dims = data['dims']
    bboxes = tf.io.parse_tensor(data['bboxes'], out_type = tf.float32)
    targets = tf.io.parse_tensor(data['targets'], out_type = tf.float32)
    false_bboxes = tf.io.parse_tensor(data['false_bboxes'], out_type = tf.float32)

    return filename, dims, labels, bboxes, targets, false_bboxes


# TODO filter more intelligently
def filter_empty(filename, dims, labels, bboxes, targets, false_bboxes):
    return tf.shape(bboxes)[0] > 1 and tf.shape(false_bboxes)[0] > 1


def random_sample_RoIs(filename, dims, labels, bboxes, targets, false_bboxes, true_frac, RoI_batch_size, imgs_root, resize_dim):
    """
    Randomly samples `RoI_batch_size` number of bounding boxes with `true_frac` true/false ratio from `bboxes` amd `false_bboxes`
    Also reads image from `imgs_root + filename`
    """

    img_path = imgs_root + os.sep + filename

    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, resize_dim)
    img = tf.cast(img, tf.float32) / 255.0

    n_true = tf.shape(bboxes)[0]
    n_false = tf.shape(false_bboxes)[0]

    n_sample_true = tf.cast(RoI_batch_size * true_frac, dtype=tf.int32)
    n_sample_false = RoI_batch_size - n_sample_true

    i_sample_true = tf.random.uniform([n_sample_true], minval=0, maxval=n_true, dtype=tf.int32)
    i_sample_false = tf.random.uniform([n_sample_false], minval=0, maxval=n_false, dtype=tf.int32)

    true_labels_sample = tf.gather(labels, i_sample_true)
    true_targets_sample = tf.gather(targets, i_sample_true)
    true_bboxes_sample = tf.gather(bboxes, i_sample_true)

    false_labels_sample = tf.fill([n_sample_false], 80)
    false_targets_sample = tf.zeros([n_sample_false, 4])
    false_bboxes_sample = tf.gather(false_bboxes, i_sample_false)

    labels_sample = tf.concat( (true_labels_sample, false_labels_sample), axis=0 )
    one_hot_labels_sample = tf.one_hot(labels_sample, 81, 1.0, 0.0, dtype=tf.float32)
    
    targets_sample = tf.concat( (true_targets_sample, false_targets_sample), axis=0 )
    targets_sample = tf.concat( (tf.cast(tf.expand_dims(labels_sample, axis=-1), dtype=tf.float32), targets_sample), axis=-1 )
    targets_sample = tf.expand_dims(targets_sample, axis=1)

    bboxes_sample = tf.concat( (true_bboxes_sample, false_bboxes_sample), axis=0 )

    return (img, bboxes_sample), { "labels": one_hot_labels_sample, "targets": targets_sample }


def val_sampled_RoIs():
    pass


def get_training_pipeline(imgs_root, shards_file, img_batch_size = 16, RoI_batch_size = 64, true_frac = 0.25, resize_dim = [512, 512]):
    threads = 12

    sampling_fn = partial(random_sample_RoIs, true_frac=true_frac, RoI_batch_size=RoI_batch_size, imgs_root=imgs_root, resize_dim=resize_dim)

    # TODO 
    #   test cache()
    #   separate img loading+decoding from random RoI sampling
    #   echo some images multiple times before random RoI sampling to increase throughput

    ds = tf.data.TFRecordDataset(shards_file, num_parallel_reads=2) \
            .map(parse_example, num_parallel_calls=threads) \
            .repeat() \
            .filter(filter_empty) \
            .map(sampling_fn, num_parallel_calls=threads) \
            .batch(img_batch_size, num_parallel_calls=threads) \
            .prefetch(tf.data.AUTOTUNE)

    return ds



def benchmark_dataset(dataset, min_t = 10, min_n = 10):
    """
    Benchmarks the given `dataset` by enumerating it and reports entries/sec
    """
    assert isinstance(dataset, tf.data.Dataset), f"`dataset` should be instance of `tf.data.Dataset`, got {type(dataset)}"
    
    logger = tf.get_logger()
    logger_level = logger.level
    logger.setLevel('ERROR')

    print("Warmup ", end='')
    for i, entry in enumerate(dataset, 1):
        if i % 4 == 0:
            print(".", end='')
        if i > 20:
            break
    
    print(" Done")


    start_time = perf_counter()
    for i, entry in enumerate(dataset, 1):
        end_time = perf_counter()
        if i >= min_n and end_time - start_time > min_t:
            break

    print(f"Perf: {i / (end_time - start_time):>10.2f} entries/sec. Sample {end_time - start_time:.2f} s.")

    logger.setLevel(logger_level)


def compute_label_distribution(dataset):
    """
    `dataset` yields -> ((imgs, rois), { "labels": labels, "targets": regression_targets })
    imgs: [batch_size, None, None, 3]
    rois: [batch_size, roi_batch, 4]
    labels: [batch_size, roi_batch, 81]
    """
    assert 2 == 3, "Update w/ respect to pipeline's output"

    distribution = np.ones([81], dtype='int32')
    start = perf_counter()

    for i, (inputs, targets) in enumerate(dataset, 1):
        labels = np.reshape(targets['labels'], [-1, 81])
        distribution += np.bincount(np.argmax(labels, axis=-1), minlength=81)

        if i % 100 == 0:
            print(f"Processed {i} samples, {np.sum(distribution)} labels in {perf_counter() - start:.2f}s.")
        if i > 7500:
            break

    distribution = 1.0 / distribution
    distribution /= np.sum(distribution)
    return distribution
    