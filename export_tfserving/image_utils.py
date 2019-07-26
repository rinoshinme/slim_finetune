import tensorflow as tf

VGG_MEAN = [104, 117, 123]
VGG_MEAN_FLOAT = [121.55213, 113.84197, 99.5037]
IMAGENET_MEAN = tf.constant([121.55213, 113.84197, 99.5037], dtype=tf.float32)


def load_base64_tensor(_input):
    def decode_and_process(base64):
        _bytes = tf.decode_base64(base64)
        _image = __tf_jpeg_process2(_bytes)
        return _image
    image = tf.map_fn(decode_and_process, _input,
                      back_prop=False, dtype=tf.float32)
    return image


def __tf_jpeg_process2(data):
    image = tf.image.decode_jpeg(data, channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_images(image, (224, 224))
    image = tf.subtract(image, IMAGENET_MEAN)
    return image


def __tf_jpeg_process(data):
    image = tf.image.decode_jpeg(data, channels=3, fancy_upscaling=True,
                                 dct_method='INTEGER_FAST')
    image = tf.image.convert_image_dtype(image, tf.float32, saturate=True)
    image = tf.image.resize_images(image, (256, 256),
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   align_corners=True)

    image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)

    image = tf.image.encode_jpeg(image, format='', quality=75,
                                 progressive=False, optimize_size=False,
                                 chroma_downsampling=True,
                                 density_unit=None,
                                 x_density=None, y_density=None,
                                 xmp_metadata=None)

    image = tf.image.decode_jpeg(image, channels=3,
                                 fancy_upscaling=False,
                                 dct_method="INTEGER_ACCURATE")

    image = tf.cast(image, dtype=tf.float32)

    image = tf.image.crop_to_bounding_box(image, 16, 16, 224, 224)

    image = tf.reverse(image, axis=[2])
    image -= VGG_MEAN

    return image
