import keras.layers as KL
import keras.backend as K
import tensorflow as tf


class RoiPooling(KL.Layer):
    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois

        super(RoiPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nchannels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.nchannels, self.pool_size, self.pool_size

    def call(self, inputs, *args, **kwargs):
        im, rois = inputs[0], inputs[1]

        outputs = []
        for i in range(self.num_rois):
            roi = rois[0][i]
            x, y, w, h = roi[0], roi[1], roi[2], roi[3]

            x = K.cast(x, 'int32')
            y = K.cast(y, 'int32')
            w = K.cast(w, 'int32')
            h = K.cast(h, 'int32')

            rs = tf.image.resize(im[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nchannels))
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))
        return final_output

    def get_config(self):
        config = {'pool_size': self.pool_size, 'num_rois': self.num_rois}
        base_config = super(RoiPooling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
