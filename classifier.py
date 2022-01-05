import keras.layers as KL
from roi_pooling import RoiPooling


def Classifier(base_layers, input_rois, num_rois, nclasses=2):
    pooling_regions = 7

    y = RoiPooling(pooling_regions, num_rois)([base_layers, input_rois])

    y = KL.TimeDistributed(KL.Flatten(name='flatten'))(y)
    y = KL.TimeDistributed(KL.Dense(4096, activation='relu', name='fc1'))(y)
    y = KL.TimeDistributed(KL.Dropout(.5))(y)
    y = KL.TimeDistributed(KL.Dense(4096, activation='relu', name='fc2'))(y)
    y = KL.TimeDistributed(KL.Dropout(.5))(y)

    y_class = KL.TimeDistributed(KL.Dense(nclasses, activation='softmax', kernel_initializer='zero'), name='dense_class')(y)
    y_regr = KL.TimeDistributed(KL.Dense(4 * (nclasses - 1), activation='linear', kernel_initializer='zero'), name='dense_regr')(y)

    return [y_class, y_regr]