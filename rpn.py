import keras.layers as KL


def RPN(base_layers, num_anchors):
    x = KL.Conv2D(512, (3, 3), padding='same', activation='relu',
                  kernel_initializer='normal', name='rpn_conv')(base_layers)

    x_class = KL.Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_class')(x)
    x_regr = KL.Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_regr')(x)

    return [x_class, x_regr, base_layers]
