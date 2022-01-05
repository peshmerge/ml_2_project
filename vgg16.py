import keras.layers as KL
import keras.backend as K


def VGG16(input_tensor=None):
    input_shape = (None, None, 3)

    if input_tensor is None:
        x = KL.Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            x = KL.Input(tensor=input_tensor, shape=input_shape)
        else:
            x = input_tensor

    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1')(x)
    x = KL.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='pooling1')(x)

    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1')(x)
    x = KL.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='pooling2')(x)

    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1')(x)
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2')(x)
    x = KL.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='pooling3')(x)

    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1')(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2')(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3')(x)
    x = KL.MaxPooling2D((2, 2), strides=(2, 2), name='pooling4')(x)

    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_1')(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_2')(x)
    x = KL.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv5_3')(x)

    return x
