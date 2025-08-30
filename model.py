from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def convolution_operation(entered_input, filters=64):
    conv1 = Conv2D(filters, kernel_size=(3,3), padding="same")(entered_input)
    bn1 = BatchNormalization()(conv1)
    act1 = ReLU()(bn1)

    conv2 = Conv2D(filters, kernel_size=(3,3), padding="same")(act1)
    bn2 = BatchNormalization()(conv2)
    act2 = ReLU()(bn2)
    return act2

def encoder(entered_input, filters=64):
    enc = convolution_operation(entered_input, filters)
    pool = MaxPooling2D(strides=(2,2))(enc)
    return enc, pool

def decoder(entered_input, skip, filters=64):
    up = Conv2DTranspose(filters, (2, 2), strides=2, padding="same")(entered_input)
    concat = Concatenate()([up, skip])
    out = convolution_operation(concat, filters)
    return out

def get_unet_model(input_shape=(160,160,3), num_classes=3):
    input_tensor = Input(input_shape)
    skip1, enc1 = encoder(input_tensor, 64)
    skip2, enc2 = encoder(enc1, 128)
    skip3, enc3 = encoder(enc2, 256)
    skip4, enc4 = encoder(enc3, 512)

    bridge = convolution_operation(enc4, 1024)

    dec1 = decoder(bridge, skip4, 512)
    dec2 = decoder(dec1, skip3, 256)
    dec3 = decoder(dec2, skip2, 128)
    dec4 = decoder(dec3, skip1, 64)

    output = Conv2D(num_classes, 1, padding="same", activation="softmax")(dec4)
    model = Model(input_tensor, output)
    return model
