from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, BatchNormalization, \
    AveragePooling2D, Dense, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from facialrec import config

def res_identity(x, filters):
    x_skip = x
    f1, f2 = filters

    # first block
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    #second block
    x = Conv2D(f1,kernel_size=(3,3), strides=(1,1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    #third block
    x = Conv2D(f2, kernel_size=(1,1), strides=(1,1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # adding imput
    x = Add()([x,x_skip])
    x = Activation(activations.relu)(x)

    return x

def res_conv(x, s, filters):

    x_skip = x
    f1, f2 = filters

    #first block
    x = Conv2D(f1, kernel_size=(1,1), strides=(s,s), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)



    #second block
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)

    # shortcut
    x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
    x_skip = BatchNormalization()(x_skip)

    # add
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x

def resnet50(in_shape):

    input_image = Input(shape=in_shape)
    x = ZeroPadding2D(padding=(3,3))(input_image)

    #first round with maxpool
    x = Conv2D(64, kernel_size=(7,7), strides=(2,2))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    #second round
    x = res_conv(x, s=1, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))
    x = res_identity(x, filters=(64, 256))

    #third round
    x = res_conv(x, s=2, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))
    x = res_identity(x, filters=(128, 512))

    #forth round
    x = res_conv(x, s=2, filters=(256,1024))
    x = res_identity(x, filters=(256,1024))
    x = res_identity(x, filters=(256,1024))
    x = res_identity(x, filters=(256,1024))
    x = res_identity(x, filters=(256,1024))
    x = res_identity(x, filters=(256,1024))

    #fift round
    x = res_conv(x, s=2, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))
    x = res_identity(x, filters=(512, 2048))

    x = AveragePooling2D((2,2), padding='same')(x)

    x = Flatten()(x)
    x = Dense(1, activation='sigmoid', kernel_initializer='he_normal')(x)
    model = Model(inputs=input_image, outputs=x)

    return model
