from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D



#this method will build our model wiht our own parameters.

def build_model(in_shape,embeddingDim=48):
    # our inputs will be the images with 28X28 dim
    inputs = Input(shape=in_shape)

    # the model arcitecture  conv - > relu -> maxpool -> dropout
    x = Conv2D(64, (2, 2), padding='same', activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)


    x = Conv2D(64, (2, 2), padding="same", activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    # prepare the final outputs
    pooledOutput = GlobalAveragePooling2D()(x)
    outputs = Dense(embeddingDim)(pooledOutput)

    # build the model
    model = Model(inputs, outputs)
    # return the model to the calling function
    return model