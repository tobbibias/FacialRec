from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D

def build_model(in_shape,embedding_shape=48):
    inputs = Input(shape = in_shape)
    x = Conv2D(64, (2, 2), padding='same', activation='relu')(inputs)
    x =