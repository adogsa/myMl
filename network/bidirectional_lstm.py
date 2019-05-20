# imported from: https://github.com/raghakot/keras-resnet

from __future__ import division

from core import BaseNetwork
import keras


class Bidirectional_LSTM(BaseNetwork):

    def __init__(self, **kwargs):
        super(Bidirectional_LSTM, self).__init__(**kwargs)

    def build(self, **kwargs):
        model = keras.models.Sequential()
        model.add(keras.layer.Bidirectional(keras.layer.LSTM(10, return_sequences=True),
                                input_shape=(5, 10)))
        model.add(keras.layer.Bidirectional(keras.layer.LSTM(10)))
        model.add(keras.layer.Dense(5))
        model.add(keras.layer.Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        return model