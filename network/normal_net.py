# imported from: https://github.com/raghakot/keras-resnet

from __future__ import division

import keras

from core import BaseNetwork


class NormalNet(BaseNetwork):

    def __init__(self, **kwargs):
        super(NormalNet, self).__init__(**kwargs)

    def build(self, **kwargs):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(256, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self._input_shape))
        model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(self._num_outputs, activation='softmax'))
        # Return a keras Model architecture.
        return model