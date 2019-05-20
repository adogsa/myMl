# imported from: https://github.com/raghakot/keras-resnet

from __future__ import division

import keras

from core import BaseNetwork
import numpy as np
import pandas as pd
from network.lib import ResnetBuilder
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix


class FewShotNet(BaseNetwork):
    def build(self,  **kwargs):
        dropout = kwargs.get('dropout', 0.2)

        # Input pair inputs.
        pair_1st = keras.Input(shape=self._input_shape)
        pair_2nd = keras.Input(shape=self._input_shape)

        # Siamese Model.
        net = keras.models.Sequential()

        # 1st layer (64@10x10)
        # net.add(keras.layers.Conv2D(filters=64, kernel_size=(10, 10),
        net.add(keras.layers.Conv2D(filters=64, kernel_size=(10, 10),
                                    input_shape=self._input_shape,
                                    activation='relu'))
        net.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # 2nd layer (128@7x7)
        net.add(keras.layers.Conv2D(filters=128, kernel_size=(7, 7),
                                    activation='relu'))
        net.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # 3rd layer (128@4x4)
        net.add(keras.layers.Conv2D(filters=128, kernel_size=(4, 4),
                                    activation='relu'))
        net.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # 4th layer (265@4x4)
        net.add(keras.layers.Conv2D(filters=256, kernel_size=(4, 4),
                                    activation='relu'))
        net.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        # 5th layer  (9216x4096)
        net.add(keras.layers.Flatten())
        net.add(keras.layers.Dense(units=4096, activation='sigmoid'))

        # Call the Sequential model on each input tensors with shared params.
        encoder_1st = net(pair_1st)
        encoder_2nd = net(pair_2nd)

        # Layer to merge two encoded inputs with the l1 distance between them.
        distance_layer = keras.layers.Lambda(self.dist_func)

        # Call this layer on list of two input tensors.
        distance = distance_layer([encoder_1st, encoder_2nd])

        # Model prediction: if image pairs are of same letter.
        output_layer = keras.layers.Dense(2, activation='sigmoid')
        # output_layer = keras.layers.Dense(1, activation='sigmoid')
        outputs = output_layer(distance)

        # Return a keras Model architecture.
        return keras.Model(inputs=[pair_1st, pair_2nd], outputs=outputs)

    # def build(self,  **kwargs):
    #     dropout = kwargs.get('dropout', 0.2)
    #
    #     # Input pair inputs.
    #     pair_1st = keras.Input(shape=self._input_shape)
    #     pair_2nd = keras.Input(shape=self._input_shape)
    #
    #     # # Return a keras Model architecture.
    #     # return ResnetBuilder.build_resnet_34(self._input_shape, self._num_outputs)
    #
    #     # net = ResnetBuilder.build_resnet_dense_34(self._input_shape, self._num_outputs)
    #     # Call the Sequential model on each input tensors with shared params.
    #     encoder_1st = ResnetBuilder.build_resnet_dense_34(self._input_shape, self._num_outputs)
    #     encoder_2nd = ResnetBuilder.build_resnet_dense_34(self._input_shape, self._num_outputs)
    #
    #     # Layer to merge two encoded inputs with the l1 distance between them.
    #     distance_layer = keras.layers.Lambda(self.dist_func)
    #
    #     # Call this layer on list of two input tensors.
    #     distance = distance_layer([encoder_1st, encoder_2nd])
    #
    #     # Model prediction: if image pairs are of same letter.
    #     output_layer = keras.layers.Dense(2, activation='sigmoid')
    #     # output_layer = keras.layers.Dense(1, activation='sigmoid')
    #     outputs = output_layer(distance)
    #
    #     # Return a keras Model architecture.
    #     return keras.Model(inputs=[pair_1st, pair_2nd], outputs=outputs)

    def predict(self, input1, input2, **kwargs):

        # Extract keyword arguments.
        kwargs.setdefault('verbose', self._verbose)
        kwargs.setdefault('steps', 2)

        Y_pred = self._model.predict([next(input1),next(input2)], **kwargs)

        Y_pred = np.round(Y_pred, 2)
        y_pred = np.argmax(Y_pred, axis=1)
        print(len(y_pred))
        print('result label desc')
        print('prdict samples')

        # for i in range(0, len(firstClasses)):
        #     Y_TRUE = [int(firstClasses[i] == secondClasses[i])]
        # # Y_TRUE = [[int(firstClasses[i] == secondClasses[i])] for i in range(0, len(firstClasses)]
        # print(pd.DataFrame(data={'first_filenames': firstFileNms, 'second_filenames': secondFileNms, 'Y_TRUE': firstClasses, 'Y_PRED': y_pred,
        #                          'Y_PRED_SCORE': [str(one_pred) for one_pred in Y_pred]}))
        # print('Confusion Matrix')
        # print(confusion_matrix(valid_data.classes, y_pred))
        # for idx, cutOffScore in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        #     print('Classification Report (left cut off score: ' + str(1 - cutOffScore) + ')')
        #     print(classification_report(valid_data.classes, [int(one_pred[1] > cutOffScore) for one_pred in Y_pred],
        #                                 target_names=valid_data.class_indices))
        return Y_pred