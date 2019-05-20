"""Base class for building models for One-shot learning.

   @description
     For training, validating/evaluating & predictions with SiameseNetwork.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: base.py
     Package: omniglot
     Created on 2nd August, 2018 @ 02:23 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""
import multiprocessing

# Built-in for Abstract Base Classes.
from abc import ABCMeta, abstractmethod

# Third-party libraries.
import keras
import tensorflow as tf
import numpy as np
from core.loss import Loss
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.callbacks import EarlyStopping
from matplotlib import pyplot
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# Classes in this file.
__all__ = ['BaseNetwork']


class BaseNetwork(object):

    # Abstract Base Class.
    __metaclass__ = ABCMeta
    # Loss functions for one-shot tasks.
    losses = Loss

    def __init__(self, visual_plot: bool=True, **kwargs):
        self._num_outputs = kwargs.get('num_outputs', 2)
        self._verbose = kwargs.get('verbose', 1)
        self._input_shape = kwargs.get('input_shape', (105, 105, 1))
        self._save_weights_only = kwargs.get('save_weights_only', False)
        self._model_dir = kwargs.get('model_dir', 'saved/models/')
        self.loss = kwargs.get('loss', BaseNetwork.losses.binary_crossentropy)
        self.lr = kwargs.get('lr', 1e-3)
        self.optimizer = kwargs.get('optimizer', keras.optimizers.Adam(lr=self.lr))
        # Path to save model's weights.
        self._save_path = 'weights.h5' if self._save_weights_only else 'network.h5'
        self._save_path = kwargs.get('save_path', self._save_path)
        self._save_path = self._model_dir + self._save_path
        self._metrics = kwargs.get('metrics', ['accuracy'])
        self._model = self.build(**kwargs)
        self.history = None;
        if visual_plot:
            SVG(self.plot_model())
            self._model.summary()
        # TODO: Get layerwise learning rates and momentum annealing scheme
        # as described in the paper.
        self._model.compile(loss=self.loss, optimizer=self.optimizer,
                            metrics=self._metrics)

    def __repr__(self):
        return ('BaseNetwork(input_shape={0}, loss={1}, optimizer={2}), metrics={3}'.format(self._input_shape, self.loss, self.optimizer, self._metrics));

        # return (f'BaseNetwork(input_shape={self._input_shape}, loss={self.loss},'
        #         f' optimizer={self.optimizer}), metrics={self.metrics}');

    def __str__(self):
        return self.__repr__()

    def __call__(self, inputs, **kwargs):
        """Calls the model on new inputs.

        See `BaseNetwork.call`.
        """
        return self.predict(inputs, **kwargs)

    @abstractmethod
    def build(self, **kwargs):
        raise NotImplementedError('Sub-class must override `build` method.')

    def train(self, train_data, valid_data,
              batch_size: int=64, steps_per_epoch: int=128, resume_training: bool=True, **kwargs):
        # Set default keyword arguments.
        kwargs.setdefault('callbacks', [EarlyStopping(monitor='acc', patience=20)])  # 조기종료 콜백함수 정의
        kwargs.setdefault('epochs', 1)
        kwargs.setdefault('steps_per_epoch', steps_per_epoch)
        kwargs.setdefault('verbose', self._verbose)
        kwargs.setdefault('use_multiprocessing', False)
        # kwargs.setdefault('use_multiprocessing', True)
        kwargs.setdefault('workers', multiprocessing.cpu_count())
        # kwargs.setdefault('pickle_safe', True)
        # kwargs.setdefault('shuffle', False)

        # Resume training.
        if resume_training and tf.gfile.Exists(self._save_path):
            self.load_model(self._save_path)

        try:
            # Fit the network.
            if valid_data is None:
                # without validation set.
                self.history = self._model.fit_generator(train_data, **kwargs)
            else:
                # with validation set.
                self.history = self._model.fit_generator(train_data, validation_data=valid_data,
                                          validation_steps=batch_size, **kwargs)
        except KeyboardInterrupt:
            # When training is unexpectedly stopped!
            print('Training interrupted by user!')

        # Save learned weights after completed training or KeyboardInterrupt.
        self.save_model()

    def plot_model(self, **kwargs):
        # Set default keyword arguments.
        kwargs.setdefault('rankdir', 'TB')
        kwargs.setdefault('show_shapes', True)
        kwargs.setdefault('show_layer_names', True)

        # Convert Keras model to plot.
        dot = model_to_dot(self._model, **kwargs)

        # Create plot as an SVG byte string to be read by IPython SVG function.
        return dot.create(prog='dot', format='svg')

    def show_graph(self, title='model accuracy', ylabel='accuracy', history_target=['acc', 'val_acc']):
        # summarize history for accuracy
        for one_target in history_target:
            pyplot.plot(self.history.history[one_target])
        pyplot.title(title)
        pyplot.ylabel(ylabel)
        pyplot.xlabel('epoch')
        pyplot.legend(['training', 'validation'], loc='lower right')
        pyplot.show()

    def save_model(self, weights_only=False):
        """Save model's parameters or weights only to an h5 file.

        Args:
            weights_only (bool, optional): Defaults to False. If set to true,
                only model's weights will be saved.
        """

        # Pretty prints.
        # self._log(f'\n{"-" * 65}\nSaving model...')

        if weights_only:
            # Save model weights.
            self._model.save_weights(filepath=self._save_path, save_format='h5')
        else:
            # Save entire model.
            # self._model.save(filepath=self._save_path)
            keras.models.save_model(model=self._model, filepath=self._save_path)

        # Pretty prints.
        # self._log(f'Saved model weights to "{self._save_path}"!\n{"-" * 65}\n')

    @staticmethod
    def load_model(save_path):
        """Load a saved model.

        Raises:
            FileExistsError: Model already saved to `Network.save_path`.

        Returns:
            keras.Model: Saved model.
        """

        model = None;
        if tf.gfile.Exists(save_path):
            # self._log(f'Loading model from {self._save_path}')
            model = keras.models.load_model(save_path)
        else:
            raise FileNotFoundError('{0} was not found.'.format(save_path))

        return model

    def predict(self, valid_data, **kwargs):
        return self.predict(self._model, valid_data, kwargs)

    @staticmethod
    def predict(model, valid_data, **kwargs):
        kwargs.setdefault('verbose', 1)
        Y_pred = model.predict_generator(valid_data, **kwargs)
        Y_pred = np.round(Y_pred, 2)
        y_pred = np.argmax(Y_pred, axis=1)
        print('result label desc')
        print(valid_data.class_indices)
        from keras.preprocessing.image import DirectoryIterator
        if type(valid_data) is DirectoryIterator:
            print('prdict samples')
            print(pd.DataFrame(data={'filenames': valid_data.filenames, 'Y_TRUE': valid_data.classes, 'Y_PRED': y_pred, 'Y_PRED_SCORE': [str(one_pred) for one_pred in Y_pred]}))
        else:
            print('Y_TRUE')
            print(valid_data.classes)
            print('Y_PRED')
            print(y_pred)
        print('Confusion Matrix')
        print(confusion_matrix(valid_data.classes, y_pred))
        for idx, cutOffScore in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
            print('Classification Report (left cut off score: ' + str(1 - cutOffScore) + ')')
            print(classification_report(valid_data.classes, [int(one_pred[1] > cutOffScore) for one_pred in Y_pred], target_names=valid_data.class_indices))
        return Y_pred

    def evaluation(self, valid_data, **kwargs):
        return self.evaluation(self._model, valid_data, kwargs)

    @staticmethod
    def evaluation(model, valid_data, **kwargs):
        kwargs.setdefault('verbose', 1)
        scores = model.evaluate_generator(valid_data, **kwargs)
        for idx, one_score in enumerate(scores):
            print("%s: %.2f%%" % (model.metrics_names[idx], one_score * 100))

    @staticmethod
    def dist_func(x):
        """Difference function. Compute difference between 2 images.

        Args:
            x (tf.Tensor): Signifying two inputs.

        Returns:
            tf.Tensor: Absolute squared difference between two inputs.
        """

        return abs(x[0] - x[1])

    @property
    def model(self):
        """Network's background model.

        Returns:
            keras.Model: Underlaying model used by Network.
        """

        return self._model

    @property
    def save_path(self):
        """Path to saved model/model's weight.

        Returns:
            str: Path to an h5 file.
        """

        return self._save_path

    @property
    def model_dir(self):
        """Directory for saving and loading model.

        Returns:
            str: Path to an h5 file.
        """

        return self._model_dir