"""Omniglot helper package.

   @description
     Network - For training, validating/evaluating & predictions with SiameseNetwork.
     Dataset - For pre-processing and loading the Omniglot dataset.
     Visualize - For visualizing the Omniglot dataset & model.
     utils - For converting values to Tensor & pre-paring dataset for `tf.estimator` API.

   @author
     Victor I. Afolabi
     Artificial Intelligence & Software Engineer.
     Email: javafolabi@gmail.com
     GitHub: https://github.com/victor-iyiola/

   @project
     File: __init__.py
     Package: omniglot
     Created on 18 May, 2018 @ 5:22 PM.

   @license
     MIT License
     Copyright (c) 2018. Victor I. Afolabi. All rights reserved.
"""

__author__ = 'Victor I. Afolabi'

# Supress TensorFlow import warnings.
from core.base import BaseNetwork
from core.loss import Loss

__all__ = [
    # Dataset helpers.
    'BaseNetwork',
    'Loss'
]
