"""
   @description
     Definition of Loss Function

   @author
     jwjung01@igsinc.co.kr
"""

# Built-in for Abstract Base Classes.

# Third-party libraries.
import keras
import tensorflow as tf

# Classes in this file.
__all__ = ['Loss']


class Loss(object):
    """Implementation of popular loss function for One-shot learning tasks."""

    @staticmethod
    def triplet_loss(y_true, y_pred, alpha=0.2):
        """Triplet Loss function to compare pairs of

        Args:
            y_pred (tf.Tensor): Encoding of anchor & positive example.
            y_true (tf.Tensor): Encoding of anchor & negative example.
            alpha (float, optional): Defaults to 0.2. Margin added to f(A, P).

        Returns:
            tf.Tensor: Triplet loss.
        """

        # Triplet loss for a single image.
        loss = tf.maximum(y_true - y_pred + alpha, 0)

        # Sum over all images.
        return tf.reduce_sum(loss, axis=1, name="Triplet_Loss")

    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        """Binary crossentropy between an output tensor and a target tensor.

        Args:
            y_true: A tensor with the same shape as `output`.
            y_pred: A tensor.

        Returns:
            tf.tensor: Binary crossentropy loss.
        """

        # Binary crossentropy loss function.
        return keras.losses.binary_crossentropy(y_true, y_pred)

    @staticmethod
    def contrastive_loss(y_true, y_pred, alpha=0.2):
        """Contrastive loss function.

        Binary cross entropy between the predictions and targets.
        There is also a L2 weight decay term in the loss to encourage
        the network to learn smaller/less noisy weights and possibly
        improve generalization:

        L(x1, x2, t) = t⋅log(p(x1 ∘ x2)) + (1−t)⋅log(1 − p(x1 ∘ x2)) + λ⋅||w||2

        Args:
            y_pred (any): Predicted distance between two inputs.
            y_true (any): Ground truth or target, t (where, t = [1 or 0]).

            alpha (float, optional): Defaults to 0.2. Slight margin
                added to prediction to avoid 0-learning.

        Returns:
            tf.Tensor: Constrictive loss function.
        """

        loss = y_true * tf.log(y_true) + (1 - y_pred) * tf.log(1 - y_pred) + alpha

        return tf.reduce_mean(loss, name="contrastive_loss")