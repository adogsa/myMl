# imported from: https://github.com/raghakot/keras-resnet

from __future__ import division

from core import BaseNetwork
from network.lib import ResnetBuilder


class Resnet(BaseNetwork):

    def __init__(self, **kwargs):
        super(Resnet, self).__init__(**kwargs)

    def build(self, **kwargs):

        # Return a keras Model architecture.
        return ResnetBuilder.build_resnet_34(self._input_shape, self._num_outputs)