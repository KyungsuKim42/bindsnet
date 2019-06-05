import torch
import numpy as np
import torch.nn.functional as F

from typing import Union, Tuple, Optional, Sequence
from abc import ABC, abstractmethod
from torch.nn.modules.utils import _pair

from .nodes import Nodes

class AbstractConnection(ABC):
    # language=rst
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(self, source: Nodes, target: Nodes,
                 nu: Optional[Union[float, Sequence[float]]] = None,
                 **kwargs) -> None:
        # language=rst
        """
        Constructor for abstract base class for connection objects.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        Keyword arguments:
        :param function update_rule: Modifies connection parameters according to
            some rule.
        :param float w_std: The variance of the initialized weight distribution.
        :param float w_mean: The mean of the initialized weight distribution.
        """
        self.w = None
        self.source = source
        self.target = target
        self.nu = nu

        assert isinstance(source, Nodes), 'Source is not a Nodes object'
        assert isinstance(target, Nodes), 'Target is not a Nodes object'

        from ..learning import NoOp

        self.update_rule = kwargs.get('update_rule', NoOp)
        self.w_mean = kwargs.get('w_mena', None)
        self.w_std = kwargs.get('w_std', None)

        if self.update_rule is None:
            self.update_rule = NoOp

        # Do we need nu?
        self.update_rule = self.update_rule(
            connection=self, nu=self.nu, **kwargs)

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        # language=rst
        """
        Compute pre-activations of downstream neurons given spikes of upstream
            neurons.
        :param s: Incoming spikes.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        learning = kwargs.get('learning', True)

        if learning:
            self.update_rule.update(**kwargs)

    @abstractmethod
    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of incoming connection weights equal to ``self.norm``.
        """
        pass

    @abstractmethod
    def reset_(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        pass


class Connection(AbstractConnection):
    # language=rst
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(self, source: Nodes, target: Nodes,
                 nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        Keyword arguments:
        :param function update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float w_std: The variance of the initialized weight distribution.
        :param float w_mean: The mean of the initialized weight distribution.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.w = kwargs.get('w', None)
        if self.w is None:
            assert self.w_mean is not None, 'Must specify the mean of weights'
            assert self.w_std is not None, 'Must specify the variance of weights'
            self.w = self.w_mean + torch.randn(source.n, target.n)*self.w_std

        self.b = kwargs.get('b', torch.zeros(target.n))


    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """
        # Compute multiplication of spike activations by connection weights and add bias.
        post = s.float().view(-1) @ self.w + self.b
        return post.view(*self.target.shape)

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def reset_(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_()
        if hasattr(self.update_rule, 'reset'):
            self.update_rule.reset()
