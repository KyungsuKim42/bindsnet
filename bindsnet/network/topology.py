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
        self.w_mean = kwargs.get('w_mean', None)
        self.w_std = kwargs.get('w_std', None)
        self.trace_tc = kwargs.get('trace_tc', None)

        self.update_rule = self.update_rule(
            connection=self, **kwargs)

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
                 **kwargs) -> None:
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
        super().__init__(source, target, nu, **kwargs)

        self.w = kwargs.get('w', None)
        if self.w is None:
            assert self.w_mean is not None, 'Must specify the mean of weights'
            assert self.w_std is not None, 'Must specify the variance of weights'
            min = self.w_mean - 0.5*self.w_std
            range = self.w_std
            self.w = torch.rand(source.n, target.n) * range + min
            # self.w = self.w_mean + torch.rand(source.n, target.n)*self.w_std

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


class DelayConnection(AbstractConnection):
    # language=rst
    """
    Fully connected layer with axonal delay for each synapses.
    """

    def __init__(self, source: Nodes, target: Nodes,
                 nu: Optional[Union[float, Sequence[float]]] = None,
                 **kwargs) -> None:
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
        :param int max_delay: The maximum value of synaptic delay.
        :param float current_tc : Decaying constant of current, for synaptic trace.
        :param float voltage_tc : Decaying constant of voltage, for synaptic trace.
        :param float resistance : Constant for current neuron model. For synaptic trace.
        """
        super().__init__(source, target, nu, **kwargs)

        self.w = kwargs.get('w', None)
        self.current_tc = kwargs.get('current_tc', None)
        self.voltage_tc = kwargs.get('voltage_tc', None)
        self.resistance = kwargs.get('resistance', None)
        self.max_delay = kwargs.get('max_delay', None)

        if self.w is None:
            assert self.w_mean is not None, 'Must specify the mean of weights'
            assert self.w_std is not None, 'Must specify the variance of weights'
            min = self.w_mean - 0.5*self.w_std
            range = self.w_std
            # should it be normal distribution or ok with current setting??
            self.w = torch.rand(source.n, target.n) * range + min
            # self.w = self.w_mean + torch.rand(source.n, target.n)*self.w_std

        self.delay = torch.randint(low=1, high=self.max_delay+1,
                                   size=[source.n, target.n])
        self.spike_queue = torch.zeros(source.n,target.n,self.max_delay).byte()
        self.arrived_spike = torch.zeros(source.n,target.n).byte()

        # For learning, we need to keep the synaptic trace.
        self.x = torch.zeros(self.w.shape)
        self.y = torch.zeros(self.w.shape)


    def compute(self, s, t):
        """
        Compute pre-activations given spikes using connection weights.
        :param s: Incoming spikes.
        :param t: Timestep of computing time. Required because of axonal delay.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """
        # Assign new spike input to the spike_queue.
        temp_t = torch.tensor(t)
        temp_t_ = torch.tensor(t + self.max_delay - 1)
        write_index = torch.fmod(temp_t_, self.delay)
        write_index = torch.unsqueeze(write_index, 2)
        read_index = torch.fmod(temp_t, self.delay)
        read_index = torch.unsqueeze(read_index, 2)
        # Write current timestep's spike into the queue
        broadcasted_spike = s.unsqueeze(dim=1).repeat(1,self.target.n).unsqueeze(dim=2)
        self.spike_queue.scatter_(2,write_index,broadcasted_spike)
        # Read arrived spike for propagation.
        self.arrived_spike = torch.gather(self.spike_queue, 2, read_index)
        self.arrived_spike = self.arrived_spike.squeeze(dim=2)

        # Update synaptic trace
        self.x -= self.current_tc * self.x
        self.x += self.arrived_spike.float()

        self.y -= self.voltage_tc * self.y
        self.y += self.resistance * self.x

        post = torch.sum(self.arrived_spike.float() * self.w, dim=0)
        return post

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

        self.spike_queue = torch.zeros(self.spike_queue.shape).byte()
        self.arrived_spike = torch.zeros(self.arrived_spike.shape).byte()
        self.x = torch.zeros(self.x.shape)
        self.y = torch.zeros(self.y.shape)

        if hasattr(self.update_rule, 'reset'):
            self.update_rule.reset()
