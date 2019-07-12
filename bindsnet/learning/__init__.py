import torch
import numpy as np

from abc import ABC
from typing import Union, Tuple, Optional, Sequence

from ..network.topology import AbstractConnection, Connection, DelayConnection


class LearningRule(ABC):
    # language=rst
    """
    Abstract base class for learning rules.
    """

    def __init__(self, connection: AbstractConnection) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

    # TODO we don't need this function anymore. should we remove?
    def update(self) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        # Implement weight decay.
        pass

class NoOp(LearningRule):
    # language=rst
    """
    Learning rule with no effect.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object which this learning rule will have no effect on.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        super().update()

    def reset(self) -> None:
        """
        Abstract method for resetting update rule.
        """
        pass


class Eligibility(LearningRule):
    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``eligibility`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDPET`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param float tc_e_trace: Time constant of the eligibility trace.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs)

        if isinstance(connection, (Connection, LocallyConnectedConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

        self.e_trace = torch.zeros(self.source.n, self.target.n)
        # TODO match with other time constants.
        self.trace_tc = self.connection.trace_tc

    def _connection_update(self, **kwargs) -> None:
        # TODO Check rigorously if this implementation is proper.
        # language=rst
        super().update()
        # Since network is simulated in sequential manner, we should use
        # source.s instead of source.pre_s again.
        source_s = self.source.s.view(-1)
        target_s = self.target.s.view(-1)

        # Update e_trace.
        self.e_trace -= self.trace_tc * self.e_trace
        self.e_trace += source_s.float().view(-1,1)

        # Reset accounted g_trace.
        self.e_trace[:,target_s] = 0

    def reset(self):
        self.e_trace = torch.zeros(self.source.n, self.target.n)

class SuperSpike(LearningRule):
    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``SuperSpike`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``SuperSpike`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.

        Keyword arguments:

        :param float tc_e_trace: Time constant of the eligibility trace.
        """
        super().__init__(connection=connection)

        if isinstance(connection, Connection):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

        self.s_trace = torch.zeros(self.source.n, self.target.n)
        self.e_trace = torch.zeros(self.source.n, self.target.n)
        # TODO match with other time constants.
        self.trace_tc = self.connection.trace_tc

    def _connection_update(self, **kwargs) -> None:
        # TODO Check rigorously if this implementation is proper.
        # language=rst
        super().update()
        # Since network is simulated in sequential manner, we should use
        # source.s instead of source.pre_s again.
        source_s = self.source.s.view(-1)
        target_s = self.target.s.view(-1)

        # Update s_trace.
        self.s_trace -= self.trace_tc * self.s_trace
        self.s_trace += source_s.float().view(-1,1)
        #self.s_trace[:,target_s] = 0

        self.e_trace -= self.trace_tc * self.e_trace

    def reset(self):
        self.s_trace = torch.zeros(self.source.n, self.target.n)
        self.e_trace = torch.zeros(self.source.n, self.target.n)

class CurrentSuperSpike(LearningRule):
    def __init__(self, connection, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``CurrentSuperSpike`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``SuperSpike`` learning rule will modify.
        Keyword Arguments
        :param float current_tc: First decaying constant for SRM kernel.
        :param float voltage_tc: Second decaying constant for SRM kernel.

        """
        super().__init__(connection=connection)

        if isinstance(connection, (Connection, DelayConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )


        # Decaying cosntants of double-exponenetial kernel.
        self.current_tc = kwargs.get('current_tc', None)
        self.voltage_tc = kwargs.get('voltage_tc', None)
        self.resistance = kwargs.get('resistance', None)

        # Temporal trace for SRM kernel
        self.temp_trace = torch.zeros(self.source.n, self.target.n)
        # Main trace.
        self.e_trace = torch.zeros(self.source.n, self.target.n)

    def surrogate_derivative(self, v):
        der = 1 / (1 + abs(1-v))**2
        return der

    def _connection_update(self, **kwargs) -> None:
        # TODO Check rigorously if this implementation is proper.
        # language=rst
        super().update()
        source_s = self.source.s.view(-1)
        target_s = self.target.s.view(-1)

        # Update temp_trace.
        self.temp_trace -= self.current_tc * self.temp_trace
        if isinstance(self.connection, Connection):
            self.temp_trace += torch.ger(self.source.y,
                                         self.surrogate_derivative(self.target.v))
        else:
            # Reset synaptic trace.
            self.connection.y[source_s,:] = 0
            self.temp_trace += (self.connection.y
                                * self.surrogate_derivative(self.target.v))
        # Update eligibility trace.
        self.e_trace -= self.voltage_tc * self.e_trace
        self.e_trace += self.temp_trace * self.resistance


    def reset(self):
        self.temp_trace = torch.zeros(self.source.n, self.target.n)
        self.e_trace = torch.zeros(self.source.n, self.target.n)
