import torch

from typing import Dict

from .nodes import AbstractInput, Nodes
from .topology import AbstractConnection
from .monitors import AbstractMonitor
from ..learning import MSTDP, MSTDPET

__all__ = [
    'load_network', 'Network', 'nodes', 'monitors', 'topology'
]


def load_network(file_name: str) -> 'Network':
    # language=rst
    """
    Loads serialized network object from disk.

    :param file_name: Path to serialized network object on disk.
    """
    try:
        return torch.load(open(file_name, 'rb'))
    except FileNotFoundError:
        print('Network not found on disk.')


class Network:
    # language=rst
    """

    Most important object of the :code:`bindsnet` package. Responsible for the simulation and interaction of nodes and
    connections.

    **Example:**

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt

        from bindsnet         import encoding
        from bindsnet.network import Network, nodes, topology, monitors

        network = Network(dt=1.0)  # Instantiates network.

        X = nodes.Input(100)  # Input layer.
        Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
        C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

        # Spike monitor objects.
        M1 = monitors.Monitor(obj=X, state_vars=['s'])
        M2 = monitors.Monitor(obj=Y, state_vars=['s'])

        # Add everything to the network object.
        network.add_layer(layer=X, name='X')
        network.add_layer(layer=Y, name='Y')
        network.add_connection(connection=C, source='X', target='Y')
        network.add_monitor(monitor=M1, name='X')
        network.add_monitor(monitor=M2, name='Y')

        # Create Poisson-distributed spike train inputs.
        data = 15 * torch.rand(1, 100)  # Generate random Poisson rates for 100 input neurons.
        train = encoding.poisson(datum=data, time=5000)  # Encode input as 5000ms Poisson spike trains.

        # Simulate network on generated spike trains.
        inpts = {'X' : train}  # Create inputs mapping.
        network.run(inpts=inpts, time=5000)  # Run network simulation.

        # Plot spikes of input and output layers.
        spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}

        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for i, layer in enumerate(spikes):
            axes[i].matshow(spikes[layer], cmap='binary')
            axes[i].set_title('%s spikes' % layer)
            axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
            axes[i].set_xticks(()); axes[i].set_yticks(())
            axes[i].set_aspect('auto')

        plt.tight_layout(); plt.show()
    """

    def __init__(self, dt: float = 1.0, is_actor_critic=False, critic_coeff=1.0, critic_bias=0.0) -> None:
        # language=rst
        """
        Initializes network object.
        :param dt: Simulation timestep. All other objects' time constants are relative to this value.
                   This is required because reward-modulated learning rules runs depending on dt.

        Keyword Arguments
        :param is_actor_critic: Boolean variable which indicates this network is actor critic network.
                If this is the case, reward prediction error signal instead of reward itself is used for learning.
        :param critic_coeff: a in "reward_prediction = a*num_spikes_in_critic + b"
        :param critic_bias: b in "reward_prediction = a*num_spikes_in_critic + b"
        """
        self.dt = dt
        self.layers = {}
        self.connections = {}
        self.monitors = {}
        self.learning = True

        self.is_actor_critic = is_actor_critic
        self.critic_coeff = critic_coeff
        self.critic_bias = critic_bias

    def add_layer(self, layer: Nodes, name: str) -> None:
        # language=rst
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer.
        """
        self.layers[name] = layer
        layer.network = self
        layer.dt = self.dt

    def add_connection(self, connection: AbstractConnection, source: str, target: str) -> None:
        # language=rst
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[(source, target)] = connection
        connection.network = self
        connection.dt = self.dt

    def add_monitor(self, monitor: AbstractMonitor, name: str) -> None:
        # language=rst
        """
        Adds a monitor on a network object to the network.

        :param monitor: An instance of class ``Monitor``.
        :param name: Logical name of monitor object.
        """
        self.monitors[name] = monitor
        monitor.network = self
        monitor.dt = self.dt

    def save(self, file_name: str) -> None:
        # language=rst
        """
        Serializes the network object to disk.

        :param file_name: Path to store serialized network object on disk.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from pathlib          import Path
            from bindsnet.network import *
            from bindsnet.network import topology

            # Build simple network.
            network = Network(dt=1.0)

            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')

            # Save the network to disk.
            network.save(str(Path.home()) + '/network.pt')
        """
        torch.save(self, open(file_name, 'wb'))

    def get_inputs(self) -> Dict[str, torch.Tensor]:
        # language=rst
        """
        Fetches outputs from network layers to use as input to downstream layers.

        :return: Inputs to all layers for the current iteration.
        """
        inpts = {}

        # Loop over network connections.
        for c in self.connections:
            # Fetch source and target populations.
            source = self.connections[c].source
            target = self.connections[c].target

            if not c[1] in inpts:
                inpts[c[1]] = torch.zeros(target.shape)

            # Add to input: source's spikes multiplied by connection weights.
            inpts[c[1]] += self.connections[c].compute(source.s)

        return inpts

    def run(self, inpts: Dict[str, torch.Tensor], time: int, **kwargs) -> None:
        # language=rst
        """
        Simulation network for given inputs and time.

        :param inpts: Dictionary of ``Tensor``s of shape ``[time, n_input]``.
        :param time: Simulation time.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if neurons should be clamped to
                                              values specified in the ``clamp_v`` argument. The ``Tensor``s have shape
                                              ``[time, n_input]``.
        :param float reward: Scalar value used in reward-modulated learning.
        :param Dict[str, torch.Tensor] masks: Mapping of connection names to boolean masks determining which weights to
                                              clamp to zero.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from bindsnet.network import Network
            from bindsnet.network.nodes import Input
            from bindsnet.network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inpts={'I' : spikes}, time=500)

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        # Parse keyword arguments.
        clamps = kwargs.get('clamp', {})
        masks = kwargs.get('masks', {})

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Get input to all layers.
        inpts.update(self.get_inputs())

        for c in self.connections:
            if isinstance(self.connections[c].update_rule, (MSTDP, MSTDPET)):
                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            for l in self.layers:
                # Update each layer of nodes.
                if isinstance(self.layers[l], AbstractInput):
                    self.layers[l].forward(x=inpts[l][t])
                else:
                    self.layers[l].forward(x=inpts[l])

                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    self.layers[l].s[clamp] = 1

            # Run synapse updates.
            for c in self.connections:
                if not isinstance(self.connections[c].update_rule, (MSTDP, MSTDPET)):
                    self.connections[c].update(
                        mask=masks.get(c, None), learning=self.learning, **kwargs
                    )

            # Get input to all layers.
            inpts.update(self.get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()

            # Re-normalize connections.
            for c in self.connections:
                self.connections[c].normalize()


    def reset_(self) -> None:
        # language=rst
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_()

        for connection in self.connections:
            self.connections[connection].reset_()

        for monitor in self.monitors:
            self.monitors[monitor].reset_()
