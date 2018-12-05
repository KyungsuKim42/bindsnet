import torch
import numpy as np

from ..network import Network
from ..network.nodes import Input
from ..network.monitors import Monitor

from typing import Callable

class Agent:
    """
    Agent contains network, encoding, action_function. It is implemented
    in order to make this project more object-oriented. It also helps
    synchronizing variables between network/encoding/action_function.
    Think of the agent as a blackbox which takes reward and observation and
    returns the action.
    """
    def __init__(
            self, network: Network = None, output_name: str= None, encoding:Callable = None,
            num_action: int = None, action_function:Callable = None,
            time: int = 1, dt: float = 1.0, is_ac:bool = False,
            critic_name:str = None, **kwargs):
        """
        Initializes the agent.

        :param network: Arbitrary network object.
        :param output_name: Name of the output layer of the network.
        :param encoding: Encoding function.
        :param num_action: Number of action.
        :param action_function: Action function.
        :param time: Amount of time that network should be simulated.
        :param dt: Time length of the single simulation timestep.

        Keyword Arguments
        :param max_prob: maximum value of firing probability for bernoulli
            encoding.
        :param epsilon: Initial epsilon value for epsison-greedy method.
        :param epsilon_decay: epsilon is decayed by epsilon*=epsilon_decay in
            every single agent step.
        :param is_ac: Boolean variable which indicates this agent is actor
            critic agent, i.e, has critic neurons or not.
        :param critic_name: Name of the critic layer.
        :param critic_coeff: Coefficient of critic layer.
        :param critic_bias: Bias of critic layer.
        """

        self.network = network
        self.output_name = output_name
        self.num_action = num_action
        self.encoding = encoding
        self.action_function = action_function
        self.time = time
        self.dt = dt
        self.timestep = int(time/dt)
        assert network.layers[output_name].n % num_action == 0, \
            "Number of "

        self.max_prob = kwargs.get('max_prob', 1.0)
        self.epsilon = kwargs.get('epsilon',0.0)
        self.epsilon_decay = kwargs.get('epsilon_decay',0.0)
        self.is_ac = kwargs.get('is_ac', False)
        self.critic_name = kwargs.get('critic_name', None)
        self.critic_coeff = kwargs.get('ciritic_coeff', 0.0)
        self.critic_bias = kwargs.get('critic_bias', 0.0)

        self.encoded = {key: torch.Tensor() for key, val in
                        self.network.layers.items() if type(val) == Input}

        self.spike_record = {}
        self.voltage_record = {}
        self.threshold_value = {}

    def reward_modulated_update(self,reward, action):
        """
        Update weights based on reward/RPE value.
        Currently, only layers that has MSTDPET update rule is updated in this
        function.
        """
        self.network.reward_modulated_update(reward, action, self.num_action)

    def step(self, obs, reward):

        for inpt in self.encoded:
            self.encoded[inpt] = self.encoding(
                    obs, time=self.time, dt=self.dt,max_prob=self.max_prob)
        # Do we need clamp or clamp_v?
        self.network.run(inpts=self.encoded, time=self.time, reward=reward)
        self.set_spike_data()
        self.set_voltage_data()
        self.output = self.spike_record[self.output_name][:,-self.timestep:]
        self.action = self.action_function(self.output, self.num_action)

        # Epsilon Greedy
        if self.epsilon is not None:
            print(f'Epsilon: {self.epsilon}')
            if np.random.rand() < self.epsilon:
                self.action = np.random.randint(self.num_action)
            self.epsilon *= self.epsilon_decay

        return self.action


    def update(self, reward):
        # If the agent has critic layer, calculate reward prediction error.
        if self.is_ac:
            num_spikes = torch.sum(
                self.spike_record[self.critic_name][:,-self.timestep:])
            reward_prediction = self.critic_coeff * num_spikes + \
                self.critic_bias
            reward -= reward_prediction
        self.network.update(reward)


    def set_monitor(self, plot_length, plot_interval):
        # Adding monitors to network.
        for l in self.network.layers:
            self.network.add_monitor(Monitor(self.network.layers[l], 's',
                    int(plot_length * plot_interval * \
                    self.timestep)), name=f'{l}_spikes')

            if 'v' in self.network.layers[l].__dict__:
                self.network.add_monitor(Monitor(self.network.layers[l],
                    'v', int(plot_length * plot_interval * \
                    self.timestep)), name=f'{l}_voltages')

    def reward_prediction(self) -> float:
        """
        Returns reward prediction based on critic layer's activity.

        :return: Reward prediction value. Calculated as "a * num_spikes - b"
        """

        assert hasattr(self, 'spike_record'), 'Pipeline has not attribute named:\
            spike_record.'
        num_spikes = torch.sum(self.spike_record[self.critic][:,-self.timestep:])
        reward_prediction = self.network.critic_coeff * num_spikes + \
            self.network.critic_bias

        return reward_prediction

    def set_spike_data(self) -> None:
        """
        Get the spike data from all layers in the agent's network.
        """
        self.spike_record = {l: self.network.monitors[f'{l}_spikes'].get('s')\
            for l in self.network.layers}

    def set_voltage_data(self) -> None:
        # language=rst
        """
        Get the voltage data and threshold value from all applicable layers in
            the agent's network.
        """
        self.voltage_record = {}
        self.threshold_value = {}
        for l in self.network.layers:
            if 'v' in self.network.layers[l].__dict__:
                self.voltage_record[l] = self.network.monitors[f'{l}_voltages']\
                    .get('v')
            if 'thresh' in self.network.layers[l].__dict__:
                self.threshold_value[l] = self.network.layers[l].thresh

    def reset(self) -> None:
        self.network.reset_()
