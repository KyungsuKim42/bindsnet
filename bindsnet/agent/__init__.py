import torch

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
            time: int = 1, dt: float = 1.0, epsilon: float = None,
            epsilon_decay: float=None, **kwargs):
        """
        Initializes the agent.

        :param network: Arbitrary network object.
        :param output_name: Name of the output layer of the network.
        :param encoding: Encoding function.
        :param num_action: Number of action.
        :param action_function: Action function.
        :param time: Amount of time that network should be simulated.
        :param dt: Time length of the single simulation timestep.
        :param epsilon: Initial epsilon value for epsison-greedy method.
        :param epsilon_decay: epsilon is decayed by epsilon*=epsilon_decay in
            every single agent step.

        Keyword Arguments
        :param max_prob: maximum value of firing probability for bernoulli
            encoding.
        """
        self.network = network
        self.output_name = output_name
        self.num_action = num_action
        self.encoding = encoding
        self.action_function = action_function
        self.time = time
        self.dt = dt
        self.timestep = int(time/dt)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.max_prob = kwargs.get('max_prob',2.0)

        self.encoded = {key: torch.Tensor() for key, val in
                        self.network.layers.items() if type(val) == Input}


    def step(self, obs, reward):

        for inpt in self.encoded:
            self.encoded[inpt] = self.encoding(
                    self.obs, time=self.time, dt=self.dt,max_prob=self.max_prob)
        # Do we need clamp or clamp_v?
        self.network.run(inpts=self.encoded, time=self.time, reward=reward)
        output = self.network.layers[output_name]
        self.action = self.action_function(self.num_action, output=self.output)

        return action

    def set_monitor(self, plot_length, plot_iterval):
        # Adding monitors to network.
        for l in self.network.layers:
            self.network.add_monitor(Monitor(self.network.layers[l], 's',
                    int(plot_length * plot_interval * \
                    self.timestep)), name=f'{l}_spikes')

            if 'v' in self.network.layers[l].__dict__:
                self.network.add_monitor(Monitor(self.network.layers[l],
                    'v', int(plot_length * plot_interval * \
                    self.timestep)), name=f'{l}_voltages')
