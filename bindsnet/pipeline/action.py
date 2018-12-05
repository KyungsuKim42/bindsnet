import torch
import numpy as np

from . import Pipeline


def select_multinomial(pipeline: Pipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action probabilistically based on spiking activity from a network layer.

    :param pipeline: Pipeline with environment that has an integer action space.
    :return: Action sampled from multinomial over activity of similarly-sized output layer.

    Keyword arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs['output']
    except KeyError:
        raise KeyError('select_multinomial() requires an "output" layer argument.')

    output = pipeline.network.layers[output]
    action_space = pipeline.env.action_space

    assert output.n % action_space.n == 0, 'Output layer size not equal to size of action space.'

    pop_size = int(output.n / action_space.n)
    spikes = output.s
    _sum = spikes.sum().float()

    # Choose action based on population's spiking.
    if _sum == 0:
        action = np.random.choice(pipeline.env.action_space.n)
    else:
        pop_spikes = torch.Tensor([spikes[(i * pop_size):(i * pop_size) + pop_size].sum() for i in range(action_space.n)]) #FIXED : range(output.n) is replaced to range(action_space.n)
        action = torch.multinomial((pop_spikes.float() / _sum).view(-1), 1)[0].numpy() # For some environments in Gym, tensor cannot be action. Maybe all other non-atari environments?

    return action


def select_softmax(output, num_action) -> int:
    # language=rst
    """
    Selects an action using softmax function based on spiking from a network layer.

    :param output: Output spike_train
    :param num_action: Number of actions
    :return: Action sampled from softmax over activity of similarly-sized output layer.

    """
    try:
        output = kwargs['output']
    except KeyError:
        raise KeyError('select_softmax() requires an "output" layer argument.')

    assert output.n == num_action, \
        'Output layer size not equal to size of action space.'

    assert hasattr(pipeline, 'spike_record'), 'Pipeline has not attribute named: spike_record.'

    # Sum of previous iterations' spikes (Not yet implemented)
    spikes = torch.sum(pipeline.spike_record[output], dim=1)
    _sum = torch.sum(torch.exp(spikes.float()))

    if _sum == 0:
        action = np.random.choice(pipeline.env.action_space.n)
    else:
        action = torch.multinomial((torch.exp(spikes.float()) / _sum).view(-1), 1)[0]

    return action

def select_max(output: torch.Tensor, num_action: int) -> int:
    """
    Selects an action that has the most number of spikes.
    If the number of the spikes are same, pick randomly.
    """
    num_spike = torch.sum(output, dim=1)
    assert num_spike.shape[0] % num_action == 0, \
        'Output layer size not equal to size of action space.'
    pop_size = int(num_spike.shape[0] / num_action)
    spike_sum = torch.Tensor([num_spike[(i * pop_size):(i * pop_size) + pop_size
                          ].sum() for i in range(num_action)])

    max_indices = (spike_sum==torch.max(spike_sum)).nonzero()
    # If two or more actions have same score, pick randomly.
    if len(max_indices) > 1:
        action = int(max_indices[np.random.randint(len(max_indices))])
    else:
        action = int(max_indices[0])

    return action

def select_random(pipeline: Pipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action randomly from the action space.

    :param pipeline: Pipeline with environment that has an integer action space.
    :return: Action randomly sampled over size of pipeline's action space.
    """
    # Choose action randomly from the action space.
    return np.random.choice(pipeline.env.action_space.n)

def num_spike(pipeline: Pipeline, **kwargs) -> torch.Tensor:
    """
    Returns accumulated spike count. It assumes that action is a vector rather than scalar.

    :param pipeline Pipeline with environment that has an integer action space.
    :return: accumulated spike count for each action element.

    Keyword Arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs['output']
    except KeyError:
        raise KeyError('num_spike() requires an "output" layer argument.')

    assert hasattr(pipeline, 'spike_record'), 'Pipeline has not attribute named: spike_record.'
    spike = pipeline.spike_record[output][:,-pipeline.timestep:]
    num_spike = torch.sum(spike, dim=1)

    # n_action is temporal variable that only exists in MNSITEnvironment. This code sould be generalized.
    n_action = pipeline.env.n_action
    assert num_spike.shape[0] % n_action == 0, 'Output layer size not equal to size of action space.'

    pop_size = int(num_spike.shape[0] / n_action)
    action = torch.Tensor([num_spike[(i * pop_size):(i * pop_size) + pop_size].sum() for i in range(n_action)])

    return action
