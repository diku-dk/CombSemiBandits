"""
Provides implemntation of a stochastic and an adversarial combinatorial multi armed bandit environment.
"""
import random
import numpy as np


class Environment:
    """
    Wrapper for environments with Bernoulli arms.
    """
    def __init__(self, gap, dim, m_size, n_steps):
        self.gap = gap
        self.dim = dim
        self.m_size = m_size
        self.n_steps = n_steps
        self.baseline = None
        self.mean_losses = None

    def reset(self):
        raise NotImplementedError

    def play(self, action, time_step):
        del time_step # unused in i.i.d.
        feedback = [-1.0 if random.random() > self.mean_losses[i] else 1.0 for i in action]
        regret = (self.mean_losses[action] - 0.5).sum() - self.baseline
        return feedback, regret


class Stochastic(Environment):
    """
    Bandit environment that sets mean rewards around 0.5 and picks losses from Bernoulli distributions.
    The gap vector at initialization determines the mean rewards.
    """

    def __init__(self, action_set, gap, dim, m_size, n_steps):
        super().__init__(gap, dim, m_size, n_steps)
        if action_set == "full":
            assert abs(gap) <= 1
            self.mean_losses = np.array([0.5 * (1.0 + gap) if i < dim / 2 else 0.5 * (1.0 - gap) for i in range(dim)])
            best_action = [i for i in range(dim) if self.mean_losses[i] < 0.5]
        elif action_set == "m-set":
            self.mean_losses = np.array([0.5 * (1.0 - gap) if i < m_size else 0.5 * (1.0 + gap) for i in range(dim)])
            best_action = [i for i in range(dim) if self.mean_losses[i] < 0.5]
        else:
            raise Exception("Invalid action set %s for stochastic environment, abort." % action_set)
        self.baseline = (self.mean_losses[best_action] - 0.5).sum()

    def reset(self) -> object:
        pass


class Adversarial(Environment):
    """
    Bandit environment that picks losses from Bernoulli distributions.
    The gap vector at initialization determines the mean rewards.
    The mean of the optimal arm iterates between being close to 1 and close to 0 to create a challenging environment
    for stochastic bandit algorithms.
    """

    def __init__(self, action_set, gap, dim, m_size, n_steps):
        super().__init__(gap, dim, m_size, n_steps)
        self.action_set = action_set
        if action_set == "full":
            assert abs(gap) <= 1
            # use only positive losses for adversarial case
            self.mean_losses = np.array([0.5] * self.dim)
            self.best_action = []
            self.baseline = 0
        elif action_set == "m-set":
            self.mean_losses = np.array([0.5 * (1.0 - gap) if i < m_size else 0.5 * (1.0 + gap) for i in range(dim)])
            best_action = [i for i in range(dim) if self.mean_losses[i] < 0.5]
            self.baseline = (self.mean_losses[best_action] - 0.5).sum()
        else:
            raise Exception("Invalid action set %s for stochastic environment, abort." % action_set)

    def play(self, action, time_step):
        if self.action_set == "full":
            # let the mean returns go to the extreme point over time, tune parameter
            gap = 1.0 / (np.power(self.n_steps, (0.5 - 0.5 * np.power(time_step / self.n_steps, 1.5))))
            mean_losses = self.mean_losses + gap
            feedback = [-1.0 if random.random() > mean_losses[i] else 1.0 for i in action]
            return feedback, len(action) * gap
        if self.action_set == "m-set":
            bias = 0.5 * (1.0 - self.gap)
            # play with the parameters until stochastic algorithms fail.
            if int(np.log(time_step) / np.log(1.1)) % 10 < 5:
                mean_losses = self.mean_losses + bias
            else:
                mean_losses = self.mean_losses - bias
            feedback = [-1.0 if random.random() > mean_losses[i] else 1.0 for i in action]
            regret = (self.mean_losses[action] - 0.5).sum() - self.baseline
            return feedback, regret
        raise Exception("Invalid action set %s for stochastic environment, abort." % self.action_set)

    def reset(self):
        pass
