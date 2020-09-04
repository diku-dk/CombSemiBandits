import random
import numpy as np

class Environment:

    def __init__(self, gap, d, m, T):
        self.gap = gap
        self.d = d
        self.m = m
        self.T = T
        self.baseline = None
        self.mean_losses = None

    def reset(self):
        raise NotImplementedError

    def play(self, action, t):
        feedback = [-1.0 if random.random() > self.mean_losses[i] else 1.0 for i in action]
        r = (self.mean_losses[action] - 0.5).sum() - self.baseline
        return feedback, r


class Stochastic(Environment):

    def __init__(self, action_set, gap, d, m, T):
        super().__init__(gap, d, m, T)
        if action_set == "full":
            assert abs(gap) <= 1
            self.mean_losses = np.array([0.5 * (1.0 + gap) if i < d / 2 else 0.5 * (1.0 - gap) for i in range(d)])
            best_action = [i for i in range(d) if self.mean_losses[i] < 0.5]
        elif action_set == "m-set":
            self.mean_losses = np.array([0.5 * (1.0 - gap) if i < m else 0.5 * (1.0 + gap) for i in range(d)])
            best_action = [i for i in range(d) if self.mean_losses[i] < 0.5]
        else:
            raise Exception("Invalid action set %s for stochastic environment, abort." % action_set)
        self.baseline = (self.mean_losses[best_action] - 0.5).sum()

    def reset(self):
        pass


class Adversarial(Environment):

    def __init__(self, action_set, gap, d, m, T):
        super().__init__(gap, d, m, T)
        self.action_set = action_set
        if action_set == "full":
            assert abs(gap) <= 1
            # use only positive losses for adversarial case
            self.mean_losses = np.array([0.5] * self.d)
            self.best_action = []
            self.baseline = 0
        elif action_set == "m-set":
            self.mean_losses = np.array([0.5 * (1.0 - gap) if i < m else 0.5 * (1.0 + gap) for i in range(d)])
            best_action = [i for i in range(d) if self.mean_losses[i] < 0.5]
            self.baseline = (self.mean_losses[best_action] - 0.5).sum()
        else:
            raise Exception("Invalid action set %s for stochastic environment, abort." % action_set)

    def play(self, action, t):
        if self.action_set == "full":

            # let the mean returns go to the extreme point over time, tune parameter
            gap = 1.0 / (np.power(self.T, (0.5 - 0.5 * np.power(t / self.T, 1.5))))
            mean_losses = self.mean_losses + gap
            feedback = [-1.0 if random.random() > mean_losses[i] else 1.0 for i in action]
            return feedback, len(action) * gap
        elif self.action_set == "m-set":
            bias = 0.5 * (1.0 - self.gap)
            # play with the parameters until stochastic algorithms fail.
            if int(np.log(t) / np.log(1.1)) % 10 < 5:
                mean_losses = self.mean_losses + bias
            else:
                mean_losses = self.mean_losses - bias
            feedback = [-1.0 if random.random() > mean_losses[i] else 1.0 for i in action]
            r = (self.mean_losses[action] - 0.5).sum() - self.baseline
            return feedback, r
        else:
            raise Exception("Invalid action set %s for stochastic environment, abort." % self.action_set)

    def reset(self):
        pass
