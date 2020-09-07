"""
Provides implementation of various semi-bandit algorithms
"""
import numpy as np
import random

EPS = 10E-12


class Bandit:
    """
    Interface for semi-bandit algorithms.
    """

    def __init__(self, action_set, dim, m_size):
        self.dim = dim
        self.m_size = m_size
        if action_set == "full":
            self.unconstrained = True
        elif action_set == "m-set":
            self.unconstrained = False
        else:
            raise Exception("Invalid action set %s for OSMD, abort." % action_set)

    def next(self):
        raise NotImplementedError

    def reset(self) -> object:
        raise NotImplementedError

    def update(self, action, feedback):
        raise NotImplementedError

    def sample_action(self, x):
        """

        :param x: List[Float], marginal probabilities
        :return: combinatorial action
        """
        if self.unconstrained:
            return [i for i, val in enumerate(x) if random.random() < val]
        else:
            # m-set problem
            order = np.argsort(-x)
            included = np.copy(x[order])
            remaining = 1.0 - included
            outer_samples = [w for w in self.split_sample(included, remaining)]
            weights = list(map(lambda z: z[0], outer_samples))
            _, left, right = outer_samples[np.random.choice(len(outer_samples), p=weights)]
            if left == right - 1:
                sample = range(self.m_size)
            else:
                candidates = [i for i in range(left, right)]
                random.shuffle(candidates)
                sample = [i for i in range(left)] + candidates[:self.m_size - left]
            action = [order[i] for i in sample]
            return action

    def split_sample(self, included, remaining):
        """

        :param included: remaining marginal probabilities of sampling a coordinate
        :param remaining: remaining marginal probabilities of not sampling a coordinate
        :return: remaining sampling distributions
        """
        prop = 1.0
        left, right = 0, self.dim
        i = self.dim
        while left < right:
            i -= 1
            active = (self.m_size - left) / (right - left)
            inactive = 1.0 - active
            if active == 0 or inactive == 0:
                yield (prop, left, right)
                return
            weight = min(included[right - 1] / active, remaining[left] / inactive)
            yield weight, left, right
            prop -= weight
            assert prop >= -EPS
            included -= weight * active
            remaining -= weight * inactive
            while right > 0 and included[right - 1] <= EPS:
                right -= 1
            while left < self.dim and remaining[left] <= EPS:
                left += 1
            assert right - left <= i
        if prop > 0.0:
            yield (prop, self.m_size, self.m_size + 1)


class ThompsonSampling(Bandit):
    """
    Implementation of Thompson Sampling for Combinatorial Semi-bandits as a baseline algorithm.
    https://arxiv.org/pdf/1803.04623.pdf
    """

    def __init__(self, dim, action_set, m_size):
        super().__init__(action_set, dim, m_size)
        self.a = np.ones(self.dim)
        self.b = np.ones(self.dim)

    def next(self, ):
        theta = np.zeros(self.dim)
        for i in range(self.dim):
            theta[i] = np.random.beta(self.a[i], self.b[i])  # the larger, the better
        return self.oracle(theta)

    def oracle(self, theta):
        if self.unconstrained:
            return [i for i in range(self.dim) if theta[i] > 0.5]
        else:
            order = np.argsort(theta)  # inreasing order
            return order[self.dim - self.m_size:]  # pick the m largest ones

    def update(self, action, feedback):
        for i in range(len(action)):
            arm = action[i]
            assert abs(feedback[i]) <= 1.0
            if random.random() >= (feedback[i] + 1.0) / 2.0:
                self.a[arm] += 1
            else:
                self.b[arm] += 1

    def reset(self):
        self.a = np.ones(self.dim)
        self.b = np.ones(self.dim)


class CombUCB(Bandit):
    """
    Implementation of CombUCB as a baseline algorithm
    https://arxiv.org/pdf/1502.03475.pdf
    """

    def __init__(self, dim, action_set, m_size):
        super().__init__(action_set, dim, m_size)
        self.t = 0
        self.te = np.zeros(self.dim)  # the T(e) in the paper: number of observations of arm e
        self.emp_sum = np.zeros(self.dim)

    def next(self, ):
        self.t += 1
        if self.unconstrained and self.t <= 1:  # full set: explore all arms in the first round
            return range(self.dim)
        elif (not self.unconstrained) and self.t <= int(self.dim + self.m_size - 1) / int(
                self.m_size):  # m-set: explore all arms in ceil(d/m) rounds
            if self.t * self.m_size <= self.dim:
                return range((self.t - 1) * self.m_size, self.t * self.m_size)
            else:
                return range(-self.m_size, 0)
        else:
            conf_width = 2 * np.sqrt(np.divide(1.5 * np.log(self.t), self.te))
            emp_avg = np.divide(self.emp_sum, self.te)
            lower_conf = emp_avg - conf_width
            return self.oracle(lower_conf)

    def oracle(self, lower_conf):
        if self.unconstrained:
            return [i for i in range(self.dim) if lower_conf[i] < 0]
        else:
            order = np.argsort(lower_conf)  # increasing order
            return order[0:self.m_size]

    def update(self, action, feedback):
        for i in range(len(action)):
            arm = action[i]
            self.emp_sum[arm] += feedback[i]
            self.te[arm] += 1

    def reset(self):
        self.t = 0
        self.te = np.zeros(self.dim)  # the T(e) in the paper: number of observations of arm e
        self.emp_sum = np.zeros(self.dim)


class OSMD(Bandit):
    """
    Container for shared utility of algorithms based on Online Stochastic Mirror Descent
    """
    L = None
    x = None
    time_step = 0.0
    gamma = 1.0
    learning_rate = None
    bias = None

    def __init__(self, dim, action_set, m_size):
        super().__init__(action_set, dim, m_size)

    def next(self, ):
        self.time_step += 1.0
        self.learning_rate = self.get_learning_rate(self.time_step)
        self.solve_optimization()
        return self.sample_action(self.x)

    def reset(self):
        self.L = np.array([0.0] * self.dim)
        self.x = [self.m_size / self.dim if not self.unconstrained else 0.5 for _ in range(self.dim)]
        self.time_step = 0.0
        self.bias = 0

    def update(self, action, feedback):
        if len(action):
            self.L[action] += np.divide((np.array(feedback) + 1.0), self.x[action])
        if self.unconstrained:
            self.L -= 1
        else:
            self.L += self.bias
            self.bias = 0

    def solve_optimization(self):
        if self.unconstrained:
            self.x = np.array([self.solve_unconstrained(l * self.learning_rate, x) for l, x in zip(self.L, self.x)])
        else:
            max_iter = 100
            iteration = 0
            upper = None
            lower = None
            step_size = 1
            while True:
                iteration += 1
                self.x = np.array(
                    [self.solve_unconstrained((l + self.bias) * self.learning_rate, x) for l, x in zip(self.L, self.x)])
                f = self.x.sum() - self.m_size
                df = self.hessian_inverse()
                next_bias = self.bias + f / df
                if f > 0:
                    lower = self.bias
                    self.bias = next_bias
                    if upper is None:
                        step_size *= 2
                        if next_bias > lower + step_size:
                            self.bias = lower + step_size
                    else:
                        if next_bias > upper:
                            self.bias = (lower + upper) / 2
                else:
                    upper = self.bias
                    self.bias = next_bias
                    if lower is None:
                        step_size *= 2
                        if next_bias < upper - step_size:
                            self.bias = upper - step_size
                    else:
                        if next_bias < lower:
                            self.bias = (lower + upper) / 2
                if iteration > max_iter or abs(f) < 100 * EPS:
                    break

            assert iteration < max_iter

    def get_learning_rate(self, time):
        return 1.0 / np.sqrt(time)

    def solve_unconstrained(self, loss, warmstart):
        raise NotImplementedError

    def hessian_inverse(self):
        raise NotImplementedError


class BobOSMD(OSMD):
    """
    Implementation of our HYBRID algorithm; the algorithm to be evaluated.
    https://arxiv.org/pdf/1901.08779.pdf
    """

    def __init__(self, dim, action_set, m_size):
        super().__init__(dim, action_set, m_size)
        if m_size is None or m_size < dim / 2:
            self.gamma = 1.0
        else:
            self.gamma = np.sqrt(1.0 / np.log(dim - (dim - m_size)))

    def solve_unconstrained(self, loss, warmstart):
        x_val, func_val, dif_func_val, dif_x = warmstart, 1.0, float('inf'), 1.0

        while True:
            func_val = loss - 0.5 / np.sqrt(x_val) + self.gamma * (1.0 - np.log(1.0 - x_val))
            dif_func_val = 0.25 / (np.sqrt(x_val) ** 3) + self.gamma / (1.0 - x_val)
            dif_x = func_val / dif_func_val
            if dif_x > x_val:
                dif_x = x_val / 2
            elif dif_x < x_val - 1.0:
                dif_x = (x_val - 1.0) / 2
            if abs(dif_x) < EPS:
                break
            x_val -= dif_x
        return x_val

    def hessian_inverse(self):
        return (1.0 / (0.25 / np.power(self.x, 1.5) + self.gamma / (1.0 - self.x))).sum() * self.learning_rate


class BarrierOSMD(OSMD):
    """
    Implementaton of BARRIER as a baseline algorithm.
    https://arxiv.org/pdf/1801.03265.pdf
    """

    def solve_unconstrained(self, loss, warmstart):
        return 1.0 if loss <= 0 else min(1.0, 1.0 / loss)

    def hessian_inverse(self):
        return max(EPS, ((self.x < 1.0) * np.power(self.x, 2)).sum() * self.learning_rate)

    def get_learning_rate(self, time):
        return 4 * np.sqrt(np.log(1.0 + time) / time)


class ShannonOSMD(OSMD):
    """
    Implementation of CombEXP3 as a baseline algorithm.
    https://arxiv.org/pdf/1502.03475.pdf
    """

    def solve_unconstrained(self, loss, warmstart):
        return min(1.0, np.exp(-loss))

    def hessian_inverse(self):
        return max(EPS, ((self.x < 1.0) * self.x).sum() * self.learning_rate)

    def get_learning_rate(self, time):
        return 0.25 / np.sqrt(time)
