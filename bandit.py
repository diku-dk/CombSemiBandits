import numpy as np
import random

EPS = 10E-12


class Bandit:

    def __init__(self, action_set, d, m):
        self.d = d
        self.m = m
        if action_set == "full":
            self.unconstrained = True
        elif action_set == "m-set":
            self.unconstrained = False
        else:
            raise Exception("Invalid action set %s for OSMD, abort." % action_set)

    def next(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def update(self, action, feedback):
        raise NotImplementedError

    def sample_action(self, x):
        if self.unconstrained:
            return [i for i, val in enumerate(x) if random.random() < val]
        else:
            # m-set problem
            order = np.argsort(-x)
            included = np.copy(x[order])
            remaining = 1.0 - included
            outer_samples = [w for w in self.split_sample(included, remaining)]
            weights = list(map(lambda x: x[0], outer_samples))
            _, left, right = outer_samples[np.random.choice(len(outer_samples), p=weights)]
            if left == right - 1:
                sample = range(self.m)
            else:
                candidates = [i for i in range(left, right)]
                random.shuffle(candidates)
                sample = [i for i in range(left)] + candidates[:self.m - left]
            action = [order[i] for i in sample]
            return action

    def split_sample(self, included, remaining):
        prop = 1.0
        left, right = 0, self.d
        i = self.d
        while left < right:
            i -= 1
            active = (self.m - left) / (right - left)
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
            while left < self.d and remaining[left] <= EPS:
                left += 1
            assert right - left <= i
        if prop > 0.0:
            yield (prop, self.m, self.m + 1)


class ThompsonSampling(Bandit):  # based on the paper "Thompson Sampling for Combinatorial Semi-bandits"
    def __init__(self, d, action_set, m):
        super().__init__(action_set, d, m)
        self.a = np.ones(self.d)
        self.b = np.ones(self.d)

    def next(self, ):
        theta = np.zeros(self.d)
        for i in range(self.d):
            theta[i] = np.random.beta(self.a[i], self.b[i])  # the larger, the better
        return self.oracle(theta)

    def oracle(self, theta):
        if self.unconstrained:
            return [i for i in range(self.d) if theta[i] > 0.5]
        else:
            order = np.argsort(theta)  # inreasing order
            return order[self.d - self.m:]  # pick the m largest ones

    def update(self, action, feedback):
        for i in range(len(action)):
            arm = action[i]
            assert abs(feedback[i]) <= 1.0
            if random.random() >= (feedback[i] + 1.0) / 2.0:
                self.a[arm] += 1
            else:
                self.b[arm] += 1

    def reset(self):
        self.a = np.ones(self.d)
        self.b = np.ones(self.d)


class CombUCB(Bandit):  # based on the paper ""
    def __init__(self, d, action_set, m):
        super().__init__(action_set, d, m)
        self.t = 0
        self.te = np.zeros(self.d)  # the T(e) in the paper: number of observations of arm e
        self.emp_sum = np.zeros(self.d)

    def next(self, ):
        self.t += 1
        if self.unconstrained and self.t <= 1:  # full set: explore all arms in the first round
            return range(self.d)
        elif (not self.unconstrained) and self.t <= int(self.d + self.m - 1) / int(
                self.m):  # m-set: explore all arms in ceil(d/m) rounds
            if self.t * self.m <= self.d:
                return range((self.t - 1) * self.m, self.t * self.m)
            else:
                return range(-self.m, 0)
        else:
            conf_width = 2 * np.sqrt(np.divide(1.5 * np.log(self.t), self.te))
            emp_avg = np.divide(self.emp_sum, self.te)
            lower_conf = emp_avg - conf_width
            return self.oracle(lower_conf)

    def oracle(self, lower_conf):
        if self.unconstrained:
            return [i for i in range(self.d) if lower_conf[i] < 0]
        else:
            order = np.argsort(lower_conf)  # increasing order
            return order[0:self.m]

    def update(self, action, feedback):
        for i in range(len(action)):
            arm = action[i]
            self.emp_sum[arm] += feedback[i]
            self.te[arm] += 1

    def reset(self):
        self.t = 0
        self.te = np.zeros(self.d)  # the T(e) in the paper: number of observations of arm e
        self.emp_sum = np.zeros(self.d)


class OSMD(Bandit):
    L = None
    x = None
    t = 0.0
    gamma = 1.0
    learning_rate = None
    bias = None

    def __init__(self, d, action_set, m):
        super().__init__(action_set, d, m)

    def next(self, ):
        self.t += 1.0
        self.learning_rate = self.get_learning_rate(self.t)
        self.solve_optimization()
        return self.sample_action(self.x)

    def reset(self):
        self.L = np.array([0.0] * self.d)
        self.x = [self.m / self.d if not self.unconstrained else 0.5 for _ in range(self.d)]
        self.t = 0.0
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
                f = self.x.sum() - self.m
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

    def get_learning_rate(self, t):
        return 1.0 / np.sqrt(t)

    def solve_unconstrained(self, loss, warmstart):
        raise NotImplementedError

    def hessian_inverse(self):
        raise NotImplementedError


class BobOSMD(OSMD):

    def __init__(self, d, action_set, m):
        super().__init__(d, action_set, m)
        if m is None or m < d / 2:
            self.gamma = 1.0
        else:
            self.gamma = np.sqrt(1.0 / np.log(d - (d - m)))

    def solve_unconstrained(self, loss, warmstart):
        x, fx, dfx, dx = warmstart, 1.0, float('inf'), 1.0

        while True:
            fx = loss - 0.5 / np.sqrt(x) + self.gamma * (1.0 - np.log(1.0 - x))
            dfx = 0.25 / (np.sqrt(x) ** 3) + self.gamma / (1.0 - x)
            dx = fx / dfx
            if dx > x:
                dx = x / 2
            elif dx < x - 1.0:
                dx = (x - 1.0) / 2
            if abs(dx) < EPS:
                break
            x -= dx
        return x

    def hessian_inverse(self):
        return (1.0 / (0.25 / np.power(self.x, 1.5) + self.gamma / (1.0 - self.x))).sum() * self.learning_rate


class BarrierOSMD(OSMD):

    def solve_unconstrained(self, loss, warmstart):
        # TODO adjust learning rate?
        return 1.0 if loss <= 0 else min(1.0, 1.0 / loss)

    def hessian_inverse(self):
        return max(EPS, ((self.x < 1.0) * np.power(self.x, 2)).sum() * self.learning_rate)

    def get_learning_rate(self, t):
        return 4 * np.sqrt(np.log(1.0 + t) / t)


class ShannonOSMD(OSMD):

    def solve_unconstrained(self, loss, warmstart):
        """
        TODO adjust learning rate?
        """
        return min(1.0, np.exp(-loss))

    def hessian_inverse(self):
        return max(EPS, ((self.x < 1.0) * self.x).sum() * self.learning_rate)

    def get_learning_rate(self, t):
        return 0.25 / np.sqrt(t)