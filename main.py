"""
Combinatorial Semi-bandit experiments suite
"""
import random
import argparse
import os
from time import time
import numpy as np
from environment import Adversarial, Stochastic
from bandit import BarrierOSMD, ShannonOSMD, CombUCB, ThompsonSampling, BobOSMD

parser = argparse.ArgumentParser(description='Run semi-bandit simulations.')
parser.add_argument('-a', dest='alg', required=True,
                    choices=['BobOSMD', 'ShannonOSMD', 'BarrierOSMD', 'ThompsonSampling', 'CombUCB'],
                    help='Bandit algorithm.')
parser.add_argument('-g', type=float, dest='gap', default='0.25',
                    help='Gap between optimal and suboptimal dimensions.')
parser.add_argument('-d', type=int,
                    help='Dimension of the action space.', default=4)
parser.add_argument('-m', type=int,
                    help='Num of active elements in M-set problem.')
parser.add_argument('-e', type=str, dest='env', default='sto',
                    help='Choose environment sto or adv.')
parser.add_argument('-t', type=int, default=10000,
                    help='Time horizon T.')
parser.add_argument('-s', type=int, dest="seed",
                    help='Random seed.')
parser.add_argument('-r', type=int, dest="runs",
                    help='Repetitions of the experiment.', default=1)
parser.add_argument('-o', type=int,
                    help='File ending if runs are broken into parts')


def setup_game(dim=7, m_size=3, gap=0.2, mode="stochastic", action_set="m-set", algorithm="BarrierOSMD", n_steps=100):
    """

    :param dim: int, dimension of the semi-bandit problem
    :param m_size: int, number of actions an agent can pick simultaneously
    :param gap: List[Float], vector of sub-optimality gaps of all arms
    :param mode: string, stochastic or adversarial environment
    :param action_set: string, m-set or full action set
    :param algorithm: string, name of the bandit algorithm
    :param n_steps: int, time horizon
    :return: bandit, environment objects already initiated the given settings
    """
    if algorithm == "BobOSMD":
        bandit = BobOSMD(dim, action_set, m_size)
    elif algorithm == "ShannonOSMD":
        bandit = ShannonOSMD(dim, action_set, m_size)
    elif algorithm == "BarrierOSMD":
        bandit = BarrierOSMD(dim, action_set, m_size)
    elif algorithm == "ThompsonSampling":
        bandit = ThompsonSampling(dim, action_set, m_size)
    elif algorithm == "CombUCB":
        bandit = CombUCB(dim, action_set, m_size)
    else:
        raise Exception("Invalid algorithm %s, abort." % algorithm)

    if mode in ['sto', 'stochastic']:
        environment = Stochastic(action_set, gap, dim, m_size, n_steps)
    elif mode in ['adv', 'adversarial']:
        environment = Adversarial(action_set, gap, dim, m_size, n_steps)

    else:
        raise Exception("Invalid mode %s, abort." % mode)

    print("Initialize %s in %s environment with dimension %d and %s action set%s. The gaps are set to %f." %
          (algorithm, mode, dim, action_set, " with m=%s" % m_size if m_size else "", gap))

    return bandit, environment


def run_simulation(bandit, environment, n_steps, snapshots):
    """

    :param bandit: algorithm to be evaluated
    :param environment: stochastic or adversarial test environment
    :param n_steps: time horizon
    :param snapshots: time positions where the regret is tracked
    :return: np.array of empirical pseudo-regret at snapshots
    """
    pseudo_regret = []
    bandit.reset()
    environment.reset()
    regret = 0
    last_print = time()
    for time_step in range(1, n_steps + 1):
        action = bandit.next()
        feedback, cur_regret = environment.play(action, time_step)

        bandit.update(action, np.array(feedback))
        regret += cur_regret
        # print(regret)
        if time_step in snapshots:
            pseudo_regret.append(regret)
        if time() - last_print > 30:
            print("finished t=", time_step)
            last_print = time()

    return np.array(pseudo_regret)


def get_plot_points(n_steps):
    """

    :param n_steps: time horizon
    :return: List[int] first 10000 time steps in quadratic grid, then exponential
    """
    points = []
    for i in range(1, 101):
        if i * i < n_steps:
            points.append(i * i)
        else:
            break
    next_point = 10000
    while True:
        next_point *= 2
        if next_point < n_steps:
            points.append(next_point)
        else:
            break
    points.append(n_steps)
    return points


def filename(conf):
    """

    :param conf: simulation settings
    :return: String, filename for logging the regret
    """
    return "%s_%s_%d_%f_%d%s" % (conf.alg, conf.env, conf.d, conf.gap, conf.m if conf.m else 0,
                                 ".%d" % conf.o if conf.o else "")


if __name__ == '__main__':
    ARGS = parser.parse_args()
    if ARGS.seed is not None:
        SEED = ARGS.seed
    else:
        SEED = round(random.random() * 100000)
    print("Start simulation with seed: %d." % SEED)
    if ARGS.m is None:
        SETTING = 'full'
    else:
        SETTING = 'm-set'
    random.seed(SEED)
    np.random.seed(SEED)

    ban, env = setup_game(dim=ARGS.d, m_size=ARGS.m, gap=ARGS.gap, mode=ARGS.env, action_set=SETTING, algorithm=ARGS.alg,
                          n_steps=ARGS.t)
    # reduce the number of snapshots of the regret
    plot_points = get_plot_points(ARGS.t)
    reg = np.zeros(shape=(len(plot_points), ARGS.runs))
    for run in range(ARGS.runs):
        print("start run %s" % run)
        reg[:, run] = run_simulation(ban, env, ARGS.t, plot_points)

    mean = reg.mean(1)
    std = reg.std(1)

    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    os.chdir(DIR_PATH + "/../")

    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/' + filename(ARGS), 'w') as file:
        file.write("# used seed %d with %d runs\n" % (SEED, ARGS.runs))
        file.writelines(["%d %d %d\n" % (plot_points[i], mean[i], std[i]) for i in range(len(plot_points))])
