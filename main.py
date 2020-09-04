import random
import numpy as np
import argparse
from time import time
import os
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


def setup_game(d=7, m=3, gap=0.2, mode="stochastic", action_set="m-set", algorithm="BarrierOSMD", n=100):
    """

    :param d: int, dimension of the semi-bandit problem
    :param m: int, number of actions an agent can pick simultaneously
    :param gap: List[Float], vector of sub-optimality gaps of all arms
    :param mode: string, stochastic or adversarial environment
    :param action_set: string, m-set or full action set
    :param algorithm: string, name of the bandit algorithm
    :param n: int, time horizon
    :return: bandit, environment objects already initiated the given settings
    """
    if algorithm == "BobOSMD":
        bandit = BobOSMD(d, action_set, m)
    elif algorithm == "ShannonOSMD":
        bandit = ShannonOSMD(d, action_set, m)
    elif algorithm == "BarrierOSMD":
        bandit = BarrierOSMD(d, action_set, m)
    elif algorithm == "ThompsonSampling":
        bandit = ThompsonSampling(d, action_set, m)
    elif algorithm == "CombUCB":
        bandit = CombUCB(d, action_set, m)
    else:
        raise Exception("Invalid algorithm %s, abort." % algorithm)

    if mode in ['sto', 'stochastic']:
        environment = Stochastic(action_set, gap, d, m, n)
    elif mode in ['adv', 'adversarial']:
        environment = Adversarial(action_set, gap, d, m, n)

    else:
        raise Exception("Invalid mode %s, abort." % mode)

    print("Initialize %s in %s environment with dimension %d and %s action set%s. The gaps are set to %f." %
          (algorithm, mode, d, action_set, " with m=%s" % m if m else "", gap))

    return bandit, environment


def run_simulation(bandit, environment, n, snapshots):
    """

    :param bandit: algorithm to be evaluated
    :param environment: stochastic or adversarial test environment
    :param n: time horizon
    :param snapshots: time positions where the regret is tracked
    :return: np.array of empirical pseudo-regret at snapshots
    """
    pseudo_regret = []
    bandit.reset()
    environment.reset()
    regret = 0
    last_print = time()
    for t in range(1, n + 1):
        action = bandit.next()
        feedback, r = environment.play(action, t)

        bandit.update(action, np.array(feedback))
        regret += r
        # print(regret)
        if t in snapshots:
            pseudo_regret.append(regret)
        if time() - last_print > 30:
            print("finished t=", t)
            last_print = time()

    return np.array(pseudo_regret)


def get_plot_points(n):
    """

    :param n: time horizon
    :return: List[int] first 10000 time steps in quadratic grid, then exponential
    """
    points = []
    for i in range(1, 101):
        if i * i < n:
            points.append(i * i)
        else:
            break
    next_point = 10000
    while True:
        next_point *= 2
        if next_point < n:
            points.append(next_point)
        else:
            break
    points.append(n)
    return points


def filename(args):
    return "%s_%s_%d_%f_%d%s" % (args.alg, args.env, args.d, args.gap, args.m if args.m else 0,
                                 ".%d" % args.o if args.o else "")


if __name__ == '__main__':
    args = parser.parse_args()
    if args.seed is not None:
        seed = args.seed
    else:
        seed = round(random.random() * 100000)
    print("Start simulation with seed: %d." % seed)
    if args.m is None:
        setting = 'full'
    else:
        setting = 'm-set'
    random.seed(seed)
    np.random.seed(seed)

    ban, env = setup_game(d=args.d, m=args.m, gap=args.gap, mode=args.env, action_set=setting, algorithm=args.alg,
                          n=args.t)
    # reduce the number of snapshots of the regret
    plot_points = get_plot_points(args.t)
    reg = np.zeros(shape=(len(plot_points), args.runs))
    for run in range(args.runs):
        print("start run %s" % run)
        reg[:, run] = run_simulation(ban, env, args.t, plot_points)

    mean = reg.mean(1)
    std = reg.std(1)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path + "/../")

    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/' + filename(args), 'w') as file:
        file.write("# used seed %d with %d runs\n" % (seed, args.runs))
        file.writelines(["%d %d %d\n" % (plot_points[i], mean[i], std[i]) for i in range(len(plot_points))])
