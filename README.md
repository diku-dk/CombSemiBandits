# CombSemiBandits
Code for [Beating Stochastic and Adversarial Semi-bandits Optimally and Simultaneously](https://scholar.google.com/scholar_url?url=https://arxiv.org/pdf/1901.08779)

It evaluates the following combinatorial bandit algorithms
* [HYBRID](https://arxiv.org/pdf/1901.08779.pdf)
* [Thompson Sampling](https://arxiv.org/pdf/1803.04623.pdf)
* [CombUCB](https://arxiv.org/pdf/1502.03475.pdf)
* [CombEXP3](https://arxiv.org/pdf/1502.03475.pdf)
* [BROAD](https://arxiv.org/pdf/1801.03265.pdf)

It supports two environment with Bernoulli random variables. The stochastic environment has means centered around 0.5 and is identically distributed. The adversarial environment is changing the means over time, iterating between phases of large expected loss and small expected loss. 

Call the script with: main.py \[-h\] -a algorithm
               \[-g gap\] \[-d dim\] \[-m m-set\] \[-e env\] \[-t time\] \[-s seed\] \[-r runs\]

mandatory argument:

    -a  Name of Algorithm

optional arguments:

  \-h, --help            show this help message and exit

  \-g                 Gap between optimal and suboptimal components.

  \-d                   Dimension of the action space.

  \-m                   Num of active elements in M-set problem.

  \-e                 Choose environment sto or adv.

  \-t                   Time horizon T.

  \-s                Random seed.

  \-r                Number of repetitions of the experiment.



