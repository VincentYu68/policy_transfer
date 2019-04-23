# Learning Transferrable and Adaptive Control Policies

This is code for the following papers: 

<a href="https://arxiv.org/abs/1810.05751">Policy Transfer with Strategy Optimization</a> 

<a href="https://arxiv.org/abs/1702.02453">Prepare for the Unknown: Learning a Universal Policy with Online System Identification</a>

## Prerequisites

To use this code you need to install <a href="https://github.com/openai/baselines">OpenAI Baselines</a>, <a href="http://dartsim.github.io/">Dart</a> and <a href="http://pydart2.readthedocs.io/en/latest/">PyDart2</a>.

You can find detailed instructions for installing OpenAI Baselines <a href="https://github.com/openai/baselines">here</a>. For installing Dart and PyDart2, you can follow the installation instructions <a href="https://github.com/DartEnv/dart-env/wiki">here</a>.

Note that the environments also depends on <a href="https://github.com/openai/gym">OpenAI Gym</a>, however it should come with Baselines.

## Installation

Run the following command from the project directory:

```bash
pip install -e .
```


## How to use

### SO-CMA

SO-CMA has two stages: training universal policy and strategy optimization.

To train a universal policy, use the code in [ppo](policy_transfer/ppo).
FOr the strategy optimization part, use the code in [test_socma](policy_transfer/policy_transfer_strategy_optimization/test_socma.py).

An example of Dart hopper transferred to MuJoCo hopper can be found in [examples](examples):

```bash
examples/socma_hopper_5d_train.sh
```

The training results will be saved to data/.

To perform strategy optimization, run:

```bash
examples/socma_hopper_5d_test.sh
```

You can also use [test_policy.py](policy_transfer/test_policy.py) to test individual policies.

### UP-OSI

Training UP-OSI involves two steps: training a universal policy and training an online system identification model.

To train a universal policy, use the code in [ppo](policy_transfer/ppo).
To train the online system identification model, use the code in [train_osi](policy_transfer/uposi/train_osi.py).

An example training script for the hopper environment is available in [examples](examples), use the following command to run the example training script:

```bash
examples/uposi_hopper_2d_train.sh
```

The training results will be saved to data/.

To test the resulting controller, run:

```bash
examples/uposi_hopper_2d_test.sh
```

and follow the prompt in the terminal. After each rollout a plot of the estimated model parameters and true model parameters is shown.


## ODE Internal Error
If you see errors like: ODE INTERNAL ERROR 1: assertion "d[i] != dReal(0.0)" failed in _dLDLTRemove(), try downloading [lcp.cpp](https://drive.google.com/file/d/1MCho3QBtyPhSoKNV77VFOvCqsMJPk3NF/view) and replace the one in dart/external/odelcpsolver/ with it. Recompile Dart and Pydart2 afterward and the issue should be gone.

## Additional feedbacks:
Please contact Wenhao Yu (wenhaoyu@gatech.edu) if you have any feedbacks/questions about this work.
