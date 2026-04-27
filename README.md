This code is modified from MADDPG, RMA-AC, and M3DDPG.

# Robust Multi-Agent Reinforcement Learning with State Uncertainty

This is the code for implementing the Robust Multi-Agent Actor-Critic (RMAAC) algorithm presented in the paper:
[Robust Multi-Agent Reinforcement Learning with State Uncertainty](https://openreview.net/forum?id=CqTkapZ6H9).
It is configured to be run in conjunction with environments from the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

The current workspace uses a single unified launcher in [RMA-AC/experiments/train.py](RMA-AC/experiments/train.py) for all supported training modes:

- `maddpg-none`: plain MADDPG
- `maddpg-earnie`: MADDPG with EARNIE regularization
- `maddpg-act_adv`: action-adversarial training
- `maddpg-obs_adv`: state-adversarial training
- `m3ddpg`: adversarial M3DDPG-style training


## Installation

- Known dependencies: Python (3.5.4), OpenAI gym (0.10.5), tensorflow (1.8.0), numpy (1.14.5)

You can use the following commands to configure the environment.

`conda create -n rmaac_env python=3.5.4`

`conda activate rmaac_env`

`conda install numpy=1.14.5`

`# conda install -c anaconda tensorflow-gpu`

`conda install tensorflow`

`# conda install gym=0.10.5`

`pip install gym==0.10.5`

## Multi-Agent Particle Environments

We demonstrate here how the code can be used in conjunction with the
[Multi-Agent Particle Environments (MPE)](https://github.com/openai/multiagent-particle-envs).

- Download and install the MPE code [here](https://github.com/openai/multiagent-particle-envs)
by following the `README`.

- Ensure that `multiagent-particle-envs` has been added to your `PYTHONPATH` (e.g. in `~/.bashrc` or `~/.bash_profile`).

## Unified Training and Evaluation

Use the RMA-AC launcher as the single entry point for training and evaluation:

```bash
cd RMA-AC/experiments
python train.py --scenario simple --mode train --variant maddpg-none
```

Replace `maddpg-none` with any of the supported variants listed above. A few common examples:

```bash
# Standard MADDPG
python train.py --scenario simple_tag --mode train --variant maddpg-none

# MADDPG with EARNIE regularization
python train.py --scenario simple_tag --mode train --variant maddpg-earnie --use-ernie

# RMA-AC action adversary
python train.py --scenario simple_tag --mode train --variant maddpg-act_adv

# RMA-AC state adversary
python train.py --scenario simple_tag --mode train --variant maddpg-obs_adv

# M3DDPG-style adversarial training
python train.py --scenario simple_tag --mode train --variant m3ddpg
```

For evaluation, use the same launcher with `--mode test`:

```bash
python train.py --scenario simple_tag --mode test --variant maddpg-none
python train.py --scenario simple_tag --mode test --variant maddpg-earnie
python train.py --scenario simple_tag --mode test --variant maddpg-act_adv
python train.py --scenario simple_tag --mode test --variant maddpg-obs_adv
python train.py --scenario simple_tag --mode test --variant m3ddpg
```

You can replace `simple` or `simple_tag` with any MPE scenario that exists in your environment.

## Command-line options

### Environment options

- `--scenario`: defines which environment in the MPE is to be used (default: `"simple"`)

- `--max-episode-len` maximum length of each episode for the environment (default: `25`)

- `--num-episodes` total number of training episodes (default: `60000`)

- `--num-adversaries` number of adversaries in the game (default: `0`)


### Core training parameters

- `--lr`: learning rate for agents (default: `1e-2`)

- `--lr-adv`: learning rate for state perturbation adversaries(default: `1e-2`)

- `--gamma`: discount factor (default: `0.95`)

- `--batch-size`: batch size (default: `1024`)

- `--num-units`: number of units in the MLP (default: `64`)

- `--noise-type`: noise format (default: `Linear`)

- `--noise-variance`: variance of gaussian noise (default: `1`)

- `--constraint-epsilon`: the constraint parameter (default: `0.5`)

### Checkpointing

- `--exp-name`: name of the experiment, used as the file name to save all results (default: `None`)

- `--save-dir`: directory where intermediate training results and model will be saved (default: `"/tmp/policy/"`)

- `--save-rate`: model is saved every time this number of episodes has been completed (default: `1000`)

- `--load-dir`: directory where training state and model are loaded from (default: `""`)

### Evaluation

- `--restore`: restores previous training state stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), and continues training (default: `False`)

- `--display`: displays to the screen the trained policy stored in `load-dir` (or in `save-dir` if no `load-dir`
has been provided), but does not continue training (default: `False`)

- `--benchmark`: runs benchmarking evaluations on saved policy, saves results to `benchmark-dir` folder (default: `False`)

- `--benchmark-iters`: number of iterations to run benchmarking for (default: `100000`)

- `--benchmark-dir`: directory where benchmarking data is saved (default: `"./benchmark_files/"`)

- `--plots-dir`: directory where training curves are saved (default: `"./learning_curves/"`)


## Paper citation

If you used this code for your experiments or found it helpful, consider citing the following paper:

<pre>
@article{
he2023robust,
title={Robust Multi-Agent Reinforcement Learning with State Uncertainty},
author={Sihong He, Songyang Han, Sanbao Su, Shuo Han, Shaofeng Zou, and Fei Miao},
journal={Transactions on Machine Learning Research},
year={2023},
url={https://openreview.net/forum?id=CqTkapZ6H9}
}
</pre>
