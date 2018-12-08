# Reward Shaping using Difference Rewards (D++)
The underlying architecture is MADDPG, and I am modifying the Multiagent Parlicle Environment such that it
returns the shaped reward. Thus take a look at "MPE_custom" repository > dpprs branch.

This is a work in progress.

# MADDPG-PyTorch
PyTorch Implementation of MADDPG is taken from Shariq Iqbal.

## Requirements

* [OpenAI baselines](https://github.com/openai/baselines), commit hash: 98257ef8c9bd23a24a330731ae54ed086d9ce4a7
* My [fork](https://github.com/shariqiqbal2810/multiagent-particle-envs) of Multi-agent Particle Environments
* [PyTorch](http://pytorch.org/), version: 0.3.0.post4
* [OpenAI Gym](https://github.com/openai/gym), version: 0.9.4
* [Tensorboard](https://github.com/tensorflow/tensorboard), version: 0.4.0rc3 and [Tensorboard-Pytorch](https://github.com/lanpa/tensorboard-pytorch), version: 1.0 (for logging)

The versions are just what I used and not necessarily strict requirements.

## How to Run

All training code is contained within `main.py`. To view options simply run:

```
python main.py --help
```

