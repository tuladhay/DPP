import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from neuroevolution.neuro_evolution import Evo

USE_CUDA = torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=discrete_action)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)

    # create evo instance with population of neural network agents
    evo = Evo.init_from_env(env, config)
    evo.initialize_fitness()
    # Put evo population into world agents
    # env.envs[0].world.policy_agents = evo
    # Start training
    agent_num = 1
    for generation in range(config.n_episodes):
        """In every generation, the population is evaluated, ranked, mutated, and re-inserted into population """
        evo.evaluate_pop(env)
        evo.rank_pop_selection_mutation(env)

        # logger.add_scalar('global_reward' % '1', evo.best_policy.fitness, generation)
        logger.add_scalar('agent%i/mean_episode_rewards' % agent_num, evo.best_policy.fitness, generation)

        if not generation % 100:
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            filename = run_dir / 'incremental' / ('model_ep%i.pt' % (generation + 1))
            save_dict = {'init_dict': evo.init_dict,
                         'agent_params': [evo.best_policy.get_params()]}
            torch.save(save_dict, filename)

    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default='simple_spread', help="Name of environment")
    parser.add_argument("--model_name", default='Exp',
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--n_episodes", default=100000, type=int)
    parser.add_argument("--save_interval", default=2000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--episode_length", default=60, type=int)
    parser.add_argument("--discrete_action",
                        action='store_true')
    parser.add_argument("--n_population",
                        default=10, type=int)
    parser.add_argument("--n_tournament",
                        default=0.3, type=int)
    parser.add_argument("--noise_stddev",
                        default=0.1, type=int)

    config = parser.parse_args()
    # Initialize from saved model?
    init_from_saved = False
    config.discrete_action = False
    model_path = ""

    run(config)
