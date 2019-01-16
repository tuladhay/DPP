import torch
import random
from operator import attrgetter
import copy
import gym
import numpy as np
from utils.agents import DDPGAgent
import pickle
import fastrand, math

class Evo:
    def __init__(self, config, agent_init_params, discrete_action=False, comm_acs_space=2):
        '''
        :param num_evo_actors: This is the number of genes/actors you want to have in the population
        :param evo_episodes: This is the number of evaluation episodes for each gene. See Algo1: 7, and Table 1
        population: initalizes 10 genes/actors
        num_elites: number of genes/actor that are selected, and do not undergo mutation (unless they are
                    selected again in the tournament selection
        tournament_genes: number of randomly selected genes to take the max(fitness) from,
                        and then put it back into the population

        noise_mean: mean for the gaussian noise for mutation
        noise_stddev: standard deviation for the gaussian noise for mutation
        '''
        self.num_actors = config.n_population
        self.population = [DDPGAgent(lr=0.01, discrete_action=discrete_action,
                                 hidden_dim=config.hidden_dim,
                                 comm_acs_space=comm_acs_space,
                                 #**params)
                                 **agent_init_params[0])  # sending the first set of params. All agents are init same
                                 for n in range(self.num_actors)]
        print("Initializing Evolutionary Agents")
        self.elite_percentage = 0.1
        self.num_elites = int(self.elite_percentage * self.num_actors)
        self.tournament_genes = 3
        # TODO: make it a percentage
        # self.tournament_genes = config.n_tournament
        self.noise_mean = 0.0
        self.noise_stddev = config.noise_stddev
        self.save_fitness = []
        self.best_policy = copy.deepcopy(self.population[0])  # for saving policy purposes
        self.episode_length = config.episode_length
        self.reward_shaping = False

    def initialize_fitness(self):
        '''
        Adds and attribute "fitness" to the genes/actors in the list of population,
        and sets the fitness of all genes/actor in the population to 0
        '''
        for gene in self.population:
            gene.fitness = 0.0
        print("Initialized gene fitness to zeros")

    def evaluate_pop(self, env):
        for gene in self.population:
            evo_obs = torch.Tensor([env.reset()])
            evo_episode_reward = 0
            for t in range(self.episode_length):
                evo_action = gene.step(evo_obs)

                agent_actions = [[ac.data.numpy() for ac in evo_action]]
                evo_next_obs, evo_rewards, evo_done, infos = env.step(agent_actions)
                g_rewards = []
                d_rewards = []
                for n in range(len(env.envs[0].agents)):
                    d_rewards.append([evo_rewards[0][n][1]])
                    g_rewards.append([evo_rewards[0][n][0]])
                d_rewards = [d_rewards]
                g_rewards = [g_rewards]
                d_rewards = np.array(d_rewards)
                g_rewards = np.array(g_rewards)
                if self.reward_shaping:
                    reward = d_rewards
                else:
                    reward = g_rewards
                reward = reward.sum()
                evo_episode_reward += reward

                evo_action = torch.Tensor(evo_action); #evo_mask = torch.Tensor([not evo_done])
                evo_next_obs = torch.Tensor([evo_next_obs])
                # evo_reward = torch.Tensor([evo_reward])
                evo_obs = copy.copy(evo_next_obs)

                # <end of time-steps>
            fitness = evo_episode_reward
            # <end of episodes>
            gene.fitness = copy.copy(fitness)

    def rank_pop_selection_mutation(self, env):
        '''
        This function takes the current evaluated population (of k , then ranks them according to their fitness,
        then selects a number of elites (e), and then selects a set S of (k-e) using tournament selection.
        It then calls the mutation function to add mutation to the set S of genes.
        In the end this will replace the current population with a new one.
        '''
        ranked_pop = copy.deepcopy(sorted(self.population, key=lambda x: x.fitness, reverse=True))  # Algo1: 9
        elites = ranked_pop[:self.num_elites]
        self.best_policy = elites[0]  # for saving policy purposes
        set_s = []

        for i in range(len(ranked_pop) - len(elites)):
            tournament_genes = [random.choice(ranked_pop) for _ in range(self.tournament_genes)]
            tournament_winner = max(tournament_genes, key=attrgetter('fitness'))
            set_s.append(copy.deepcopy(tournament_winner))

        # mutated_set_S = copy.deepcopy(self.mutation(set_s))
        self.mutate_genes(set_s)
        mutated_set_S = copy.deepcopy(set_s)
        self.population = []
        # Addition of lists
        self.population = copy.deepcopy(elites + mutated_set_S)
        print("Best fitness = " + str(elites[0].fitness))

        self.save_fitness.append(elites[0].fitness)

    def mutation(self, set_s):
        """
        :param set_s: This is the set of (k-e) genes that are going to be mutated by adding noise
        :return: Returns the mutated set of (k-e) genes

        Adds noise to the weights and biases of each layer of the network
        But why is a noise (out of 1) being added? Since we cant really say how big or small the parameters should be.
        """
        for i in range(len(set_s)):
            ''' Noise to Linear 1 weights and biases'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.linear1.weight))
            noise = torch.FloatTensor(noise)
            # gene.actor.linear1.weight.data = gene.actor.linear1.weight.data + noise
            noise = torch.mul(set_s[i].actor.linear1.weight.data, noise)
            set_s[i].actor.linear1.weight.data = copy.deepcopy(set_s[i].actor.linear1.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.linear1.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.linear1.bias.data, noise)
            set_s[i].actor.linear1.bias.data = copy.deepcopy(set_s[i].actor.linear1.bias.data + noise)

            '''Noise to Linear 2 weights and biases'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.linear2.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.linear2.weight.data, noise)
            set_s[i].actor.linear2.weight.data = copy.deepcopy(set_s[i].actor.linear2.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.linear2.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.linear2.bias.data, noise)
            set_s[i].actor.linear2.bias.data = copy.deepcopy(set_s[i].actor.linear2.bias.data + noise)

            ''' Noise to mu layer weights and biases'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.mu.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.mu.weight.data, noise)
            set_s[i].actor.mu.weight.data = copy.deepcopy(set_s[i].actor.mu.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.mu.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.mu.bias.data, noise)
            set_s[i].actor.mu.bias.data = copy.deepcopy(set_s[i].actor.mu.bias.data + noise)

            ''' LayerNorm 1'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerN1.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerN1.weight.data, noise)
            set_s[i].actor.layerN1.weight.data = copy.deepcopy(set_s[i].actor.layerN1.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerN1.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerN1.bias.data, noise)
            set_s[i].actor.layerN1.bias.data = copy.deepcopy(set_s[i].actor.layerN1.bias.data + noise)

            ''' LayerNorm 2'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerN2.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerN2.weight.data, noise)
            set_s[i].actor.layerN2.weight.data = copy.deepcopy(set_s[i].actor.layerN2.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerN2.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerN2.bias.data, noise)
            set_s[i].actor.layerN2.bias.data = copy.deepcopy(set_s[i].actor.layerN2.bias.data + noise)

            ''' LayerNorm MU'''
            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerNmu.weight))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerNmu.weight.data, noise)
            set_s[i].actor.layerNmu.weight.data = copy.deepcopy(set_s[i].actor.layerNmu.weight.data + noise)

            noise = np.random.normal(loc=self.noise_mean, scale=self.noise_stddev,
                                     size=np.shape(set_s[i].actor.layerNmu.bias))
            noise = torch.FloatTensor(noise)
            noise = torch.mul(set_s[i].actor.layerNmu.bias.data, noise)
            set_s[i].actor.layerNmu.bias.data = copy.deepcopy(set_s[i].actor.layerNmu.bias.data + noise)

        return set_s

    def mutate_genes(self, set_s):
        for gene in set_s:
            self.mutate_inplace(gene)

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def mutate_inplace(self, gene):
        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        num_params = len(list(gene.policy.parameters()))
        #ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
        model_params = gene.policy.state_dict()

        for i, key in enumerate(model_params): #Mutate each param

            if key == 'lnorm1.gamma' or key == 'lnorm1.beta' or  key == 'lnorm2.gamma' or key == 'lnorm2.beta' or key == 'lnorm3.gamma' or key == 'lnorm3.beta': continue

            # References to the variable keys
            W = model_params[key]
            if len(W.shape) == 2: #Weights, no bias

                num_weights= W.shape[0]*W.shape[1]
                ssne_prob = 1#ssne_probabilities[i]

                if random.random() < ssne_prob:
                    num_mutations = fastrand.pcg32bounded(int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
                    for _ in range(num_mutations):
                        ind_dim1 = fastrand.pcg32bounded(W.shape[0])
                        ind_dim2 = fastrand.pcg32bounded(W.shape[-1])
                        random_num = random.random()

                        if random_num < super_mut_prob:  # Super Mutation probability
                            W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
                        elif random_num < reset_prob:  # Reset probability
                            W[ind_dim1, ind_dim2] = random.gauss(0, 1)
                        else:  # mutauion even normal
                            W[ind_dim1, ind_dim2] += random.gauss(0, mut_strength *W[ind_dim1, ind_dim2])

                        # Regularization hard limit
                        W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2], 1000000)

    @classmethod
    def init_from_env(cls, env, config):
        """Instantiate instance of this class from multiagent-environment"""
        agent_init_params = []
        for acsp, obsp in zip(env.action_space, env.observation_space):
            num_in_pol = obsp.shape[0]
            discrete_action = False                               # I added this
            num_out_pol = 0
            for ac in acsp.spaces:                      # Replacement for the "get_shape" lambda function
                num_out_pol += ac.shape[0]

            num_in_critic = 0
            for oobsp in env.observation_space:
                num_in_critic += oobsp.shape[0]
            for oacsp in env.action_space:          # Replacement
                for k in oacsp.spaces:
                    num_in_critic += k.shape[0]

            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'config': config,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'comm_acs_space': env.action_space[0].spaces[1].shape[0]}      # for actor policy comm.
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename, map_location='cpu')
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance

