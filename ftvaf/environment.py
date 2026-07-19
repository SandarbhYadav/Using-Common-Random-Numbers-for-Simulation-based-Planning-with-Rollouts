import gymnasium as gym
import numpy as np
from copy import deepcopy


# Financial Parameters
investment_mu = 0.15
investment_sigma = 0.2
total_years = 20


class Env():

    def __init__(self):

        # Action space: 1. Payout
        self.action_space = gym.spaces.Box(0, 1, dtype=np.double)
        # Observation space: 1.Corpus 2.Years 3.Fraction_lives
        self.observation_space = gym.spaces.Box(np.array([0, 0, 0] + [0] * 41), np.array([36, 35, 1] + [1] * 41), shape=(44,), dtype=np.double)

        self.mu = investment_mu
        self.sigma = investment_sigma
        self.total_years = total_years
        self.create_lookups()

    def seed(self, seed=None):
        if seed is not None:
            self.curr_seed = seed
            self.rng = np.random.default_rng(seed)

    def create_lookups(self):
        self.death_lambda_n = 1000  # Probably 1000 people
        death_prob = np.genfromtxt("prob_death.csv", delimiter=",", names=True)
        self.death_lambda_lookup = death_prob["prob_death"][10:] * self.death_lambda_n  # Age 60 onwards

    def reset(self, seed=None, options={"custom": False}):
        if seed is not None:
            self.seed(seed=seed)
        if options["custom"]:
            self.state = deepcopy(options["custom_state"])
        else:
            self.state = self.observation_space.sample()
            self.state[0] = 1.0
            self.state[1] = 0
            self.state[2] = 1.0
            lives = self.rng.integers(60, 70, self.death_lambda_n)  # Generate between 60-70 so will go till 90 max for 20 years
            self.state[3:] = (np.histogram(lives, bins=range(60, 102))[0]) / self.death_lambda_n
        return self.state, {}

    def GBM_return(self, wealth):
        # Generate a random number from the standard normal distribution
        Z = self.rng.standard_normal()
        # Calculate the next wealth level
        next_wealth = wealth * np.exp((self.mu - 0.5 * (self.sigma ** 2)) + self.sigma * Z)
        return next_wealth

    def step_lives(self, ):
        random_death = np.array(
            [self.rng.poisson(lamda, 1) for lamda in self.death_lambda_lookup]) / self.death_lambda_n
        random_death = random_death.flatten()
        self.state[3:] *= (1 - random_death)
        self.state[4:] = self.state[3:-1]  # One year has passed so ages updated
        self.state[3] = 0
        self.state[2] = np.sum(self.state[3:])

    def step(self, action):
        corpus = deepcopy(self.state[0])
        num_lives_old = deepcopy(self.state[2])
        # Update the years elapsed
        self.state[1] += 1
        # Update lives
        self.step_lives()
        num_lives_new = deepcopy(self.state[2])

        # Step rewards
        reward = corpus * action

        # Update the corpus
        corpus = corpus * (1 - action)
        if (corpus > 0):
            corpus = corpus * (num_lives_new / num_lives_old)
            corpus = self.GBM_return(corpus)
        self.state[0] = corpus

        # Termination and Truncation
        terminated = True if self.state[1] >= self.total_years else False  # Horizon termination
        truncated = (True if self.state[0] <= (0.05 * self.state[2]) else False) and (not terminated)

        # Terminal rewards
        if truncated:
            bankruptcy_penalty_coefficient = 5
            reward += (self.state[0] + num_lives_new * (self.state[1] - (self.total_years + 1)) * bankruptcy_penalty_coefficient)
        elif terminated:
            soft_penalty_coef = 0.03
            reward += -1 * self.state[0] * soft_penalty_coef

        info = {}

        return (
            self.state,
            reward,
            terminated,
            truncated,
            info
        )
