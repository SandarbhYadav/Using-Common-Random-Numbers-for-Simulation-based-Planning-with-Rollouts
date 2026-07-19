import numpy as np


class FixedHorizonMDPEnv:

    def __init__(self, P, R, mu0):
        self.P = P
        self.R = R
        self.mu0 = mu0

        self.H, self.S, self.A, _ = P.shape

        self.t = None
        self.state = None
        self.seed = None
        self.rng = np.random.default_rng()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.state = self.rng.choice(self.S, p=self.mu0)

        return self.state

    def step(self, action, my_seed):
        assert self.t < self.H, "Episode has terminated."

        self.seed = my_seed
        self.rng = np.random.default_rng(my_seed)

        reward = self.R[self.t, self.state, action]
        next_state = self.rng.choice(
            self.S, p=self.P[self.t, self.state, action]
        )

        self.t += 1
        done = self.t == self.H

        self.state = next_state

        info = {"t": self.t}

        return next_state, reward, done, info
