import numpy as np


class FixedHorizonMDPEnv:
    def __init__(self, P, R):
        self.P = P
        self.R = R

        self.H, self.S, self.A, _ = P.shape

        self.t = None
        self.state = None
        self.seed = None
        self.rng = np.random.default_rng()

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
