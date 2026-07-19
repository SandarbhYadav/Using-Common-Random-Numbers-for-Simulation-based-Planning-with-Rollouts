import math
import random
from collections import defaultdict


# Mixed radix encoding
def encode6(x1, x2, x3, x4, x5, x6, ranges):
    A, B, C, D, E, F = ranges
    return ((((x1 * B + x2) * C + x3) * D + x4) * E + x5) * F + x6


def encode5(x1, x2, x3, x4, x5, ranges):
    A, B, C, D, E = ranges
    return (((x1 * B + x2) * C + x3) * D + x4) * E + x5


def encode3(x1, x2, x3, ranges):
    A, B, C = ranges
    return (x1 * B + x2) * C + x3


def encode2(x1, x2, ranges):
    A, B = ranges
    return x1 * B + x2


class UCTPlanner:
    def __init__(
        self,
        env,
        num_runs,
        num_simulations,
        num_states,
        num_actions,
        horizon,
        max_depth,
        c_ucb,
        mode=None,
        run_ind=0
    ):
        self.env = env
        self.num_runs = num_runs
        self.num_simulations = num_simulations
        self.S = num_states
        self.A = num_actions
        self.H = horizon
        self.D = max_depth
        self.c = c_ucb
        self.mode = mode
        self.run_ind = run_ind
        self.count_root_actions = [0 for a in range(num_actions)]  # Number of times each action is taken at root node

        self.N = defaultdict(int)          # Stores visits of (s, t) with default values 0
        self.N_sa = defaultdict(int)       # Stores visits of (s, a, t) with default values 0
        self.Q = defaultdict(float)        # Stores Q-values of (s, a, t) with default values 0.0

    def plan(self, root_state, root_time):
        for _ in range(self.num_simulations):
            self.env.state = root_state  # Reset root state for each simulation
            self.env.t = root_time  # Reset root time for each simulation
            self.simulate(root_state, root_time)  # UCT simulation

        best_act = max(range(self.A), key=lambda a: self.Q[(root_state, a, root_time)])

        return best_act

    def simulate(self, root_state, root_time):

        trajectory = []   # List of (s, a, t) tuples
        rewards = []      # List of rewards obtained along the trajectory

        state = root_state
        t = root_time
        depth = 0

        ranges6 = (self.A, self.A, self.S, self.H, self.num_simulations, self.num_runs)
        ranges5 = (self.A, self.S, self.H, self.num_simulations, self.num_runs)
        root_action = None

        # Traverse tree using UCB policy
        while t < self.H and depth < self.D:

            self.N[(state, t)] += 1

            action = self.select_ucb(state, t)

            if depth < 1:  # So that root action and it's count does not get modified each step
                root_action = action
                self.count_root_actions[root_action] += 1

            if (self.mode == "random") or (self.mode == "mixed"):
                my_seed1 = encode6(root_action, action, state, t, self.count_root_actions[root_action], self.run_ind, ranges6)
            elif self.mode == "controlled":
                my_seed1 = encode5(action, state, t, self.count_root_actions[root_action], self.run_ind, ranges5)
            next_state, reward, done, _ = self.env.step(action=action, my_seed=my_seed1)

            trajectory.append((state, action, t))
            rewards.append(reward)

            state = next_state
            t += 1
            depth += 1

        rollout_return = 0.0
        cur_state = state
        cur_t = t

        ranges3 = (self.A, self.num_simulations, self.num_runs)
        ranges2 = (self.num_simulations, self.num_runs)
        if self.mode == "random":
            roll_seed = encode3(root_action, self.count_root_actions[root_action], self.run_ind, ranges3)
        elif (self.mode == "controlled") or (self.mode == "mixed"):
            roll_seed = encode2(self.count_root_actions[root_action], self.run_ind, ranges2)

        roll_rng = random.Random(roll_seed)

        # Perform rollout
        while cur_t < self.H:
            action = roll_rng.randrange(self.A)  # Uniform random rollout policy
            if self.mode == "random":
                my_seed2 = encode6(root_action, action, cur_state, cur_t, self.count_root_actions[root_action], self.run_ind, ranges6)
            elif (self.mode == "controlled") or (self.mode == "mixed"):
                my_seed2 = encode5(action, cur_state, cur_t, self.count_root_actions[root_action], self.run_ind, ranges5)
            cur_state, reward, done, _ = self.env.step(action=action, my_seed=my_seed2)
            rollout_return += reward
            cur_t += 1

        # Backpropagate values
        G = rollout_return

        for (state, action, t), reward in reversed(
            list(zip(trajectory, rewards))
        ):
            G += reward

            self.N_sa[(state, action, t)] += 1
            alpha = 1.0 / self.N_sa[(state, action, t)]
            self.Q[(state, action, t)] += alpha * (G - self.Q[(state, action, t)])

    def select_ucb(self, state, t):

        total_visits = self.N[(state, t)]

        # Unvisited actions first
        for a in range(self.A):
            if self.N_sa[(state, a, t)] == 0:  # Visit every action once
                return a

        best_value = -float("inf")
        best_action = None

        for a in range(self.A):
            q = self.Q[(state, a, t)]
            n_sa = self.N_sa[(state, a, t)]

            ucb = q + self.c * math.sqrt(math.log(total_visits) / n_sa)
            if ucb > best_value:
                best_value = ucb
                best_action = a

        return best_action
