import copy
import argparse
import numpy as np
import pandas as pd
from uct import UCTPlanner
from environment import FixedHorizonMDPEnv
from multiprocessing import Pool


def generate_mdp(S, A, H, seed=None):
    rng = np.random.default_rng(seed)

    p = rng.random((H, S, A, S))
    p /= p.sum(axis=-1, keepdims=True)

    r = rng.uniform(-1, 1, size=(H, S, A))

    mu0 = rng.random(S)
    mu0 /= mu0.sum()

    return p, r, mu0


def optimal_value(S, A, P, R, H, s0):

    V = np.zeros((H + 1, S))  # Optimal value function
    pi = np.zeros((H, S), dtype=int)  # Optimal policy

    for t in reversed(range(H)):
        for s in range(S):
            q_sa = np.zeros(A)
            for a in range(A):
                q_sa[a] = R[t, s, a] + np.sum(P[t, s, a] * V[t + 1])

            V[t, s] = np.max(q_sa)
            pi[t, s] = np.argmax(q_sa)

    return V[0, s0], V, pi


def one_run(env, num_runs, num_rollouts, numStates, numActions, horizon, search_depth, c_ucb, mode, run_ind, s0):

    env.reset(seed=run_ind)  # Independent runs
    env.state = s0  # Fixed start state
    state = s0
    done = False
    ret = 0.0
    time_step = 0

    while not done:

        # New planner for each decision time step
        planner = UCTPlanner(
            env=copy.deepcopy(env),
            num_runs=num_runs,
            num_simulations=num_rollouts,
            num_states=numStates,
            num_actions=numActions,
            horizon=horizon,
            max_depth=search_depth,
            c_ucb=c_ucb,
            mode=mode,
            run_ind=run_ind
        )

        action = planner.plan(state, time_step)
        state, reward, done, _ = env.step(action)
        ret += reward
        time_step += 1

    return np.round(ret, 6).item()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int, default=1)
    parser.add_argument("rollouts", type=int, default=1)
    parser.add_argument("depth", type=int, default=1)
    parser.add_argument("horizon", type=int, default=10)
    parser.add_argument("processes", type=int, default=1)

    args = parser.parse_args()
    num_runs = args.runs
    num_rollouts = args.rollouts
    search_depth = args.depth
    horizon = args.horizon
    num_processes = args.processes

    print("Number of runs:", num_runs)
    print("Number of rollouts:", num_rollouts)
    print("Search depth:", search_depth)
    print("Horizon:", horizon)
    print("Number of processes:", num_processes)

    numStates, numActions = 7, 4
    P, R, mu0 = generate_mdp(numStates, numActions, horizon, seed=0)  # Synthetic MDP

    env = FixedHorizonMDPEnv(P, R, mu0)

    start_state = env.reset(seed=0)  # Fixed start state

    """
    Optimal Value Calculation
    """

    opt_val, opt_val_fun, opt_pol = optimal_value(S=numStates, A=numActions, P=P, R=R, H=horizon, s0=start_state)

    """
    UCT
    """

    ucb_exploration_constant = 2.0

    # Random (Independent)

    with Pool(num_processes) as pool:
        rand_vals = pool.starmap(
            one_run,
            [(env, num_runs, num_rollouts, numStates, numActions, horizon, search_depth, ucb_exploration_constant,
              "random", r_ind, start_state) for r_ind in range(num_runs)]
        )

    # Controlled (Dependent)

    with Pool(num_processes) as pool:
        cont_vals = pool.starmap(
            one_run,
            [(env, num_runs, num_rollouts, numStates, numActions, horizon, search_depth, ucb_exploration_constant,
              "controlled", r_ind, start_state) for r_ind in range(num_runs)]
        )

    # Mixed (Depth-dependent)

    with Pool(num_processes) as pool:
        mix_vals = pool.starmap(
            one_run,
            [(env, num_runs, num_rollouts, numStates, numActions, horizon, search_depth, ucb_exploration_constant,
              "mixed", r_ind, start_state) for r_ind in range(num_runs)]
        )

    run_wise_df = pd.DataFrame({
        'Run': [i for i in range(num_runs)],
        'DP_val': [np.round(opt_val, 6) for j in range(num_runs)],
        'Rand_val': rand_vals,
        'Cont_val': cont_vals,
        'Mix_val': mix_vals
    })

    run_wise_df.to_csv(f"res/h{horizon}_d{search_depth}_e{num_rollouts}.csv")
