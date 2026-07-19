import argparse
import numpy as np
import pandas as pd
from copy import deepcopy
from environment import Env
from multiprocessing import Pool

actions = [act/100 for act in range(0, 101)]

env = Env()
roll_env = Env()


# Mixed radix encoding
def encode2(x1, x2, ranges):
    A, B = ranges
    return x1 * B + x2


def rollout(roll_i, mode):
    if mode == "mixed":
        my_seed1 = roll_i  # Each action get same sequence of random numbers in rollout phase
        roll_env.seed(seed=my_seed1)
    total_reward = 0.0
    comp = False
    while not comp:
        _, r, t1, t2, _ = roll_env.step(action=0.11)  # Rollout policy
        total_reward += r
        comp = t1 or t2
    return total_reward


def plan(sta, rollouts, mode):

    returns_of_actions = []  # Stores avg return (across rollouts) for each action

    ranges2 = (len(actions), rollouts)

    for act_ind in range(len(actions)):
        total_return = 0.0  # Stores the sum of returns across rollouts
        action = actions[act_ind]
        for roll in range(rollouts):
            if (mode == "random") or (mode == "mixed"):
                my_seed = encode2(act_ind, roll, ranges2)  # Each action gets different sequence of random numbers
            elif mode == "controlled":
                my_seed = roll  # Each action gets same sequence of random numbers
            options = {"custom": True, "custom_state": sta}
            roll_env.reset(seed=my_seed, options=options)
            _, re, te, tr, _ = roll_env.step(action=action)
            complete = te or tr
            total_return += (re + (rollout(roll_i=roll, mode=mode) if (not complete) else 0.0))
        avg_return = total_return/rollouts
        returns_of_actions.append(avg_return)

    return actions[np.argmax(returns_of_actions)]


def one_run(num_rollouts, mode, ss, run_ind):
    options = {"custom": True, "custom_state": deepcopy(ss)}
    env.reset(seed=run_ind, options=options)  # Independent runs from same starting state
    state = deepcopy(ss)  # Same initial state across runs
    done = False
    reward_list = []

    # Decision time planning
    while not done:
        act = plan(sta=deepcopy(state), rollouts=num_rollouts, mode=mode)
        state, reward, term, trunc, info = env.step(action=act)
        reward_list.append(reward)
        done = term or trunc

    return np.round(np.sum(reward_list), 6)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int, default=1)
    parser.add_argument("rollouts", type=int, default=1)
    parser.add_argument("processes", type=int, default=1)

    args = parser.parse_args()
    num_runs = args.runs
    num_rollouts = args.rollouts
    num_processes = args.processes

    print("Number of runs:", num_runs)
    print("Number of rollouts:", num_rollouts)
    print("Number of processes:", num_processes)

    start_state, _ = env.reset(seed=0)  # Same initial state across runs

    # Random (Independent)

    with Pool(num_processes) as pool:
        rand_return_list = pool.starmap(
            one_run,
            [(num_rollouts, "random", deepcopy(start_state), r_ind) for r_ind in range(num_runs)]
        )

    # Controlled (Dependent)

    with Pool(num_processes) as pool:
        cont_return_list = pool.starmap(
            one_run,
            [(num_rollouts, "controlled", deepcopy(start_state), r_ind) for r_ind in range(num_runs)]
        )

    # Mixed (Depth-dependent)

    with Pool(num_processes) as pool:
        mix_return_list = pool.starmap(
            one_run,
            [(num_rollouts, "mixed", deepcopy(start_state), r_ind) for r_ind in range(num_runs)]
        )

    run_wise_df = pd.DataFrame({
        'Run': [i+1 for i in range(num_runs)],
        'Random': rand_return_list,
        'Controlled': cont_return_list,
        'Mixed': mix_return_list
    })

    run_wise_df.to_csv(f"res/rollouts_{num_rollouts}.csv")
