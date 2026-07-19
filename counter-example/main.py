import argparse
import numpy as np
import pandas as pd
from environment import FixedHorizonMDPEnv


def generate_mdp(S, A, H):

    ss = 0

    p = np.zeros((H, S, A, S))
    r = np.zeros((H, S, A))

    for t in range(0, H):
        # State 0 action 0
        p[t, 0, 0, 1] = 0.5
        p[t, 0, 0, 2] = 0.5

        """State indexed 3 acts as terminal state"""

        # State 1 action 0
        p[t, 1, 0, 3] = 1.0
        # State 1 action 1
        p[t, 1, 1, 3] = 1.0

        # State 2 action 0
        p[t, 2, 0, 3] = 1.0
        # State 2 action 1
        p[t, 2, 1, 3] = 1.0

        # Rewards
        r[t, 1, 0] = 2.0
        r[t, 1, 1] = 4.0
        r[t, 2, 0] = 3.0
        r[t, 2, 1] = 2.0

    return p, r, ss


def generate_policies(H):

    pi0 = []
    pi1 = []

    for t in range(H):
        pi0.append([0, 0, 0, 1])  # Last 1 is placeholder for terminal state
        pi1.append([0, 1, 1, 1])  # Last 1 is placeholder for terminal state

    policies = [np.array(pi0), np.array(pi1)]

    return policies


def dp_evaluation(policy, P, R):

    H, S = policy.shape

    # Value function
    V = np.zeros((H + 1, S))

    for t in reversed(range(H)):
        for s in range(S):
            a = policy[t, s]
            V[t, s] = R[t, s, a] + np.sum(P[t, s, a] * V[t + 1])

    return V[0]  # Returning values corresponding to zeroth time step


# Mixed radix encoding
def encode6(x1, x2, x3, x4, x5, x6, ranges):
    A, B, C, D, E, F = ranges
    return ((((x1 * B + x2) * C + x3) * D + x4) * E + x5) * F + x6


def encode5(x1, x2, x3, x4, x5, ranges):
    A, B, C, D, E = ranges
    return (((x1 * B + x2) * C + x3) * D + x4) * E + x5


def mc_evaluation(env, pi_det, ss, n_episodes, r_ind, p_ind, mode,
                  n_policies, n_runs, l_horizon, n_states, n_actions):

    ranges6 = (n_policies, n_runs, n_episodes, l_horizon, n_states, n_actions)
    ranges5 = (n_runs, n_episodes, l_horizon, n_states, n_actions)

    returns = []  # Stores returns of episodes

    for i in range(n_episodes):

        env.t = 0  # Set time zero
        env.state = ss  # Override state to fixed start state

        done = False
        t = 0
        ret = 0.0

        while not done:
            s = env.state
            a = pi_det[t, s]
            if mode == "random":
                my_seed = encode6(p_ind, r_ind, i, t, s, a, ranges6)
            elif mode == "controlled":
                my_seed = encode5(r_ind, i, t, s, a, ranges5)
            _, r, done, _ = env.step(action=a, my_seed=my_seed)
            ret += r
            t += 1
        returns.append(ret)

    v_hat = np.mean(returns)
    return v_hat


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int, default=1)
    parser.add_argument("rollouts", type=int, default=1)
    parser.add_argument("horizon", type=int, default=2)

    args = parser.parse_args()
    num_runs = args.runs
    num_rollouts = args.rollouts
    horizon = args.horizon

    print("Number of runs:", num_runs)
    print("Number of rollouts:", num_rollouts)
    print("Horizon:", horizon)

    numStates, numActions = 4, 2
    P, R, start_state = generate_mdp(numStates, numActions, horizon)

    env = FixedHorizonMDPEnv(P, R)

    numPolicies = 2
    list_of_policies = generate_policies(horizon)

    """
    DP evaluation
    """

    dp_values_of_policies = []  # For fixed start state
    for policy in list_of_policies:
        val = dp_evaluation(policy, P, R)[start_state]
        dp_values_of_policies.append(val.item())

    best_policy_val = np.max(dp_values_of_policies)
    best_policy_indices = [i for i, v in enumerate(dp_values_of_policies) if v == best_policy_val]

    """
    MC evaluation
    """

    run_list = []
    dp_best_val = []
    dp_best_indices = []

    rand_best_val = []  # Best policy value identified across runs (empirical)
    cont_best_val = []  # Best policy value identified across runs (empirical)

    rand_best_val_true = []  # Best policy value identified across runs
    cont_best_val_true = []  # Best policy value identified across runs

    rand_best_indices = []  # Best policy index identified across runs
    cont_best_indices = []  # Best policy index identified across runs

    rand_best_identified = []  # Whether the best policy identified across runs
    cont_best_identified = []  # Whether the best policy identified across runs

    for run_ind in range(num_runs):

        print("Run number:", run_ind + 1)

        run_list.append(run_ind+1)
        dp_best_val.append(np.round(best_policy_val, 6))
        dp_best_indices.append(best_policy_indices)

        rand_values = []  # Values of random policies
        cont_values = []  # Values of controlled policies

        for pol_ind, policy in enumerate(list_of_policies):

            """Random"""
            rand_val = mc_evaluation(env=env, pi_det=policy, ss=start_state, n_episodes=num_rollouts,
                                     r_ind=run_ind, p_ind=pol_ind, mode="random",
                                     n_policies=numPolicies, n_runs=num_runs, l_horizon=horizon,
                                     n_states=numStates, n_actions=numActions)
            rand_values.append(rand_val.item())

            """Controlled"""
            cont_val = mc_evaluation(env=env, pi_det=policy, ss=start_state, n_episodes=num_rollouts,
                                     r_ind=run_ind, p_ind=pol_ind, mode="controlled",
                                     n_policies=numPolicies, n_runs=num_runs, l_horizon=horizon,
                                     n_states=numStates, n_actions=numActions)

            cont_values.append(cont_val.item())

        r_best_val = np.max(rand_values)
        c_best_val = np.max(cont_values)

        r_best_indices = [i for i, v in enumerate(rand_values) if v == r_best_val]
        c_best_indices = [i for i, v in enumerate(cont_values) if v == c_best_val]

        r_best_val_true = dp_values_of_policies[r_best_indices[0]]
        c_best_val_true = dp_values_of_policies[c_best_indices[0]]

        rand_best_val.append(np.round(r_best_val, 6).item())
        cont_best_val.append(np.round(c_best_val, 6).item())

        rand_best_val_true.append(np.round(r_best_val_true, 6))
        cont_best_val_true.append(np.round(c_best_val_true, 6))

        rand_best_indices.append(r_best_indices)
        cont_best_indices.append(c_best_indices)

        if r_best_val_true == best_policy_val:
            rand_best_identified.append(1)
        else:
            rand_best_identified.append(0)

        if c_best_val_true == best_policy_val:
            cont_best_identified.append(1)
        else:
            cont_best_identified.append(0)

    run_wise_df = pd.DataFrame({
        'Run': run_list,
        'Rand_best_identified': rand_best_identified,
        'Cont_best_identified': cont_best_identified,
        'DP_best_val': dp_best_val,
        'Rand_best_val_true': rand_best_val_true,
        'Cont_best_val_true': cont_best_val_true,
        'Rand_best_val_emp': rand_best_val,
        'Cont_best_val_emp': cont_best_val,
        'DP_best_indices': dp_best_indices,
        'Rand_best_indices': rand_best_indices,
        'Cont_best_indices': cont_best_indices,
    })

    run_wise_df.to_csv(f"res/r{num_rollouts}.csv")