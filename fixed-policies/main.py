import argparse
import numpy as np
import pandas as pd
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


def generate_policies(num_policies, H, S, A, D, ss, seed=None):

    rng = np.random.default_rng(seed)

    shared_tail = rng.integers(A, size=(H - D, S))

    policies = []

    for _ in range(num_policies):
        prefix = rng.integers(A, size=(D, S))
        if D == 1:
            prefix[0][ss] = _
        policy = np.zeros((H, S), dtype=int)
        policy[:D] = prefix
        policy[D:] = shared_tail
        policies.append(policy)

    return policies


def dp_evaluation(policy, P, R):

    H, S = policy.shape

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


def mc_evaluation(env, pi_det, ss, n_episodes, depth, r_ind, p_ind, mode,
                  n_policies, n_runs, l_horizon, n_states, n_actions):

    ranges6 = (n_policies, n_runs, n_episodes, l_horizon, n_states, n_actions)
    ranges5 = (n_runs, n_episodes, l_horizon, n_states, n_actions)

    returns = []  # Stores returns of episodes

    for i in range(n_episodes):

        env.t = 0  # Set time zero
        env.state = ss  # Set fixed start state

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
            elif mode == "mixed":
                if t < depth:
                    my_seed = encode6(p_ind, r_ind, i, t, s, a, ranges6)
                else:
                    my_seed = encode5(r_ind, i, t, s, a, ranges5)
            _, r, done, _ = env.step(action=a, my_seed=my_seed)
            ret += r
            t += 1
        returns.append(ret)

    v_hat = np.mean(returns)
    return v_hat


def one_run(list_of_policies, numPolicies, env, start_state, num_runs, num_rollouts, horizon, search_depth, numStates, numActions, mode, run_ind):

    values = []  # Values of policies

    for pol_ind, policy in enumerate(list_of_policies):

        val = mc_evaluation(env=env, pi_det=policy, ss=start_state, n_episodes=num_rollouts,
                                 depth=search_depth, r_ind=run_ind, p_ind=pol_ind, mode=mode,
                                 n_policies=numPolicies, n_runs=num_runs, l_horizon=horizon,
                                 n_states=numStates, n_actions=numActions)
        values.append(val.item())

    return values


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

    numPolicies = 100
    if search_depth == 1:
        numPolicies = numActions  # As only 4 policies are possible at depth 1
    list_of_policies = generate_policies(numPolicies, horizon, numStates, numActions, search_depth, start_state, seed=747)

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

    # Random (Independent)

    with Pool(num_processes) as pool:
        rand_values = pool.starmap(
            one_run,
            [(list_of_policies, numPolicies, env, start_state, num_runs, num_rollouts, horizon, search_depth, numStates,
              numActions, "random", r_ind) for r_ind in range(num_runs)]
        )

    # Controlled (Dependent)

    with Pool(num_processes) as pool:
        cont_values = pool.starmap(
            one_run,
            [(list_of_policies, numPolicies, env, start_state, num_runs, num_rollouts, horizon, search_depth, numStates,
              numActions, "controlled", r_ind) for r_ind in range(num_runs)]
        )

    # Mixed (Depth-dependent)

    with Pool(num_processes) as pool:
        mix_values = pool.starmap(
            one_run,
            [(list_of_policies, numPolicies, env, start_state, num_runs, num_rollouts, horizon, search_depth, numStates,
              numActions, "mixed", r_ind) for r_ind in range(num_runs)]
        )

    rand_best_val = []  # Best policy value identified across runs (empirical)
    cont_best_val = []  # Best policy value identified across runs (empirical)
    mix_best_val = []  # Best policy value identified across runs (empirical)

    rand_best_val_true = []  # True value of the identified best policy across runs
    cont_best_val_true = []  # True value of the identified best policy across runs
    mix_best_val_true = []  # True value of the identified best policy across runs

    rand_best_indices = []  # Best policy indices identified across runs
    cont_best_indices = []  # Best policy indices identified across runs
    mix_best_indices = []  # Best policy indices identified across runs

    rand_best_identified = []  # Whether one of the best policies identified across runs
    cont_best_identified = []  # Whether one of the best policies identified across runs
    mix_best_identified = []  # Whether one of the best policies identified across runs

    for run_ind in range(num_runs):

        r_best_val = np.max(rand_values[run_ind])
        c_best_val = np.max(cont_values[run_ind])
        m_best_val = np.max(mix_values[run_ind])

        rand_best_val.append(np.round(r_best_val, 6).item())
        cont_best_val.append(np.round(c_best_val, 6).item())
        mix_best_val.append(np.round(m_best_val, 6).item())

        r_best_indices = [i for i, v in enumerate(rand_values[run_ind]) if v == r_best_val]
        c_best_indices = [i for i, v in enumerate(cont_values[run_ind]) if v == c_best_val]
        m_best_indices = [i for i, v in enumerate(mix_values[run_ind]) if v == m_best_val]

        rand_best_indices.append(r_best_indices)
        cont_best_indices.append(c_best_indices)
        mix_best_indices.append(m_best_indices)

        r_best_val_true = dp_values_of_policies[r_best_indices[0]]
        c_best_val_true = dp_values_of_policies[c_best_indices[0]]
        m_best_val_true = dp_values_of_policies[m_best_indices[0]]

        rand_best_val_true.append(np.round(r_best_val_true, 6))
        cont_best_val_true.append(np.round(c_best_val_true, 6))
        mix_best_val_true.append(np.round(m_best_val_true, 6))

        if r_best_val_true == best_policy_val:
            rand_best_identified.append(1)
        else:
            rand_best_identified.append(0)

        if c_best_val_true == best_policy_val:
            cont_best_identified.append(1)
        else:
            cont_best_identified.append(0)

        if m_best_val_true == best_policy_val:
            mix_best_identified.append(1)
        else:
            mix_best_identified.append(0)

    run_wise_df = pd.DataFrame({
        'Run': [i+1 for i in range(num_runs)],
        'Rand_best_identified': rand_best_identified,
        'Cont_best_identified': cont_best_identified,
        'Mix_best_identified': mix_best_identified,
        'DP_best_val': [np.round(best_policy_val, 6) for j in range(num_runs)],
        'Rand_best_val_true': rand_best_val_true,
        'Cont_best_val_true': cont_best_val_true,
        'Mix_best_val_true': mix_best_val_true,
        'Rand_best_val_emp': rand_best_val,
        'Cont_best_val_emp': cont_best_val,
        'Mix_best_val_emp': mix_best_val,
        'DP_best_indices': [best_policy_indices for k in range(num_runs)],
        'Rand_best_indices': rand_best_indices,
        'Cont_best_indices': cont_best_indices,
        'Mix_best_indices': mix_best_indices
    })

    run_wise_df.to_csv(f"res/h{horizon}_d{search_depth}_e{num_rollouts}.csv")
