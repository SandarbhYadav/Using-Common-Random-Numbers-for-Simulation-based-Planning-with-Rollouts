import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int, default=10)

    args = parser.parse_args()
    num_runs = args.runs

    horizon = 2
    rollout_list = [10*i for i in range(1, 6)]

    dp_means = []

    # Across rollouts
    random_probs = []
    controlled_probs = []
    random_probs_se = []
    controlled_probs_se = []

    # Empirical value of identified policy
    random_means_emp = []
    controlled_means_emp = []
    random_means_emp_se = []
    controlled_means_emp_se = []

    # True values of identified policy
    random_means_true = []
    controlled_means_true = []
    random_means_true_se = []
    controlled_means_true_se = []

    for num_rollouts in rollout_list:
        print("Running for rollouts:", num_rollouts)

        cmd_random = "python", "main.py", str(num_runs), str(num_rollouts), str(horizon)
        subprocess.check_output(cmd_random, universal_newlines=True)

        res = pd.read_csv(f"res/r{num_rollouts}.csv")

        # Means and std errors are over runs
        r_prob = res['Rand_best_identified'].mean()
        c_prob = res['Cont_best_identified'].mean()
        r_prob_se = (res['Rand_best_identified'].std()) / (np.sqrt(res['Rand_best_identified'].count()))
        c_prob_se = (res['Cont_best_identified'].std()) / (np.sqrt(res['Cont_best_identified'].count()))

        random_probs.append(r_prob)
        controlled_probs.append(c_prob)
        random_probs_se.append(r_prob_se)
        controlled_probs_se.append(c_prob_se)

        # Means and std errors are over runs
        dpm = res['DP_best_val'].mean()
        rme = res['Rand_best_val_emp'].mean()
        cme = res['Cont_best_val_emp'].mean()
        rme_se = (res['Rand_best_val_emp'].std()) / (np.sqrt(res['Rand_best_val_emp'].count()))
        cme_se = (res['Cont_best_val_emp'].std()) / (np.sqrt(res['Cont_best_val_emp'].count()))

        dp_means.append(dpm)
        random_means_emp.append(rme)
        controlled_means_emp.append(cme)
        random_means_emp_se.append(rme_se)
        controlled_means_emp_se.append(cme_se)

        # Means and std errors are over runs
        rmt = res['Rand_best_val_true'].mean()
        cmt = res['Cont_best_val_true'].mean()
        rmt_se = (res['Rand_best_val_true'].std()) / (np.sqrt(res['Rand_best_val_true'].count()))
        cmt_se = (res['Cont_best_val_true'].std()) / (np.sqrt(res['Cont_best_val_true'].count()))

        random_means_true.append(rmt)
        controlled_means_true.append(cmt)
        random_means_true_se.append(rmt_se)
        controlled_means_true_se.append(cmt_se)

    print()
    plt.figure(figsize=(6, 6))
    plt.plot(rollout_list, random_probs, color='tab:red', linestyle='--', label="Independent")
    plt.fill_between(rollout_list, np.array(random_probs) - np.array(random_probs_se),
                     np.array(random_probs) + np.array(random_probs_se), color='tab:red', alpha=0.2)
    plt.plot(rollout_list, controlled_probs, color='tab:green', label="Dependent")
    plt.fill_between(rollout_list, np.array(controlled_probs) - np.array(controlled_probs_se),
                     np.array(controlled_probs) + np.array(controlled_probs_se), color='tab:green', alpha=0.2)
    plt.xlabel("Number of Monte Carlo simulations", fontsize=20)
    plt.ylabel("Probability of identifying better policy", fontsize=20)
    plt.xticks(rollout_list, fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.savefig(f'plots/prob.jpg')
    plt.close()

    plt.plot(rollout_list, dp_means, color='black', label="Best policy value")
    plt.plot(rollout_list, random_means_true, color='tab:red', linestyle='--', label="Independent")
    plt.fill_between(rollout_list, np.array(random_means_true) - np.array(random_means_true_se),
                     np.array(random_means_true) + np.array(random_means_true_se), color='tab:red', alpha=0.2)
    plt.plot(rollout_list, controlled_means_true, color='tab:green', label="Dependent")
    plt.fill_between(rollout_list, np.array(controlled_means_true) - np.array(controlled_means_true_se),
                     np.array(controlled_means_true) + np.array(controlled_means_true_se), color='tab:green', alpha=0.2)
    plt.xlabel("Number of Monte Carlo simulations", fontsize=14)
    plt.ylabel("Value of identified policy", fontsize=14)
    plt.xticks(rollout_list, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig(f'plots/true.jpg')
    plt.close()

    plt.plot(rollout_list, dp_means, color='black', label="Best policy value")
    plt.plot(rollout_list, random_means_emp, color='tab:red', linestyle='--', label="Independent")
    plt.fill_between(rollout_list, np.array(random_means_emp) - np.array(random_means_emp_se),
                     np.array(random_means_emp) + np.array(random_means_emp_se), color='tab:red', alpha=0.2)
    plt.plot(rollout_list, controlled_means_emp, color='tab:green', label="Dependent")
    plt.fill_between(rollout_list, np.array(controlled_means_emp) - np.array(controlled_means_emp_se),
                     np.array(controlled_means_emp) + np.array(controlled_means_emp_se), color='tab:green', alpha=0.2)
    plt.xlabel("Number of Monte Carlo simulations", fontsize=14)
    plt.ylabel("Empirical value of identified policy", fontsize=14)
    plt.xticks(rollout_list, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    plt.savefig(f'plots/emp.jpg')
    plt.close()

