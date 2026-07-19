import time
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("runs", type=int, default=1)
    parser.add_argument("processes", type=int, default=1)

    args = parser.parse_args()
    num_runs = args.runs
    num_processes = args.processes

    depth_list = [1, 2, 3]
    horizon_list = [20]

    start_time = time.time()

    for hor in horizon_list:
        print("Running for horizon:", hor)
        print()
        for dep in depth_list:
            print("Running for depth:", dep)

            dp_means = []

            # Across rollouts
            random_probs = []
            controlled_probs = []
            mix_probs = []
            random_probs_se = []
            controlled_probs_se = []
            mix_probs_se = []

            # Empirical value of identified policy
            random_means_emp = []
            controlled_means_emp = []
            mix_means_emp = []
            random_means_emp_se = []
            controlled_means_emp_se = []
            mix_means_emp_se = []

            # True values of identified policy
            random_means_true = []
            controlled_means_true = []
            mix_means_true = []
            random_means_true_se = []
            controlled_means_true_se = []
            mix_means_true_se = []

            if dep == 1:
                rollout_list = [2 ** p for p in range(1, 7)]
            else:
                rollout_list = [2 ** p for p in range(1, 11)]

            log_rollout_list = [np.log2(i) for i in rollout_list]

            for num_rollouts in rollout_list:

                print("Running for rollouts:", num_rollouts)

                cmd = "python", "main.py", str(num_runs), str(num_rollouts), str(dep), str(hor), str(num_processes)
                subprocess.check_output(cmd, universal_newlines=True)

                res = pd.read_csv(f"res/h{hor}_d{dep}_e{num_rollouts}.csv")

                # Means and standard errors are over runs
                r_prob = res['Rand_best_identified'].mean()
                c_prob = res['Cont_best_identified'].mean()
                m_prob = res['Mix_best_identified'].mean()
                r_prob_se = (res['Rand_best_identified'].std())/(np.sqrt(res['Rand_best_identified'].count()))
                c_prob_se = (res['Cont_best_identified'].std())/(np.sqrt(res['Cont_best_identified'].count()))
                m_prob_se = (res['Mix_best_identified'].std())/(np.sqrt(res['Mix_best_identified'].count()))

                random_probs.append(r_prob)
                controlled_probs.append(c_prob)
                mix_probs.append(m_prob)
                random_probs_se.append(r_prob_se)
                controlled_probs_se.append(c_prob_se)
                mix_probs_se.append(m_prob_se)

                # Means and standard errors are over runs
                dpm = res['DP_best_val'].mean()
                rme = res['Rand_best_val_emp'].mean()
                cme = res['Cont_best_val_emp'].mean()
                mme = res['Mix_best_val_emp'].mean()
                rme_se = (res['Rand_best_val_emp'].std())/(np.sqrt(res['Rand_best_val_emp'].count()))
                cme_se = (res['Cont_best_val_emp'].std())/(np.sqrt(res['Cont_best_val_emp'].count()))
                mme_se = (res['Mix_best_val_emp'].std())/(np.sqrt(res['Mix_best_val_emp'].count()))

                dp_means.append(dpm)
                random_means_emp.append(rme)
                controlled_means_emp.append(cme)
                mix_means_emp.append(mme)
                random_means_emp_se.append(rme_se)
                controlled_means_emp_se.append(cme_se)
                mix_means_emp_se.append(mme_se)

                # Means and standard errors are over runs
                rmt = res['Rand_best_val_true'].mean()
                cmt = res['Cont_best_val_true'].mean()
                mmt = res['Mix_best_val_true'].mean()
                rmt_se = (res['Rand_best_val_true'].std())/(np.sqrt(res['Rand_best_val_true'].count()))
                cmt_se = (res['Cont_best_val_true'].std())/(np.sqrt(res['Cont_best_val_true'].count()))
                mmt_se = (res['Mix_best_val_true'].std())/(np.sqrt(res['Mix_best_val_true'].count()))

                random_means_true.append(rmt)
                controlled_means_true.append(cmt)
                mix_means_true.append(mmt)
                random_means_true_se.append(rmt_se)
                controlled_means_true_se.append(cmt_se)
                mix_means_true_se.append(mmt_se)

                elapsed = time.time() - start_time  # seconds (float)

                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)

                print(f"Time elapsed: {hours:02d}hr {minutes:02d}min {seconds:02d}sec")

            print()
            plt.figure(figsize=(10, 6))
            plt.plot(rollout_list, random_probs, color='tab:red')
            plt.fill_between(rollout_list, np.array(random_probs) - np.array(random_probs_se), np.array(random_probs) + np.array(random_probs_se), color='tab:red', alpha=0.2)
            plt.plot(rollout_list, controlled_probs, color='tab:green')
            plt.fill_between(rollout_list, np.array(controlled_probs) - np.array(controlled_probs_se), np.array(controlled_probs) + np.array(controlled_probs_se), color='tab:green', alpha=0.2)
            plt.plot(rollout_list, mix_probs, color='tab:blue')
            plt.fill_between(rollout_list, np.array(mix_probs) - np.array(mix_probs_se), np.array(mix_probs) + np.array(mix_probs_se), color='tab:blue', alpha=0.2)
            plt.xlabel("Number of rollouts")
            plt.ylabel("Probability of identifying best policy")
            plt.xticks(rollout_list)
            plt.legend(["Random", "Random CI", "Controlled", "Controlled CI", "Mixed", "Mixed CI"])
            plt.title(f"Averaged across {num_runs} runs")
            plt.savefig(f'plots_prob/plot_h{hor}_d{dep}.jpg')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(rollout_list, dp_means, color='black')
            plt.plot(rollout_list, random_means_true, color='tab:red')
            plt.fill_between(rollout_list, np.array(random_means_true) - np.array(random_means_true_se), np.array(random_means_true) + np.array(random_means_true_se), color='tab:red', alpha=0.2)
            plt.plot(rollout_list, controlled_means_true, color='tab:green')
            plt.fill_between(rollout_list, np.array(controlled_means_true) - np.array(controlled_means_true_se), np.array(controlled_means_true) + np.array(controlled_means_true_se), color='tab:green', alpha=0.2)
            plt.plot(rollout_list, mix_means_true, color='tab:blue')
            plt.fill_between(rollout_list, np.array(mix_means_true) - np.array(mix_means_true_se), np.array(mix_means_true) + np.array(mix_means_true_se), color='tab:blue', alpha=0.2)
            plt.xlabel("Number of rollouts")
            plt.ylabel("True value of identified policy")
            plt.xticks(rollout_list)
            plt.legend(["DP", "Random", "Random CI", "Controlled", "Controlled CI", "Mixed", "Mixed CI"])
            plt.title(f"Averaged across {num_runs} runs")
            plt.savefig(f'plots_val_true/plot_h{hor}_d{dep}.jpg')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(rollout_list, dp_means, color='black')
            plt.plot(rollout_list, random_means_emp, color='tab:red')
            plt.fill_between(rollout_list, np.array(random_means_emp) - np.array(random_means_emp_se), np.array(random_means_emp) + np.array(random_means_emp_se), color='tab:red', alpha=0.2)
            plt.plot(rollout_list, controlled_means_emp, color='tab:green')
            plt.fill_between(rollout_list, np.array(controlled_means_emp) - np.array(controlled_means_emp_se), np.array(controlled_means_emp) + np.array(controlled_means_emp_se), color='tab:green', alpha=0.2)
            plt.plot(rollout_list, mix_means_emp, color='tab:blue')
            plt.fill_between(rollout_list, np.array(mix_means_emp) - np.array(mix_means_emp_se), np.array(mix_means_emp) + np.array(mix_means_emp_se), color='tab:blue', alpha=0.2)
            plt.xlabel("Number of rollouts")
            plt.ylabel("Empirical value of identified policy")
            plt.xticks(rollout_list)
            plt.legend(["DP", "Random", "Random CI", "Controlled", "Controlled CI", "Mixed", "Mixed CI"])
            plt.title(f"Averaged across {num_runs} runs")
            plt.savefig(f'plots_val_emp/plot_h{hor}_d{dep}.jpg')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(log_rollout_list, random_probs, color='tab:red')
            plt.fill_between(log_rollout_list, np.array(random_probs) - np.array(random_probs_se), np.array(random_probs) + np.array(random_probs_se), color='tab:red', alpha=0.2)
            plt.plot(log_rollout_list, controlled_probs, color='tab:green')
            plt.fill_between(log_rollout_list, np.array(controlled_probs) - np.array(controlled_probs_se), np.array(controlled_probs) + np.array(controlled_probs_se), color='tab:green', alpha=0.2)
            plt.plot(log_rollout_list, mix_probs, color='tab:blue')
            plt.fill_between(log_rollout_list, np.array(mix_probs) - np.array(mix_probs_se), np.array(mix_probs) + np.array(mix_probs_se), color='tab:blue', alpha=0.2)
            plt.xlabel("Number of rollouts (in powers of 2)")
            plt.ylabel("Probability of identifying best policy")
            plt.xticks(log_rollout_list)
            plt.legend(["Random", "Random CI", "Controlled", "Controlled CI", "Mixed", "Mixed CI"])
            plt.title(f"Averaged across {num_runs} runs")
            plt.savefig(f'log_plots_prob/plot_h{hor}_d{dep}.jpg')
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(log_rollout_list, dp_means, color='black', label="Best policy value")
            plt.plot(log_rollout_list, random_means_true, color='tab:red', linestyle='--', label="Independent")
            plt.fill_between(log_rollout_list, np.array(random_means_true) - np.array(random_means_true_se), np.array(random_means_true) + np.array(random_means_true_se), color='tab:red', alpha=0.2, label="_hidden")
            plt.plot(log_rollout_list, controlled_means_true, color='tab:green', label="Dependent")
            plt.fill_between(log_rollout_list, np.array(controlled_means_true) - np.array(controlled_means_true_se), np.array(controlled_means_true) + np.array(controlled_means_true_se), color='tab:green', alpha=0.2, label="_hidden")
            plt.plot(log_rollout_list, mix_means_true, color='tab:blue', linestyle='-.', label="Depth-dependent")
            plt.fill_between(log_rollout_list, np.array(mix_means_true) - np.array(mix_means_true_se), np.array(mix_means_true) + np.array(mix_means_true_se), color='tab:blue', alpha=0.2, label="_hidden")
            plt.xlabel("Number of Monte Carlo simulations", fontsize=20)
            plt.ylabel("Value of identified policy", fontsize=20)
            plt.xticks(log_rollout_list, [rf"$2^{{{int(np.log2(v))}}}$" for v in rollout_list], fontsize=18)
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
            plt.yticks(fontsize=18)
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig(f'log_plots_val_true/syn_d{dep}.jpg')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(log_rollout_list, dp_means, color='black')
            plt.plot(log_rollout_list, random_means_emp, color='tab:red')
            plt.fill_between(log_rollout_list, np.array(random_means_emp) - np.array(random_means_emp_se), np.array(random_means_emp) + np.array(random_means_emp_se), color='tab:red', alpha=0.2)
            plt.plot(log_rollout_list, controlled_means_emp, color='tab:green')
            plt.fill_between(log_rollout_list, np.array(controlled_means_emp) - np.array(controlled_means_emp_se), np.array(controlled_means_emp) + np.array(controlled_means_emp_se), color='tab:green', alpha=0.2)
            plt.plot(log_rollout_list, mix_means_emp, color='tab:blue')
            plt.fill_between(log_rollout_list, np.array(mix_means_emp) - np.array(mix_means_emp_se), np.array(mix_means_emp) + np.array(mix_means_emp_se), color='tab:blue', alpha=0.2)
            plt.xlabel("Number of rollouts (in powers of 2)")
            plt.ylabel("Empirical value of identified policy")
            plt.xticks(log_rollout_list)
            plt.legend(["DP", "Random", "Random CI", "Controlled", "Controlled CI", "Mixed", "Mixed CI"])
            plt.title(f"Averaged across {num_runs} runs")
            plt.savefig(f'log_plots_val_emp/plot_h{hor}_d{dep}.jpg')
            plt.close()
