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

    depth_list = [1, 2, 3, 4, 5]
    horizon_list = [20]

    start_time = time.time()

    for hor in horizon_list:
        print("Running for horizon:", hor)
        print()
        for dep in depth_list:
            print("Running for depth:", dep)

            dp_means = []
            random_means = []
            controlled_means = []
            mix_means = []
            random_means_se = []
            controlled_means_se = []
            mix_means_se = []

            rollout_list = [2 ** p for p in range(1, 11)]

            log_rollout_list = [np.log2(i) for i in rollout_list]

            for num_rollouts in rollout_list:

                print("Running for rollouts:", num_rollouts)

                cmd_random = "python", "main.py", str(num_runs), str(num_rollouts), str(dep), str(hor), str(num_processes)
                subprocess.check_output(cmd_random, universal_newlines=True)

                res = pd.read_csv(f"res/h{hor}_d{dep}_e{num_rollouts}.csv")

                # Means and standard errors are over runs
                dpm = res['DP_val'].mean()
                rme = res['Rand_val'].mean()
                cme = res['Cont_val'].mean()
                mme = res['Mix_val'].mean()
                rme_se = (res['Rand_val'].std()) / (np.sqrt(res['Rand_val'].count()))
                cme_se = (res['Cont_val'].std()) / (np.sqrt(res['Cont_val'].count()))
                mme_se = (res['Mix_val'].std()) / (np.sqrt(res['Mix_val'].count()))

                dp_means.append(dpm)
                random_means.append(rme)
                controlled_means.append(cme)
                mix_means.append(mme)
                random_means_se.append(rme_se)
                controlled_means_se.append(cme_se)
                mix_means_se.append(mme_se)

                elapsed = time.time() - start_time  # seconds (float)

                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)

                print(f"Time elapsed: {hours:02d}hr {minutes:02d}min {seconds:02d}sec")

            print()

            plt.figure(figsize=(10, 6))
            plt.plot(rollout_list, dp_means, color='black')
            plt.plot(rollout_list, random_means, color='tab:red')
            plt.fill_between(rollout_list, np.array(random_means) - np.array(random_means_se), np.array(random_means) + np.array(random_means_se), color='tab:red', alpha=0.2)
            plt.plot(rollout_list, controlled_means, color='tab:green')
            plt.fill_between(rollout_list, np.array(controlled_means) - np.array(controlled_means_se), np.array(controlled_means) + np.array(controlled_means_se), color='tab:green', alpha=0.2)
            plt.plot(rollout_list, mix_means, color='tab:blue')
            plt.fill_between(rollout_list, np.array(mix_means) - np.array(mix_means_se), np.array(mix_means) + np.array(mix_means_se), color='tab:blue', alpha=0.2)
            plt.xlabel("Number of simulations")
            plt.ylabel("Value")
            plt.xticks(rollout_list)
            plt.legend(["DP", "Random", "Random CI", "Controlled", "Controlled CI", "Mixed", "Mixed CI"])
            plt.title(f"Averaged across {num_runs} runs")
            plt.savefig(f'plots/plot_h{hor}_d{dep}.jpg')
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.plot(log_rollout_list, dp_means, color='black', label="Optimal value")
            plt.plot(log_rollout_list, random_means, color='tab:red', linestyle='--', label="Independent")
            plt.fill_between(log_rollout_list, np.array(random_means) - np.array(random_means_se), np.array(random_means) + np.array(random_means_se), color='tab:red', alpha=0.2)
            plt.plot(log_rollout_list, controlled_means, color='tab:green', label="Dependent")
            plt.fill_between(log_rollout_list, np.array(controlled_means) - np.array(controlled_means_se), np.array(controlled_means) + np.array(controlled_means_se), color='tab:green', alpha=0.2)
            plt.plot(log_rollout_list, mix_means, color='tab:blue', linestyle='-.', label="Depth-dependent")
            plt.fill_between(log_rollout_list, np.array(mix_means) - np.array(mix_means_se), np.array(mix_means) + np.array(mix_means_se), color='tab:blue', alpha=0.2)
            plt.xlabel("Number of simulations", fontsize=20)
            plt.ylabel("Cumulative reward", fontsize=20)
            plt.xticks(log_rollout_list, [rf"$2^{{{int(np.log2(v))}}}$" for v in rollout_list], fontsize=18)
            plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
            plt.yticks(fontsize=18)
            plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig(f'log_plots/plot_h{hor}_d{dep}.jpg')
            plt.close()
