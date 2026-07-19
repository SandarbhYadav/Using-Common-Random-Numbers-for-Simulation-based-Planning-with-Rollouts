import time
import argparse
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("games", type=int, default=1)
    parser.add_argument("processes", type=int, default=1)

    args = parser.parse_args()
    num_games = args.games
    num_processes = args.processes

    depth_list = [2]

    start_time = time.time()

    for dep in depth_list:
        print("Running for depth:", dep)
        random_means = []
        controlled_means = []
        mix_means = []
        random_se = []
        controlled_se = []
        mix_se = []

        rollout_list = [int(2**(i/2)) for i in range(12, 21)]
        log_rollout_list = [np.log2(i) for i in rollout_list]

        for num_rollouts in rollout_list:
            print("Running for rollouts:", num_rollouts)

            cmd_random = "python", "main.py", str(num_rollouts), str(dep), str(num_games), str(num_processes)
            subprocess.check_output(cmd_random, universal_newlines=True)

            res = pd.read_csv(f"res/d{dep}_e{num_rollouts}.csv")

            rm = res['Rand'].mean()
            cm = res['Cont'].mean()
            mm = res['Mix'].mean()
            rse = (res['Rand'].std()) / (np.sqrt(res['Rand'].count()))
            cse = (res['Cont'].std()) / (np.sqrt(res['Cont'].count()))
            mse = (res['Mix'].std()) / (np.sqrt(res['Mix'].count()))

            random_means.append(rm)
            controlled_means.append(cm)
            mix_means.append(mm)
            random_se.append(rse)
            controlled_se.append(cse)
            mix_se.append(mse)

            elapsed = time.time() - start_time  # seconds (float)

            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)

            print(f"Time elapsed: {hours:02d}hr {minutes:02d}min {seconds:02d}sec")

        print()

        plt.figure(figsize=(10, 6))
        plt.plot(rollout_list, random_means, color='tab:red')
        plt.fill_between(rollout_list, np.array(random_means) - np.array(random_se), np.array(random_means) + np.array(random_se), color='tab:red', alpha=0.2)
        plt.plot(rollout_list, controlled_means, color='tab:green')
        plt.fill_between(rollout_list, np.array(controlled_means) - np.array(controlled_se), np.array(controlled_means) + np.array(controlled_se), color='tab:green', alpha=0.2)
        plt.plot(rollout_list, mix_means, color='tab:blue')
        plt.fill_between(rollout_list, np.array(mix_means) - np.array(mix_se), np.array(mix_means) + np.array(mix_se), color='tab:blue', alpha=0.2)
        plt.xlabel("Number of simulations")
        plt.ylabel("Win percentage of UCT agent against Random agent")
        plt.xticks(rollout_list)
        plt.legend(["Random", "Random CI", "Controlled", "Controlled CI", "Mixed", "Mixed CI"])
        plt.title(f"Averaged across {num_games} games")
        plt.savefig('plot_non_log.jpg')
        plt.close()

        plt.figure(figsize=(7.5, 6))
        plt.plot(log_rollout_list, random_means, color='tab:red', linestyle='--', label="Independent")
        plt.fill_between(log_rollout_list, np.array(random_means) - np.array(random_se), np.array(random_means) + np.array(random_se), color='tab:red', alpha=0.2)
        plt.plot(log_rollout_list, controlled_means, color='tab:green', label="Dependent")
        plt.fill_between(log_rollout_list, np.array(controlled_means) - np.array(controlled_se), np.array(controlled_means) + np.array(controlled_se), color='tab:green', alpha=0.2)
        plt.plot(log_rollout_list, mix_means, color='tab:blue', linestyle='-.', label="Depth-dependent")
        plt.fill_between(log_rollout_list, np.array(mix_means) - np.array(mix_se), np.array(mix_means) + np.array(mix_se), color='tab:blue', alpha=0.2)
        plt.xlabel("Number of simulations", fontsize=24, fontname="Times New Roman")
        plt.ylabel("Win percentage", fontsize=24, fontname="Times New Roman")
        plt.xticks(log_rollout_list, [rf"$2^{{{int(np.log2(v))}}}$" for v in rollout_list], fontsize=18)
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
        plt.yticks(fontsize=18)
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
        plt.legend(fontsize=24, prop={'family': 'Times New Roman', 'size': 24})
        plt.tight_layout()
        plt.savefig('plot_log.jpg')
        plt.close()
