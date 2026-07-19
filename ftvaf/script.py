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

    rollout_list = [(4*i) for i in range(1, 6)]

    # Across rollouts
    random_means = []
    controlled_means = []
    mix_means = []
    random_se = []
    controlled_se = []
    mix_se = []

    start_time = time.time()

    for num_rollouts in rollout_list:

        print("Running for rollouts:", num_rollouts)

        cmd_random = "python", "main.py", str(num_runs), str(num_rollouts), str(num_processes)
        subprocess.check_output(cmd_random, universal_newlines=True)

        res = pd.read_csv(f"res/rollouts_{num_rollouts}.csv")

        ran = res["Random"]
        con = res["Controlled"]
        mix = res["Mixed"]

        # Means
        rm = ran.mean()
        cm = con.mean()
        mm = mix.mean()

        # Standard errors
        rse = ran.std() / np.sqrt(ran.count())
        cse = con.std() / np.sqrt(con.count())
        mse = mix.std() / np.sqrt(mix.count())

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

    plt.figure(figsize=(7, 6))
    plt.plot(rollout_list, random_means, color='tab:red', linestyle='--', label="Independent")
    plt.fill_between(rollout_list, np.array(random_means) - np.array(random_se), np.array(random_means) + np.array(random_se), color='tab:red', alpha=0.2)
    plt.plot(rollout_list, controlled_means, color='tab:green', label="Dependent")
    plt.fill_between(rollout_list, np.array(controlled_means) - np.array(controlled_se), np.array(controlled_means) + np.array(controlled_se), color='tab:green', alpha=0.2)
    plt.plot(rollout_list, mix_means, color='tab:blue', linestyle='-.', label="Depth-dependent")
    plt.fill_between(rollout_list, np.array(mix_means) - np.array(mix_se), np.array(mix_means) + np.array(mix_se), color='tab:blue', alpha=0.2)
    plt.xlabel("Number of simulations", fontsize=24, fontname="Times New Roman")
    plt.ylabel("Average episodic reward", fontsize=24, fontname="Times New Roman")
    plt.xticks(rollout_list, fontsize=18)
    plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.yticks(fontsize=18)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
    plt.legend(fontsize=24, prop={'family': 'Times New Roman', 'size': 24})
    plt.tight_layout()
    plt.savefig('res_ftvaf.jpg')
    plt.close()

    ran_stats = []
    con_stats = []
    mix_stats = []

    for i in range(len(rollout_list)):
        ranstat = str(np.round(random_means[i], 4)) + " (" + str(np.round(random_se[i], 4)) + ")"
        ran_stats.append(ranstat)
        constat = str(np.round(controlled_means[i], 4)) + " (" + str(np.round(controlled_se[i], 4)) + ")"
        con_stats.append(constat)
        mixstat = str(np.round(mix_means[i], 4)) + " (" + str(np.round(mix_se[i], 4)) + ")"
        mix_stats.append(mixstat)

    res_df = pd.DataFrame({
        'Simulations': rollout_list,
        'Random': ran_stats,
        'Controlled': con_stats,
        'Mixed': mix_stats
    })

    res_df.to_csv('res_table.csv')
