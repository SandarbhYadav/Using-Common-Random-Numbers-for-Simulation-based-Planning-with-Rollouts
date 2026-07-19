import copy
import time
import argparse
import numpy as np
import pandas as pd
from uct import UCTPlanner
from ludopy.game import Game
from multiprocessing import Pool


def run_one_game(num_rollouts, search_depth, mode, g_ind):

    game = Game(ghost_players=[2, 3], seed=g_ind)  # Players 0 and 1 playing
    rng = np.random.default_rng(seed=g_ind)

    there_is_a_winner = False
    time_step = 0

    while not there_is_a_winner:

        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
        if len(move_pieces):
            # Opponent
            if player_i == 0:
                piece_to_move = rng.choice(move_pieces)
            # UCT player
            else:
                if len(move_pieces) > 1:

                    # New planner for each decision time step
                    planner = UCTPlanner(
                        iterations=num_rollouts,
                        ucb_depth=search_depth,
                        exploration_c=1.4,
                        mode=mode
                    )

                    piece_to_move = planner.plan(root_game=copy.deepcopy(game))
                else:
                    piece_to_move = move_pieces[0]
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = game.answer_observation(piece_to_move)
        time_step += 1

    winner = game.first_winner_was
    if winner == 1:
        score = 1
    else:
        score = 0
    return score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=int, default=1)
    parser.add_argument("depth", type=int, default=1)
    parser.add_argument("games", type=int, default=1)
    parser.add_argument("processes", type=int, default=1)

    args = parser.parse_args()
    num_rollouts = args.rollouts
    search_depth = args.depth
    num_games = args.games
    num_processes = args.processes

    print("Number of rollouts:", num_rollouts)
    print("Search depth:", search_depth)
    print("Number of games:", num_games)
    print("Number of processes:", num_processes)

    start_time = time.time()

    # Random (Independent)

    print("Running random")

    with Pool(num_processes) as pool:
        rand_results = pool.starmap(
            run_one_game,
            [(num_rollouts, search_depth, "random", i) for i in range(num_games)]
        )

    elapsed = time.time() - start_time  # seconds (float)

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"Time elapsed: {hours:02d}hr {minutes:02d}min {seconds:02d}sec")

    # Controlled (Dependent)

    print("Running controlled")

    with Pool(num_processes) as pool:
        cont_results = pool.starmap(
            run_one_game,
            [(num_rollouts, search_depth, "controlled", i) for i in range(num_games)]
        )

    elapsed = time.time() - start_time  # seconds (float)

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"Time elapsed: {hours:02d}hr {minutes:02d}min {seconds:02d}sec")

    # Mixed (Depth-dependent)

    print("Running mixed")

    with Pool(num_processes) as pool:
        mix_results = pool.starmap(
            run_one_game,
            [(num_rollouts, search_depth, "mixed", i) for i in range(num_games)]
        )

    elapsed = time.time() - start_time  # seconds (float)

    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print(f"Time elapsed: {hours:02d}hr {minutes:02d}min {seconds:02d}sec")

    game_wise_df = pd.DataFrame({
        'Game_ind': [gi for gi in range(num_games)],
        'Rand': rand_results,
        'Cont': cont_results,
        'Mix': mix_results
    })

    game_wise_df.to_csv(f"res/d{search_depth}_e{num_rollouts}.csv")
