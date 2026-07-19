import math
import numpy as np
from copy import deepcopy


# Mixed radix encoding
def encode2(x1, x2, ranges):
    A, B = ranges
    return x1 * B + x2


class BaseNode:
    def __init__(self, game_state, parent=None, depth=0):
        self.game_state = game_state
        self.parent = parent
        self.depth = depth

        self.visits = 0
        self.total_reward = 0.0
        self.average_reward = 0.0

        self.is_terminal = (game_state.get_winner_of_game() != -1)


class DecisionNode(BaseNode):
    def __init__(self, game_state, parent=None, depth=0):
        super().__init__(game_state, parent, depth)

        # Valid actions at this node
        self.untried_actions = list(game_state.current_move_pieces)
        self.children = {}  # action -> ChanceNode

    def ucb(self, child, c):
        if child.visits == 0:
            return float("inf")
        return (
            child.average_reward
            + c * math.sqrt(math.log(self.visits) / child.visits)
        )


class ChanceNode(BaseNode):
    def __init__(self, game_state, parent=None, depth=0, action=None):
        super().__init__(game_state, parent, depth)

        self.children = {}  # dice_value -> DecisionNode
        self.action_from_parent = action
        self.outcomes = [1, 2, 3, 4, 5, 6]
        self.prob = 1.0 / 6.0


class UCTPlanner:
    def __init__(self, iterations, ucb_depth, exploration_c=1.4, mode=None):
        self.iterations = iterations
        self.D = ucb_depth
        self.c = exploration_c
        self.mode = mode

        self.root_action = None
        self.count_root_actions = [0 for a in range(4)]  # Number of times each action is taken at root node across rollouts

        self.ranges = (4, iterations)

        self.dice_rng = np.random.default_rng()
        self.opp_rng = np.random.default_rng()

    def select(self, node):
        while not node.is_terminal and node.depth < self.D:

            if isinstance(node, DecisionNode):
                if node.untried_actions:
                    return node
                if not node.children:  # For corner case when agent doesn't have legal actions for a state
                    return node

                node = max(node.children.values(), key=lambda child: node.ucb(child, self.c))
                corresponding_action = node.action_from_parent

                if node.parent.depth == 0:  # If parent is root node
                    self.root_action = corresponding_action
                    self.count_root_actions[self.root_action] += 1

                    if (self.mode == "random") or (self.mode == "mixed"):
                        my_seed = encode2(self.root_action, self.count_root_actions[self.root_action], self.ranges)
                    elif self.mode == "controlled":
                        my_seed = self.count_root_actions[self.root_action]
                    self.dice_rng = np.random.default_rng(my_seed)
                    self.opp_rng = np.random.default_rng(my_seed)

            elif isinstance(node, ChanceNode):
                unvisited = [d for d in node.outcomes if d not in node.children]
                if unvisited:  # Expand if chance node has unvisited dice outcomes
                    return node
                # All dice outcomes have been visited at least once now
                dice = self.dice_rng.choice(node.outcomes)
                node = node.children[dice]

        return node

    def expand(self, node):

        if isinstance(node, DecisionNode):

            if len(node.untried_actions):
                action = node.untried_actions.pop()
            elif not node.children:
                action = -1  # Corner case when agent doesn't have legal actions for a state

            if node.depth == 0:
                self.root_action = action
                self.count_root_actions[self.root_action] += 1

                if (self.mode == "random") or (self.mode == "mixed"):
                    my_seed1 = encode2(self.root_action, self.count_root_actions[self.root_action], self.ranges)
                elif self.mode == "controlled":
                    my_seed1 = self.count_root_actions[self.root_action]

                self.dice_rng = np.random.default_rng(my_seed1)
                self.opp_rng = np.random.default_rng(my_seed1)

            game = deepcopy(node.game_state)  # Deep copy from parent for child

            game.answer_observation(action)

            child = ChanceNode(game, parent=node, depth=node.depth + 1, action=action)
            node.children[action] = child
            return child

        elif isinstance(node, ChanceNode):

            remaining = [d for d in node.outcomes if d not in node.children]

            game = deepcopy(node.game_state)  # Deep copy from parent for child

            # Opponent has to be skipped if agent gets extra chance
            while game.current_player == 0:  # Six on dice can give extra chance to opponent
                opp_dice = self.dice_rng.choice([1, 2, 3, 4, 5, 6])
                (_, opp_moves, _, _, _, _), _ = game.get_observation(forced_dice=opp_dice)
                if len(opp_moves):
                    opp_action = self.opp_rng.choice(opp_moves)
                else:
                    opp_action = -1
                game.answer_observation(opp_action)

            dice = self.dice_rng.choice(remaining)
            game.get_observation(forced_dice=dice)  # For moving to next state for child node

            child = DecisionNode(game, parent=node, depth=node.depth + 1)
            node.children[dice] = child
            return child

    def rollout(self, node):
        game = deepcopy(node.game_state)
        depth = node.depth

        if self.mode == "random":
            roll_seed = encode2(self.root_action, self.count_root_actions[self.root_action], self.ranges)
        elif (self.mode == "controlled") or (self.mode == "mixed"):
            roll_seed = self.count_root_actions[self.root_action]
        roll_rng = np.random.default_rng(seed=roll_seed)

        if self.mode == "mixed":
            my_seed2 = self.count_root_actions[self.root_action]
            self.dice_rng = np.random.default_rng(my_seed2)
            self.opp_rng = np.random.default_rng(my_seed2)

        if isinstance(node, ChanceNode):
            dice = self.dice_rng.choice([1, 2, 3, 4, 5, 6])
            (_, moves, _, _, _, _), curr_player = game.get_observation(forced_dice=dice)
        else:
            moves = game.current_move_pieces
            curr_player = game.current_player

        while game.get_winner_of_game() == -1:
            if curr_player == 1:
                if len(moves):
                    act = roll_rng.choice(moves)
                else:
                    act = -1
                _, _, _, _, _, there_is_a_winner = game.answer_observation(act)
                depth += 2  # Factoring chance nodes also
            else:
                if len(moves):
                    ac = self.opp_rng.choice(moves)
                else:
                    ac = -1
                _, _, _, _, _, there_is_a_winner = game.answer_observation(ac)

            if there_is_a_winner:
                break
            dic = self.dice_rng.choice([1, 2, 3, 4, 5, 6])
            (_, moves, _, _, _, _), curr_player = game.get_observation(forced_dice=dic)

        winner = game.first_winner_was
        if winner == 1:
            reward = 1
        else:
            reward = 0

        return reward

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            node.average_reward = node.total_reward / node.visits
            node = node.parent

    def run_iteration(self, root):

        # Selection
        node = self.select(root)

        # Expansion
        if not node.is_terminal and node.depth < self.D:
            node = self.expand(node)

        # Rollout
        reward = self.rollout(node)

        # Backprop
        self.backpropagate(node, reward)

    def plan(self, root_game):

        root = DecisionNode(deepcopy(root_game))

        for _ in range(self.iterations):
            self.run_iteration(root)
        # Return action corresponding to best child node
        best_action, _ = max(root.children.items(), key=lambda item: item[1].average_reward)

        return best_action
