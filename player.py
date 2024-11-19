#!/usr/bin/env python3
import random

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

import numpy as np
from fishing_game_core.game_tree import State


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    
    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        # random_move = random.randrange(5)
        # return ACTION_TO_STR[random_move]
        
        max_depth = 3
        alpha = -float('inf')
        beta = float('inf')
        best_move = 0
        max_score = -float('inf')
        player_id = 0  
       
        children_nodes = initial_tree_node.compute_and_get_children()
        for child in children_nodes:
            # score = self.minmax(child, max_depth, player_id)
            score = self.alphabeta(child, max_depth, alpha, beta, player_id)
            if score >= max_score:
                max_score = score
                best_move = child.move
        # print(f'++++++++++++++ {ACTION_TO_STR[best_move]}')
        return ACTION_TO_STR[best_move]
    
    def minmax(self, node: Node, max_depth, player_id):
        if node.depth == max_depth:
            # print(f'================== player: {player_id}, depth: {node.depth}')
            return self.heuristic(node.state, player_id)
        
        children_node = node.compute_and_get_children()
        if player_id == 0:
            best_possible = -float('inf')
            for child in children_node:
                score = self.minmax(child, max_depth, 1)
                best_possible = max(best_possible, score)
            # print(f'================== player: {player_id}, depth: {node.depth}, best: {best_possible}')
            return best_possible
        elif player_id == 1:
            best_possible = float('inf')
            for child in children_node:
                score = self.minmax(child, max_depth, 0)
                best_possible = min(best_possible, score)
            # print(f'================== player: {player_id}, depth: {node.depth}, best: {best_possible}')
            return best_possible
    
    def alphabeta(self, node: Node, max_depth, alpha, beta, player_id):
        if node.depth == max_depth:
            return self.heuristic(node.state, player_id)
        
        children_node = node.compute_and_get_children()
        if player_id == 0:
            best_possible = -float('inf')
            for child in children_node:
                score = self.alphabeta(child, max_depth, alpha, beta, 1)
                best_possible = max(best_possible, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return best_possible
        elif player_id == 1:
            best_possible = float('inf')
            for child in children_node:
                score = self.alphabeta(child, max_depth, alpha, beta, 0)
                best_possible = min(best_possible, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return best_possible
        
    def heuristic(self, state: State, player):
        result_value = 0

        competitor = 1 - player

        result_value += state.player_scores[player] - state.player_scores[competitor]

        hook_pos = state.hook_positions[player]
        for fish_id, fish_pos in state.fish_positions.items():
            fish_score = state.fish_scores[fish_id]
            weigth = self.distance_between_hook_fish(fish_pos, hook_pos)
            result_value += fish_score / (weigth + 1e-10) * 10
        return result_value
    
    def distance_between_hook_fish(self, fish_pos, hook_pos):
        return np.sqrt((fish_pos[0] - hook_pos[0]) ** 2 + (fish_pos[1] - hook_pos[1]) ** 2)
