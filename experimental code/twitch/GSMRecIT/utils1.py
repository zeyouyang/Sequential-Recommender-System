#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np


def build_graph(train_data):
    graph = nx.DiGraph()
    for seq in train_data:
        for i in range(len(seq) - 1):
            if graph.get_edge_data(seq[i], seq[i + 1]) is None:
                weight = 1
            else:
                weight = graph.get_edge_data(seq[i], seq[i + 1])['weight'] + 1
            graph.add_edge(seq[i], seq[i + 1], weight=weight)
    for node in graph.nodes:
        sum = 0
        for j, i in graph.in_edges(node):
            sum += graph.get_edge_data(j, i)['weight']
        if sum != 0:
            for j, i in graph.in_edges(i):
                graph.add_edge(j, i, weight=graph.get_edge_data(j, i)['weight'] / sum)
    return graph


def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max


def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs_game = data[0]
        inputs_game , masks_game, len_max = data_masks(inputs_game , [0])
        self.inputs_game  = np.asarray(inputs_game)
        self.mask_game = np.asarray(masks_game)
        self.len_max = len_max
        self.targets_game = np.asarray(data[1])
        self.length = len(inputs_game)


        inputs_duration = data[2]
        inputs_duration , masks_duration, len_max = data_masks(inputs_duration , [0])
        self.inputs_duration  = np.asarray(inputs_duration)
        self.mask_duration = np.asarray(masks_duration)
        self.targets_duration = np.asarray(data[3])
        

        inputs_interval = data[4]
        inputs_interval , masks_interval, len_max = data_masks(inputs_interval , [0])
        self.inputs_interval  = np.asarray(inputs_interval)
        self.mask_interval = np.asarray(masks_interval)
        self.targets_interval = np.asarray(data[5])
        

        self.length = len(inputs_game)
        self.shuffle = shuffle
        self.graph = graph

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)

            self.inputs_game= self.inputs_game[shuffled_arg]
            self.inputs_duration= self.inputs_duration[shuffled_arg]
            self.inputs_interval= self.inputs_interval[shuffled_arg]

            
            self.mask_game= self.mask_game[shuffled_arg]
            self.mask_duration= self.mask_duration[shuffled_arg]
            self.mask_interval= self.mask_interval[shuffled_arg]


            self.targets_game = self.targets_game[shuffled_arg]
            self.targets_duration = self.targets_duration[shuffled_arg]
            self.targets_interval = self.targets_interval[shuffled_arg]

        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices
#####改到這

    def get_slice(self, i):
        inputs_game, mask_game, targets_game = self.inputs_game[i], self.mask_game[i], self.targets_game[i]
        items_game, n_node_game, A_game, alias_inputs_game = [], [], [], []
        #items, n_node, A, alias_inputs = [], [], [], []
        for u_input_game in inputs_game:
             n_node_game.append(len(np.unique(u_input_game)))
        max_n_node_game = np.max(n_node_game)
        for u_input_game in inputs_game:
            node_game = np.unique(u_input_game)
            items_game.append(node_game.tolist() + (max_n_node_game - len(node_game)) * [0])
            u_A_game = np.zeros((max_n_node_game, max_n_node_game))
            for i in np.arange(len(u_input_game) - 1):
            #for i in np.arange(len(u_input) - 1):
                if u_input_game[i + 1] == 0:
                    break
                u_game = np.where(node_game == u_input_game[i])[0][0]
                v_game = np.where(node_game == u_input_game[i + 1])[0][0]
                u_A_game[u_game][v_game] = 1
            u_sum_in_game = np.sum(u_A_game, 0)
            u_sum_in_game[np.where(u_sum_in_game == 0)] = 1
            u_A_in_game = np.divide(u_A_game, u_sum_in_game)
            u_sum_out_game = np.sum(u_A_game, 1)
            u_sum_out_game[np.where(u_sum_out_game == 0)] = 1
            u_A_out_game = np.divide(u_A_game.transpose(), u_sum_out_game)
            u_A_game = np.concatenate([u_A_in_game, u_A_out_game]).transpose()
            A_game.append(u_A_game)
            alias_inputs_game.append([np.where(node_game == i)[0][0] for i in u_input_game])
        return alias_inputs_game, A_game, items_game, mask_game, targets_game
    def get_slice1(self, i):   
        inputs_duration, mask_duration, targets_duration = self.inputs_duration[i], self.mask_duration[i], self.targets_duration[i]
        items_duration, n_node_duration, A_duration, alias_inputs_duration = [], [], [], []
        #items, n_node, A, alias_inputs = [], [], [], []
        for u_input_duration in inputs_duration:
             n_node_duration.append(len(np.unique(u_input_duration)))
        max_n_node_duration = np.max(n_node_duration)
        for u_input_duration in inputs_duration:
            node_duration = np.unique(u_input_duration)
            items_duration.append(node_duration.tolist() + (max_n_node_duration - len(node_duration)) * [0])
            u_A_duration = np.zeros((max_n_node_duration, max_n_node_duration))
            for i in np.arange(len(u_input_duration) - 1):
            #for i in np.arange(len(u_input) - 1):
                if u_input_duration[i + 1] == 0:
                    break
                u_duration = np.where(node_duration == u_input_duration[i])[0][0]
                v_duration = np.where(node_duration == u_input_duration[i + 1])[0][0]
                u_A_duration[u_duration][v_duration] = 1
            u_sum_in_duration = np.sum(u_A_duration, 0)
            u_sum_in_duration[np.where(u_sum_in_duration == 0)] = 1
            u_A_in_duration = np.divide(u_A_duration, u_sum_in_duration)
            u_sum_out_duration = np.sum(u_A_duration, 1)
            u_sum_out_duration[np.where(u_sum_out_duration == 0)] = 1
            u_A_out_duration = np.divide(u_A_duration.transpose(), u_sum_out_duration)
            u_A_duration = np.concatenate([u_A_in_duration, u_A_out_duration]).transpose()
            A_duration.append(u_A_duration)
            alias_inputs_duration.append([np.where(node_duration == i)[0][0] for i in u_input_duration])
        return alias_inputs_duration, A_duration, items_duration, mask_duration, targets_duration 
    def get_slice2(self, i):
        inputs_interval, mask_interval, targets_interval = self.inputs_interval[i], self.mask_interval[i], self.targets_interval[i]
        items_interval, n_node_interval, A_interval, alias_inputs_interval = [], [], [], []
        #items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs_interval:
             n_node_interval.append(len(np.unique(u_input)))
        max_n_node = np.max(n_node_interval)
        for u_input in inputs_interval:
            node = np.unique(u_input)
            items_interval.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
            #for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1
            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A_interval.append(u_A)
            alias_inputs_interval.append([np.where(node == i)[0][0] for i in u_input])  
        return alias_inputs_interval, A_interval, items_interval, mask_interval, targets_interval
