#!/usr/bin/env python
# coding: utf-8
# Author: Gourang Gaurav

import numpy as np
import random


class DecisionTree:

    def __init__(self, split_measure='entropy', min_impurity_split=0.40,
                 no_splits=2, min_samples_split=2, max_depth=None,
                 max_features=None, random_state=None, print_flag=False):

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        self.tree = {}
        self.node_id = 0  # check for removal from the object

        self.split_measure = split_measure.lower()
        self.min_impurity_split = float(min_impurity_split)
        self.no_splits = int(no_splits)
        self.min_samples_split = int(min_samples_split)
        if max_depth is None:
            self.max_depth = float('inf')
        else:
            self.max_depth = int(max_depth)
        self.max_features = max_features

        self.print_flag = print_flag

    def _is_categorical(self, xi):
        if issubclass(xi.dtype.type, np.character):
            return True
        elif issubclass(xi.dtype.type, np.integer):
            if len(np.unique(xi)) / len(xi) < 0.2:
                return True
        return False

    def _calculate_node_measure(self, y_train):
        tot_no_records = len(y_train)
        prob_ci = []
        if tot_no_records > 0:
            for ci_cnt in np.unique(y_train, return_counts=True)[1]:
                prob_ci.append(ci_cnt / tot_no_records)

        prob_ci = np.array(prob_ci)
        if self.split_measure == 'entropy':
            return -np.sum(prob_ci * np.log2(prob_ci))
        elif self.split_measure == 'gini':
            return 1 - np.sum(prob_ci * prob_ci)

    def _calculate_split_measure(self, *args):
        total_len = 0
        split_entropy = 0.0
        for arg in args:
            arg_len = len(arg)
            total_len += arg_len
            split_entropy += len(arg) * self._calculate_node_measure(arg)
        return split_entropy / total_len

    def _get_split_node(self, x_train, y_train):
        min_ent = float('inf')
        no_features = len(x_train[0])
        features = range(no_features)
        if self.max_features in {'auto', 'sqrt', 'AUTO', 'SQRT'}:
            avail_features = random.sample(features,
                                           int(np.ceil(np.sqrt(no_features))))
        elif self.max_features is None:
            avail_features = features
        elif type(self.max_features) is int:
            avail_features = random.sample(features,
                                           min(self.max_features, no_features))
        elif type(self.max_features) is float:
            avail_features = random.sample(
                features,
                min(int(self.max_features*no_features), no_features)
            )

        for xi in avail_features:
            if not self._is_categorical(x_train[:, xi]):
                y_train_splits = []
                percentiles = np.linspace(0, 100, self.no_splits, endpoint=False)[1:]
                cur_x_train = x_train[:, xi]
                cur_y_train = y_train
                percentile_values = np.percentile(x_train[:, xi], percentiles)
                for p in range(self.no_splits - 1):
                    cond = cur_x_train < percentile_values[p]
                    y_train_splits.append(cur_y_train[cond])
                    cur_x_train = cur_x_train[~cond]
                    cur_y_train = cur_y_train[~cond]
                y_train_splits.append(cur_y_train)
                split_entropy = self._calculate_split_measure(*y_train_splits)
                if split_entropy < min_ent:
                    min_ent = split_entropy
                    split_node = (self.node_id, {'leaf': False, 'class': np.argmax(np.bincount(y_train)),
                                                 'attr': xi, 'val': percentile_values,
                                                 'childs': np.array([None] * self.no_splits)})
            else:
                print("Entered categorical")
                # split_entropy = _calculate_split_entropy()
                # if split_entropy < min_ent:
                #     min_ent = split_entropy
                #     split_atr = xi
        return split_node

    def _generate_tree(self, x_train, y_train, parent=None, node_inp_path=None):
        if self._calculate_node_measure(y_train) < self.min_impurity_split:
            clas = np.argmax(np.bincount(y_train))
            if self.node_id > 0:
                self.tree.update({self.node_id: {'leaf': True, 'class': clas, 'attr': None,
                                                 'val': None, 'childs': None,
                                                 'depth': self.tree[parent]['depth'] + 1}})
                self.tree[parent]['childs'][node_inp_path] = self.node_id
            else:
                self.tree[self.node_id]['depth'] = 0
            self.node_id += 1
            return False

        cur_node_id, split_details = self._get_split_node(x_train, y_train)
        if cur_node_id > 0:
            self.tree[parent]['childs'][node_inp_path] = cur_node_id
            split_details['depth'] = self.tree[parent]['depth'] + 1
        else:
            split_details['depth'] = 0
        self.tree.update({cur_node_id: split_details})
        self.node_id += 1

        split_attr = split_details['attr']
        split_cond_values = split_details['val']
        len_cond_values = len(split_cond_values)
        if (split_details['depth'] < self.max_depth and
                len(x_train) >= self.min_samples_split):
            if len(split_cond_values) != 0:
                # numeric attribute
                cond = x_train[:, split_attr] < split_cond_values[0]
                for i in range(len_cond_values):
                    if cond.any():
                        try:
                            self._generate_tree(x_train[cond, :], y_train[cond],
                                                cur_node_id, i)
                        except RecursionError:
                            # maximum recursion depth exceeded while calling a Python object
                            # this node will be considered leaf node for test data
                            pass

                    if i != (len_cond_values - 1):
                        cond = ((x_train[:, split_attr] >= split_cond_values[i])
                                & (x_train[:, split_attr] < split_cond_values[i + 1]))

                cond = x_train[:, split_attr] >= split_cond_values[len_cond_values - 1]
                if cond.any():
                    try:
                        self._generate_tree(x_train[cond, :], y_train[cond], cur_node_id,
                                            len_cond_values)
                    except RecursionError:
                        # maximum recursion depth exceeded while calling a Python object
                        # this node will be considered leaf node for test data
                        pass

            else:
                print("Entered Categorical")
        else:
            self.tree[cur_node_id]['leaf'] = True

    def learn(self, x_train, y_train):
        self._generate_tree(x_train, y_train)
        # if self.print_flag:
            # print("#Nodes: {0}".format(len(self.tree)), end='; ')
        return self.tree
        # with open('tree_dict_dump.txt', 'w') as f:
        #     for i in self.tree.items():
        #         f.write(str(i) + '\n')

    def classify(self, x_test):
        results = []
        len_data = len(x_test)
        for i in range(len_data):
            node = self.tree[0]
            while not node['leaf']:
                cond_vals = node['val']
                len_cond_vals = len(cond_vals)
                test_attr_value = x_test[i, node['attr']]
                leaf_node = False
                if len_cond_vals != 0:
                    for j in range(len_cond_vals):
                        if test_attr_value < cond_vals[j]:
                            node_id = node['childs'][j]
                            if node_id is not None:
                                node = self.tree[node_id]
                                break
                            else:
                                leaf_node = True
                                break
                    if leaf_node:
                        break
                    if (j == len_cond_vals - 1) and (test_attr_value >= cond_vals[len_cond_vals - 1]):
                        node_id = node['childs'][len_cond_vals]
                        if node_id is not None:
                            node = self.tree[node_id]
                        else:
                            break
                else:
                    print("entered categorical field")
                    # do later

            results.append(node['class'])
        return results
