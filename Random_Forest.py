#!/usr/bin/env python
# coding: utf-8

import numpy as np
from Decision_Tree import DecisionTree
import Cross_Validation as cv
import random


class RandomForestClassifier:
    def __init__(self, no_estimators=10, split_measure='entropy',
                 min_impurity_split=0.40, max_depth=None,
                 min_samples_split=2, no_splits=2, max_features='auto',
                 bootstrap=True, random_state=None, print_flag=True):

        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

        self.fitted_trees = list()
        self.no_estimators = int(no_estimators)
        self.bootstrap = bootstrap

        self.split_measure = split_measure
        self.min_impurity_split = min_impurity_split
        self.no_splits = no_splits
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.print_flag = print_flag

    def learn(self, x_train, y_train):
        len_train_data = len(x_train)
        for i in range(self.no_estimators):
            if self.bootstrap:
                idx = np.random.randint(0, len_train_data, len_train_data)
                x_train_tree = x_train[idx]
                y_train_tree = y_train[idx]  # test here please
            else:
                x_train_tree = x_train
                y_train_tree = y_train

            dt = DecisionTree(
                random_state=None, split_measure=self.split_measure,
                min_impurity_split=self.min_impurity_split, max_depth=self.max_depth,
                min_samples_split=self.min_samples_split, no_splits=self.no_splits,
                max_features=self.max_features, print_flag=self.print_flag
            )
            dt.learn(x_train_tree, y_train_tree)
            self.fitted_trees.append(dt)

    def classify(self, x_test):
        trees_output = []
        for i in range(self.no_estimators):
            trees_output.append(self.fitted_trees[i].classify(x_test))

        trees_output = np.array(trees_output)
        preds = []
        for i in range(len(trees_output[0])):
            preds.append(np.argmax(np.bincount(trees_output[:, i])))

        return preds
