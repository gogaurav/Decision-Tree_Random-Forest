import numpy as np
from copy import deepcopy


def cross_validation(estimator, x, y, k=10, metrics="accuracy"):
    print("{0}-fold Cross-Validation:".format(k))
    print(("Hyperparameters used- Split Measure: {0}, min_impurity_split"
           ": {1}, No_Splits: {2}, Min_Samples_Split: {3}"
           ).format(estimator.split_measure, estimator.min_impurity_split,
                    estimator.no_splits, estimator.min_samples_split))
    np.random.seed(0)
    len_data = len(x)
    fold_size = np.ceil(len_data / k)
    permut = np.random.permutation(np.arange(len_data))
    metrics_values = []
    for i in range(k):
        print("Iteration {0}".format(i+1), end=' - ')
        val_idx = permut[int(i * fold_size): int((i + 1) * fold_size)]
        val_idx = val_idx
        mask = np.ones(len_data, dtype='bool')
        mask[val_idx] = False
        x_train = x[mask, :]
        y_train = y[mask]
        # print(np.mean(y_train))
        x_val = x[~mask, :]
        y_val = y[~mask]
        cur_estimator = deepcopy(estimator)
        cur_estimator.learn(x_train, y_train)
        preds = cur_estimator.classify(x_val)
        if metrics == "accuracy":
            metrics_score = np.mean(y_val == preds)
            print("{0}: {1}".format(metrics, metrics_score))
            metrics_values.append(metrics_score)
        del cur_estimator

    return metrics_values

# def cross_validation(estimator_class, data, K=10, metrics="accuracy"):
#     metrics_values = []
#     for j in range(K):
#         training_set = [x for i, x in enumerate(data) if i % K != j]
#         test_set = np.array([x for i, x in enumerate(data) if i % K == j])
#         training_set = np.array(training_set)
#         x_train = training_set[:, :-1].astype(float)
#         y_train = training_set[:, -1].astype(int)
#         x_val = test_set[:, :-1].astype(float)
#         y_val = test_set[:, -1].astype(int)
#         print(np.mean(y_train))
#         estimator = estimator_class(split_measure='entropy',
#                                     min_impurity_split=0.23,
#                                     no_splits=3, print_flag=True)
#         estimator.learn(x_train, y_train)
#         preds = estimator.classify(x_val)
#         print("hello")
#         if metrics == "accuracy":
#             print("Training accuracy: {0}".format(np.mean(y_train == estimator.classify(x_train))))
#             metrics_values.append(np.mean(y_val == preds))
#             print(metrics_values)
#     return metrics_values


    # def cross_validation(estimator_class, x, y, K=10, metrics="accuracy"):
#     length = len(x)
#     metrics_values = []
#     for j in range(K):
#         mask = np.array([True if i % K != j else False for i in range(length)])
#         x_train = x[mask, :-1]
#         y_train = y[mask]
#         print(np.mean(y_train))
#         x_val = x[~mask, :-1]
#         y_val = y[~mask]
#         estimator = estimator_class(split_measure='entropy',
#                                     min_impurity_split=0.23,
#                                     no_splits=3, print_flag=True)
#         estimator.learn(x_train, y_train)
#         preds = estimator.classify(x_val)
#         print("hello")
#         if metrics == "accuracy":
#             print("Training accuracy: {0}".format(np.mean(y_train == estimator.classify(x_train))))
#             metrics_values.append(np.mean(y_val == preds))
#             print(metrics_values)
#     return metrics_values
