import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def metric_val(input_vector, matrix_to_search, n_points_in_train_val):
    n_steps = int(np.ceil(matrix_to_search.shape[0] / n_points_in_train_val))

    curr_point = matrix_to_search.shape[0]
    min_dist = 100500
    for i_step in range(n_steps):
        jump = min(n_points_in_train_val, matrix_to_search.shape[0]-curr_point)
        curr_vector = matrix_to_search[curr_point-jump:curr_point].mean(0).reshape(1, -1)
        curr_dist = euclidean_distances(curr_vector, input_vector)
        if curr_dist < min_dist:
            min_dist = curr_dist
            min_point = curr_point

        curr_point -= n_points_in_train_val

    return (min_point, min_dist)


def metric_val(input_vector, matrix_to_search, n_points_in_train_val):
    n_steps = int(np.ceil(matrix_to_search.shape[0] / n_points_in_train_val)) - 1

    curr_point = matrix_to_search.shape[0] - 1
    min_dist = 100500

    for i_step in range(n_steps):

        jump = min(n_points_in_train_val, curr_point)

        curr_vector = np.median(matrix_to_search[curr_point - jump:curr_point], 0).reshape(1, -1)
        curr_dist = euclidean_distances(curr_vector, input_vector)

        if curr_dist < min_dist:
            min_dist = curr_dist
            min_point = curr_point

        curr_point -= n_points_in_train_val

    return (min_point, min_dist)
